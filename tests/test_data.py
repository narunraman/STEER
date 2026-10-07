from collections import Counter

import pytest
from conftest import STAGED_ROOT

from steer_bench.data import NONE_OPTION, group_rows, load_rows, none_option_selection
from steer_bench.prompts import PROMPT_SETS

ANSWER = PROMPT_SETS["open2025"]["answer"]
from steer_bench.tasks import build_samples


def _elements(samples):
    return Counter(s.metadata["element"] for s in samples)


def test_filter_element(data_root):
    s = build_samples("steer_me", element="tiny", data_dir=str(data_root))
    assert _elements(s) == {"tiny": 3}
    s = build_samples("steer_me", element="tiny,words", data_dir=str(data_root))
    assert _elements(s) == {"tiny": 3, "words": 9}
    s = build_samples("steer_me", element=["tiny", "pairs"], data_dir=str(data_root))
    assert _elements(s) == {"tiny": 3, "pairs": 3}
    with pytest.raises(ValueError, match="unknown"):
        build_samples("steer_me", element="nope", data_dir=str(data_root))


def test_filter_module_and_setting(data_root):
    by_slug = build_samples("steer_me", module="demand", data_dir=str(data_root))
    by_name = build_samples("steer_me", module="Comparative Statics of Demand", data_dir=str(data_root))
    assert _elements(by_slug) == _elements(by_name) == {"tiny": 3, "words": 9}
    s = build_samples("steer_me", setting="single_agent", data_dir=str(data_root))
    assert _elements(s) == {"pairs": 3, "multi": 2}
    s = build_samples("steer_me", setting="Single Agent", module="axioms", data_dir=str(data_root))
    assert _elements(s) == {"pairs": 3}
    assert {x.metadata["setting"] for x in s} == {"single_agent"}
    assert {x.metadata["module"] for x in s} == {"axioms"}


def test_held_elements_excluded_unless_named(data_root):
    assert "sc_axioms" not in _elements(build_samples("steer", data_dir=str(data_root)))
    assert _elements(build_samples("steer", element="sc_axioms", data_dir=str(data_root))) == {"sc_axioms": 1}
    assert _elements(build_samples("steer", include_held=True, data_dir=str(data_root)))["sc_axioms"] == 1
    # held only in STEER; enforceability and tfp_shocks were released in October 2026
    assert "sc_axioms" in _elements(build_samples("steer_me", data_dir=str(data_root)))
    from steer_bench.data import HELD_ELEMENTS
    assert not {"enforceability", "tfp_shocks"} & (HELD_ELEMENTS["steer"] | HELD_ELEMENTS["steer_me"])


def test_data_dir_env_and_benchmark_subdir(data_root, monkeypatch):
    monkeypatch.setenv("STEER_BENCH_DATA_DIR", str(data_root))
    assert len(build_samples("steer", element="tiny")) == 3   # finds STEER/ case-insensitively
    assert len(build_samples("steer_me", element="tiny", data_dir=str(data_root / "steer_me"))) == 3


def test_groups_are_packed(data_root):
    s = {x.id: x for x in build_samples("steer_me", element="pairs,multi", data_dir=str(data_root))}
    assert set(s) == {"pairs/g0", "pairs/g1", "pairs/g2", "multi/m0", "multi/m1"}
    assert [p["question_id"] for p in s["pairs/g0"].metadata["parts"]] == ["pairs/g0/0", "pairs/g0/1"]
    assert s["multi/m0"].target == "B,D"          # letters continue: part1 options are C-E
    assert s["pairs/g0"].target == "same:01,12"


def test_mc_prompt_verbatim(data_root):
    [s] = [x for x in build_samples("steer_me", element="tiny", data_dir=str(data_root)) if x.id == "tiny/q0"]
    assert s.input == (
        "Q: tiny q0\nA. x\nB. y\nC. z\n"
        "\nAnswer by writing the option letter corresponding to the correct option. "
        "RESPOND WITH ONLY A SINGLE LETTER.\nA:"
    )
    assert s.input.endswith("\n" + ANSWER)


# ------------------------------------------------------------------ "No other option is correct"
def _none_samples(root, seed=42, **kw):
    return {x.id: x for x in build_samples("steer_me", format="none", seed=seed, data_dir=str(root), **kw)}


def test_none_option_one_in_n_and_reproducible(data_root):
    a = _none_samples(data_root, element="consumer_surplus")
    b = _none_samples(data_root, element="consumer_surplus")
    replaced = {k for k, s in a.items() if s.metadata["none_option_replaces_correct"]}
    assert len(a) == 40 and len(replaced) == 10            # exactly 1 in n (n = 4 options)
    assert replaced == {k for k, s in b.items() if s.metadata["none_option_replaces_correct"]}
    for k, s in a.items():
        [p] = s.metadata["parts"]
        assert p["options"].count(NONE_OPTION) == 1
        assert p["options"] == b[k].metadata["parts"][0]["options"]   # same wrong option too
        j = p["options"].index(NONE_OPTION)
        assert (j == p["correct_index"]) == (k in replaced)
        assert NONE_OPTION in s.input
    c = _none_samples(data_root, seed=7, element="consumer_surplus")
    assert {k for k, s in c.items() if s.metadata["none_option_replaces_correct"]} != replaced


def test_none_option_independent_of_filters(data_root):
    a = _none_samples(data_root, element="consumer_surplus")
    b = _none_samples(data_root, setting="equilibria")
    assert {k: s.metadata["parts"] for k, s in a.items()} == {k: s.metadata["parts"] for k, s in b.items()}


def test_none_option_skips_consistency(data_root):
    s = _none_samples(data_root, element="tiny,pairs")
    assert _elements(s.values()) == {"tiny": 3}


@pytest.mark.skipif(not (STAGED_ROOT / "steer_me").exists(), reason="staged data not available")
def test_none_option_on_staged_consumer_surplus():
    rows = load_rows("steer_me", element="consumer_surplus", data_dir=str(STAGED_ROOT))
    groups = group_rows(rows)
    chosen = none_option_selection(groups, 42)
    assert len(groups) == 1000 and len(chosen) == 250


# ------------------------------------------------------------------ few-shot
def test_few_shot_prefix(data_root):
    s = build_samples("steer_me", element="consumer_surplus", shots=2, data_dir=str(data_root))
    x = next(v for v in s if v.metadata["type"] == "linear" and v.metadata["domain"] == "food")
    prefix = x.metadata["prefix"]
    assert prefix.count("Q: cs example") == 2
    assert prefix.endswith("\n\n")
    assert x.input.startswith(prefix) and x.input[len(prefix):].startswith("Q: cs question")
    # examples are taken from the matching (element, type, domain) cell: linear = odd example ids
    for i in (0, 2, 4):
        assert f"cs example {i}\n" not in prefix
    # each example ends with the answer suffix followed by its correct letter
    assert prefix.count(ANSWER) == 2
    import re
    assert len(re.findall(re.escape(ANSWER) + r" [A-D]\n", prefix)) == 2
    # deterministic
    s2 = build_samples("steer_me", element="consumer_surplus", shots=2, data_dir=str(data_root))
    assert [v.metadata["prefix"] for v in s] == [v.metadata["prefix"] for v in s2]
    # no prefix for shots=0
    assert all(v.metadata["prefix"] == "" for v in build_samples("steer_me", element="tiny", data_dir=str(data_root)))


def test_free_keeps_numeric_answer_graded_only(data_root):
    s = build_samples("steer_me", format="free", data_dir=str(data_root), element="consumer_surplus,words,pairs")
    assert _elements(s) == {"consumer_surplus": 40}
    with pytest.raises(ValueError):
        build_samples("steer_me", format="free", shots=1, data_dir=str(data_root), element="consumer_surplus")
