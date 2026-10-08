"""The published dataset layout, locally (data_dir) and through `datasets` (the Hub loader).

The Hub loader is exercised against local copies laid out exactly like the Hub repositories
(per-element configs in the README header), so it runs without network access.
"""

import os
import re
from collections import Counter
from pathlib import Path

import pytest
from conftest import mock_model
from inspect_ai import eval as inspect_eval

import steer_bench
from steer_bench import data as D
from steer_bench import steer, steer_me
from steer_bench.tasks import build_samples

# The release build that will be uploaded (skipped where it is not available).
RELEASE = Path(os.environ.get("STEER_BENCH_RELEASE",
                              "~/UBC/autosteer-release/2026-10-08-10d5303")).expanduser()


def _elements(rows):
    return Counter(r["element"] for r in rows)


def test_release_layout_via_data_dir(release_root):
    rows = D.load_rows("steer_me", element="tiny,pairs", data_dir=str(release_root))
    assert _elements(rows) == {"tiny": 3, "pairs": 6}
    assert {r["setting"] for r in rows} == {"Consumption Decisions", "Single Agent"}   # columns
    # setting/module filters by name or slug
    by_name = D.load_rows("steer_me", setting="Single Agent", data_dir=str(release_root))
    by_slug = D.load_rows("steer_me", setting="single_agent", data_dir=str(release_root))
    assert _elements(by_name) == _elements(by_slug) == {"pairs": 6, "multi": 4}
    assert _elements(D.load_rows("steer_me", module="utility_axioms", data_dir=str(release_root))) == {"pairs": 6}
    # benchmark directories named steer / steer-me under a parent
    assert len(D.load_rows("steer", element="tiny", data_dir=str(release_root))) == 3
    # few-shot split; an element without one contributes nothing
    shots = D.load_rows("steer_me", "few_shot", data_dir=str(release_root))
    assert set(_elements(shots)) == {"consumer_surplus"}
    s = build_samples("steer_me", element="consumer_surplus", shots=2, data_dir=str(release_root))
    assert all(x.metadata["shots_used"] == 2 for x in s)


def test_hub_loader_on_a_local_repo(release_root):
    repo = str(release_root / "steer-me")
    rows = D.load_rows_hub("steer_me", "test", ["tiny", "multi"], repo=repo)
    assert _elements(rows) == {"tiny": 3, "multi": 4}
    everything = D.load_rows_hub("steer_me", "test", repo=repo)               # default config
    assert _elements(everything) == _elements(D.load_rows("steer_me", include_held=True, data_dir=repo))
    assert D.load_rows_hub("steer_me", "few_shot", ["words"], repo=repo) == []   # no few_shot split
    assert _elements(D.load_rows_hub("steer_me", "test", modules=["Utility Axioms"], repo=repo)) == {"pairs": 6}
    with pytest.raises(ValueError, match="unknown"):
        D.load_rows_hub("steer_me", "test", ["nope"], repo=repo)


def test_hub_loader_uses_the_pinned_revision(monkeypatch):
    calls = []

    def fake_load_dataset(path, name=None, split=None, revision=None):
        calls.append((path, name, split, revision))
        return [{"element": name, "base_id": "b0", "question_text": "q", "options": ["1", "2"],
                 "correct_index": 0, "setting": "S", "module": "M"}]

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setitem(D.HF_REVISIONS, "steer", "0123abcd" * 5)
    monkeypatch.delenv(D.DATA_DIR_ENV, raising=False)
    rows = D.load_rows("steer", element="add_sub,mult_div")
    assert calls == [("narunraman/steer", "add_sub", "test", "0123abcd" * 5),
                     ("narunraman/steer", "mult_div", "test", "0123abcd" * 5)]
    assert [r["element"] for r in rows] == ["add_sub", "mult_div"]
    calls.clear()
    monkeypatch.setitem(D.HF_REVISIONS, "steer_me", None)
    with pytest.warns(UserWarning, match="no pinned revision"):
        D.load_rows("steer_me")
    assert calls == [("narunraman/steer-me", "default", "test", None)]


def test_revisions_pinned_for_a_release_version():
    """A release (version without .dev) must evaluate a fixed dataset commit."""
    if ".dev" in steer_bench.__version__:
        pytest.skip(f"development version {steer_bench.__version__}")
    assert all(D.HF_REVISIONS[b] for b in D.BENCHMARKS), D.HF_REVISIONS


@pytest.mark.skipif(not (RELEASE / "steer").exists(), reason="release build not available")
@pytest.mark.parametrize("bench,task,folder,element", [
    ("steer", steer, "steer", "independence"),         # consistency-graded, two parts
    ("steer_me", steer_me, "steer-me", "consumer_surplus"),
])
def test_release_build_end_to_end(bench, task, folder, element, tmp_path):
    """One element per benchmark from the release build: loader (local and through `datasets`),
    solver and scorer."""
    hub_rows = D.load_rows_hub(bench, "test", [element], repo=str(RELEASE / folder))
    local_rows = D.load_rows(bench, element=element, data_dir=str(RELEASE))
    assert len(hub_rows) == len(local_rows) > 0
    assert {r["question_id"] for r in hub_rows} == {r["question_id"] for r in local_rows}
    def first_offered_letter(last, all_):   # a valid letter for any part (letters continue)
        letter = re.findall(r"(?:^|\n)([A-Z])\. ", last)[0]
        return letter, {letter: 0.7, "Z": 0.3}

    [log] = inspect_eval(task(element=element, data_dir=str(RELEASE)),
                         model=mock_model(first_offered_letter),
                         limit=20, log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    m = {k: v.value for k, v in log.results.scores[0].metrics.items()}
    assert m["all/no_answer_rate"] == 0.0 and 0.0 <= m["all/exact_match"] <= 1.0
