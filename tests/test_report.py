"""Report cards from logs and from converted cells."""

import math

import pytest
from conftest import mock_model
from inspect_ai import eval as inspect_eval
from test_eval import pairs_policy, tiny_policy

from steer_bench import steer_me
from steer_bench.cli import main as cli
from steer_bench.export import cells_from_logs
from steer_bench.export import main as export_main
from steer_bench.report import build_report, cell_colors, format_key, load_inputs, load_taxonomy


@pytest.fixture(scope="module")
def logdir(data_root, tmp_path_factory):
    d = tmp_path_factory.mktemp("rlogs")
    for kw, pol in [({"element": "tiny"}, tiny_policy), ({"element": "tiny", "format": "none"}, tiny_policy),
                    ({"element": "pairs"}, pairs_policy)]:
        [log] = inspect_eval(steer_me(data_dir=str(data_root), **kw), model=mock_model(pol),
                             log_dir=str(d), display="none")
        assert log.status == "success"
    return d


def test_taxonomy_bundled():
    t = load_taxonomy()
    assert "copy" in t["source"].lower()
    steer = t["benchmarks"]["steer"]["settings"]
    assert steer[0]["name"] == "Foundations" and steer[0]["modules"][0]["elements"][0]["element"] == "add_sub"
    names = {e["element"] for s in steer for m in s["modules"] for e in m["elements"]}
    assert {"enforceability", "independence"} <= names
    assert len(t["benchmarks"]["steer_me"]["settings"]) == 5


def test_format_key():
    a = "shots=0;expl={};explprompt=0;reps=0;retries=0;decoding=logprobs;prompts=open2025@x"
    assert format_key("shown_options", False, a.format(0)) == "mc"
    assert format_key("shown_options", False, a.format(3)) == "shown"
    assert format_key("hidden_options", False, a.format(3)) == "hidden"
    assert format_key("shown_options", True, a.format(0)) == "none"
    assert format_key("no_options", False, a.format(3)) == "free"
    assert format_key("shown_options", False, a.format(0).replace("shots=0", "shots=5")) == "mc, 5-shot"


def test_report_matches_task_metrics(logdir, data_root, tmp_path):
    cells, models, bench = load_inputs([str(logdir)])
    assert bench == "steer_me"
    r = build_report(cells, models, bench)
    assert r["formats"] == ["mc", "none"] and r["prompts"][0].startswith("open2025@")
    # tiny (hand-computed in test_eval): exact match 2/3, normalized accuracy (1 - 0.5 + 1)/3
    m = r["per"][("tiny", "mc")]
    assert m["exact_match"] == pytest.approx(2 / 3)
    assert m["normalized_accuracy"] == pytest.approx(0.5)
    assert not math.isnan(m["ece"])
    assert math.isnan(r["per"][("pairs", "mc")]["ece"])  # consistency groups: no calibration
    # elements not in the taxonomy are listed under "Other"
    assert r["settings"][-1]["name"].startswith("Other")
    # macro average is the mean of the element scores
    s = r["summary"]["mc"]
    assert s["macro_normalized_accuracy"] == pytest.approx(
        (r["per"][("tiny", "mc")]["normalized_accuracy"] + r["per"][("pairs", "mc")]["normalized_accuracy"]) / 2)


def test_report_from_cells_equals_report_from_logs(logdir, tmp_path):
    export_main([str(logdir), "--out", str(tmp_path / "cells")])
    a = build_report(*load_inputs([str(logdir)]))
    b = build_report(*load_inputs([str(tmp_path / "cells")]))
    assert a["per"].keys() == b["per"].keys()
    for k in a["per"]:
        for key in ("exact_match", "normalized_accuracy", "ece", "brier", "epa"):
            x, y = a["per"][k][key], b["per"][k][key]
            assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y)


def test_cli_writes_html_and_md(logdir, tmp_path):
    h, m = tmp_path / "r.html", tmp_path / "r.md"
    assert cli(["report", str(logdir), "--out", str(h), "--out", str(m)]) == 0
    page = h.read_text()
    assert page.startswith("<!doctype html>") and "mockllm/model on STEER-ME" in page
    assert "<link" not in page and "<script" not in page and "http" not in page.split("<main>")[0]
    assert "| tiny |" in m.read_text()


def test_report_real_taxonomy_order():
    import pandas as pd
    rows = []
    for el in ["enforceability", "add_sub", "independence"]:  # deliberately out of order
        rows.append({"run_id": "inspect/m", "element": el, "question_format": "shown_options", "car": False,
                     "adaptation": "shots=0;expl=0;prompts=open2025@x", "domain": "d", "type": "t",
                     "perspective": None, "difficulty": 1, "n_rows": 10, "n_no_answer": 0, "n": 10,
                     "n_correct": 5, "sum_norm": 0.3, "n_prob": 0})
    r = build_report(pd.DataFrame(rows), None, "steer")
    order = [e["element"] for s in r["settings"] for m in s["modules"] for e in m["elements"]]
    assert order == ["add_sub", "independence", "enforceability"]
    assert r["settings"][0]["name"] == "Foundations"


def test_colors_diverge_from_neutral():
    assert cell_colors(0.0, "light")[0] == "#f0efec"
    assert cell_colors(1.0, "light")[0] == "#2a78d6" and cell_colors(-1.0, "light")[0] == "#d03b33"
    assert cell_colors(float("nan"), "light")[0] == "transparent"


def test_mixed_benchmarks_need_choice(data_root, tmp_path):
    from steer_bench import steer
    for t in (steer(element="tiny", data_dir=str(data_root)), steer_me(element="tiny", data_dir=str(data_root))):
        inspect_eval(t, model=mock_model(tiny_policy), log_dir=str(tmp_path), display="none")
    with pytest.raises(ValueError, match="several benchmarks"):
        load_inputs([str(tmp_path)])
    c, _, b = load_inputs([str(tmp_path)], "steer")
    assert b == "steer" and c["n_rows"].sum() == 3
    assert cells_from_logs([str(tmp_path)])[0]["n_rows"].sum() == 6
