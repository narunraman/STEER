"""The .eval -> scored-cells converter, checked against the task's own metrics."""

import json
import math

import pandas as pd
import pytest
from conftest import mock_model
from inspect_ai import eval as inspect_eval
from test_eval import pairs_policy, tiny_policy

from steer_bench import steer_me
from steer_bench.export import KEYS, STAT_COLS, cells_from_logs, main
from steer_bench.prompts import PROMPT_SETS, prompts_hash


def _metrics_from_cells(c: pd.DataFrame, v: str = "cond") -> dict:
    """steer-scoring metrics.add_metrics, with calibration divided by n_prob."""
    s = c[STAT_COLS + ["n_prob"]].sum()
    ece = sum(abs(s[f"ece_{v}_acc{i}"] - s[f"ece_{v}_conf{i}"]) for i in range(10))
    return {
        "exact_match": s["n_correct"] / s["n"],
        "normalized_accuracy": s["sum_norm"] / s["n"],
        "no_answer_rate": s["n_no_answer"] / s["n_rows"],
        "epa": s[f"sum_p_correct_{v}"] / s["n_prob"] if s["n_prob"] else math.nan,
        "brier": s[f"sum_brier_{v}"] / s["n_prob"] if s["n_prob"] else math.nan,
        "ece": ece / s["n_prob"] if s["n_prob"] else math.nan,
    }


@pytest.fixture(scope="module")
def logs(data_root, tmp_path_factory):
    d = tmp_path_factory.mktemp("logs")
    out = []
    out += inspect_eval(steer_me(element="tiny", data_dir=str(data_root)), model=mock_model(tiny_policy),
                        log_dir=str(d), display="none")
    out += inspect_eval(steer_me(element="tiny", format="none", data_dir=str(data_root)),
                        model=mock_model(tiny_policy), log_dir=str(d), display="none")
    out += inspect_eval(steer_me(element="pairs", data_dir=str(data_root)), model=mock_model(pairs_policy),
                        log_dir=str(d), display="none")
    assert all(x.status == "success" for x in out)
    return d, out


def test_cells_schema_and_counts(logs):
    d, _ = logs
    cells, models, prov = cells_from_logs([str(d)])
    assert list(cells.columns) == KEYS + STAT_COLS + ["n_prob"]
    assert len(prov["logs"]) == 3 and not prov["skipped"]
    assert set(cells["run_id"]) == {"inspect/mockllm/model"}
    assert set(zip(cells["question_format"], cells["car"])) == {("shown_options", False), ("shown_options", True)}
    h = prompts_hash(PROMPT_SETS["open2025"])
    assert set(cells["adaptation"]) == {
        f"shots=0;expl=0;explprompt=0;reps=0;retries=0;decoding=logprobs;prompts=open2025@{h}"}
    assert models.loc[0, "prompts"] == f"open2025@{h}"
    assert cells["n_rows"].sum() == 3 + 3 + 3
    # additive: one cell per (element, format, car, domain, type, perspective, difficulty)
    assert not cells.duplicated(KEYS).any()
    assert cells["n_rows"].dtype == "int32" and cells["sum_norm"].dtype == "float64"
    assert models.loc[0, "has_logprobs"] and models.loc[0, "n_rows"] == 9


def test_cells_reproduce_task_metrics(logs):
    d, out = logs
    cells, _, _ = cells_from_logs([str(d)])
    for log in out:
        el = log.samples[0].metadata["element"]
        car = log.eval.task_args.get("format") == "none"
        c = cells[(cells["element"] == el) & (cells["car"] == car)]
        got = _metrics_from_cells(c)
        want = {k.split("/")[1]: v.value for k, v in log.results.scores[0].metrics.items() if k.startswith("all/")}
        for k, w in want.items():
            if math.isnan(w):
                assert math.isnan(got[k]), k
            else:
                assert got[k] == pytest.approx(w), k


def test_tiny_cell_values(logs):
    d, _ = logs
    cells, _, _ = cells_from_logs([str(d)])
    c = cells[(cells["element"] == "tiny") & (~cells["car"])].iloc[0]
    assert (c["n_rows"], c["n"], c["n_correct"], c["n_prob"], c["n_top_correct"]) == (3, 3, 2, 3, 2)
    assert c["sum_norm"] == pytest.approx(1.5)
    assert c["sum_p_correct_cond"] == pytest.approx(7 / 9 + 4 / 9 + 0.95)
    assert (c["ece_cond_n5"], c["ece_cond_n7"], c["ece_cond_n9"]) == (1, 1, 1)
    assert c["ece_cond_conf7"] == pytest.approx(7 / 9) and c["ece_cond_acc5"] == 0


def test_cli_writes_outputs(logs, tmp_path):
    d, _ = logs
    assert main([str(d), "--out", str(tmp_path / "out")]) == 0
    cells = pd.read_parquet(tmp_path / "out" / "cells.parquet")
    assert len(cells) > 0
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert prov["n_cells"] == len(cells)
    assert (tmp_path / "out" / "models.parquet").exists()


def test_stat_cols_match_steer_scoring():
    agg = pytest.importorskip("steer_scoring.aggregate")
    assert STAT_COLS == agg.STAT_COLS and KEYS == agg.KEYS


def test_duplicate_samples_in_a_log_count_once(logs, monkeypatch):
    """eval-retry logs can contain a sample twice; the converter keeps one copy."""
    import inspect_ai.log as L
    d, _ = logs
    expected = cells_from_logs([str(d)])[0]["n_rows"].sum()
    original = L.read_eval_log_samples

    def twice(*a, **k):
        for s in original(*a, **k):
            yield s
            yield s

    monkeypatch.setattr(L, "read_eval_log_samples", twice)
    assert cells_from_logs([str(d)])[0]["n_rows"].sum() == expected
