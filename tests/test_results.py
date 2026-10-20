"""The per-response rows: one per sample, summing to the cells exporter's numbers, under the public model name only."""

import json

import pandas as pd
from test_export import logs  # noqa: F401 - the shared fixture

from steer_bench.results import KEY, check_against_cells, main, rows_from_logs


def test_rows_sum_to_the_cells(logs):  # noqa: F811
    d, out = logs
    rows = rows_from_logs([str(d)], "test-model")
    assert len(rows) == sum(len(x.samples) for x in out)
    assert check_against_cells(rows, [str(d)]) == []
    assert len({tuple(r[k] for k in KEY) for r in rows}) == len(rows)


def test_rows_name_the_model_as_given_and_nothing_else(logs):  # noqa: F811
    d, out = logs
    rows = rows_from_logs([str(d)], "test-model")
    served = {x.eval.model for x in out}
    assert {r["model"] for r in rows} == {"test-model"}
    text = json.dumps([{k: v for k, v in r.items() if k != "messages"} for r in rows], default=str)
    assert not any(m in text for m in served)


def test_answers_and_probabilities(logs):  # noqa: F811
    d, _ = logs
    rows = rows_from_logs([str(d)], "test-model")
    for r in rows:
        if r["format"] != "free" and r["answered"]:
            p = r["parts"][0]
            assert p["letter"] and p["chosen_index"] is not None
            if p["option_probs"] is not None:
                assert len(p["option_probs"]) == len(p["options"])


def test_cli_writes_parquet_with_the_fixed_schema(logs, tmp_path):  # noqa: F811
    import pyarrow.parquet as pq

    from steer_bench.results import schema

    d, _ = logs
    f = tmp_path / "rows.parquet"
    assert main([str(d), "--model", "test-model", "--out", str(f)]) == 0
    assert len(pd.read_parquet(f)) > 0
    assert pq.read_schema(f).remove_metadata() == schema()
