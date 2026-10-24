"""`steer-bench run`: several models through eval_set, then a card per model and an index."""

import pandas as pd
import pytest

from steer_bench.cli import main as cli
from steer_bench.report import write_cards
from steer_bench.run import model_args_for, parse_args_list, parse_models


def test_models_from_a_file_or_a_list(tmp_path):
    f = tmp_path / "models.txt"
    f.write_text("# shortlist\nsllm/qwen-3-8b\n\nopenai/gpt-4o-mini  # an API model\n")
    assert parse_models(str(f)) == ["sllm/qwen-3-8b", "openai/gpt-4o-mini"]
    assert parse_models("a/x, b/y") == ["a/x", "b/y"]
    with pytest.raises(ValueError):
        parse_models(" , ")


def test_options_are_parsed_like_inspects():
    assert parse_args_list(["format=mc", "limit=5", "keep=false"], "-T") == {"format": "mc", "limit": 5, "keep": False}
    with pytest.raises(ValueError, match="key=value"):
        parse_args_list(["format"], "-T")


def _cells(run_id, norm):
    rows = []
    for el in ["add_sub", "independence", "enforceability"]:
        rows.append({"run_id": run_id, "element": el, "question_format": "shown_options", "car": False,
                     "adaptation": "shots=0;expl=0;prompts=open2025@x", "domain": "d", "type": "t",
                     "perspective": None, "difficulty": 1, "n_rows": 10, "n_no_answer": 0, "n": 10,
                     "n_correct": 5, "sum_norm": norm * 10, "n_prob": 0})
    return rows


def test_index_ranks_the_models_and_links_their_cards(tmp_path):
    cells = pd.DataFrame(_cells("inspect/sllm/weak-model", 0.1) + _cells("inspect/openai/strong-model", 0.6))
    index = write_cards(cells, None, "steer", tmp_path / "cards")
    page = index.read_text()
    assert (tmp_path / "cards" / "sllm__weak-model.html").exists()
    assert (tmp_path / "cards" / "openai__strong-model.html").exists()
    assert page.index("strong-model") < page.index("weak-model")  # best first
    assert "href='openai__strong-model.html'" in page and "STEER: 2 models" in page
    assert "Foundations" in page  # a column per setting


def test_run_evaluates_every_model_and_writes_the_cards(data_root, tmp_path, capsys):
    out, logs = tmp_path / "cards", tmp_path / "logs"
    code = cli(["run", "steer_me", "--models", "mockllm/model,mockllm/other", "-T", f"data_dir={data_root}",
                "-T", "element=tiny", "--log-dir", str(logs), "--out", str(out)])
    assert code == 0, capsys.readouterr()
    assert sorted(p.name for p in out.iterdir()) == ["index.html", "mockllm__model.html", "mockllm__other.html"]
    assert "wrote 2 report cards" in capsys.readouterr().out


def test_slurm_llm_jobs_are_cancelled_when_their_model_is_done_unless_kept():
    models = ["sllm/qwen-3-8b", "openai/gpt-4o-mini"]
    assert model_args_for(models, {}) == [{"keep": False}, {}]  # other providers do not know keep
    assert model_args_for(models, {"keep": True, "cluster": "c"}) == [{"keep": True, "cluster": "c"},
                                                                      {"keep": True, "cluster": "c"}]
