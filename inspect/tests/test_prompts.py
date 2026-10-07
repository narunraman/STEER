"""Prompt sets: the default 2025 open-model strings, custom prompt files, and their record in logs."""

import json

import pytest
from conftest import mock_model
from inspect_ai import eval as inspect_eval

from steer_bench import steer_me
from steer_bench.prompts import DEFAULT_PROMPTS, PROMPT_SETS, load_prompts, prompts_hash
from steer_bench.tasks import build_samples

P = PROMPT_SETS["open2025"]


def test_default_is_open2025_verbatim():
    # strings recovered from the 2025 open-model run pickles (steer-drafts parity.md)
    assert DEFAULT_PROMPTS == "open2025"
    assert P["reasoning"] == "\nPlease reason step by step, and put your final answer within \\boxed{}."
    assert P["answer"] == ("\nAnswer by writing the option letter corresponding to the correct option. "
                           "RESPOND WITH ONLY A SINGLE LETTER.\nA:")


def _turns(log, sample_id):
    s = next(x for x in log.samples if x.id == sample_id)
    return [m.text for m in s.messages if m.role == "user"]


@pytest.mark.parametrize("fmt", ["shown", "hidden"])
def test_multi_part_turns_exact(fmt, data_root, tmp_path):
    [log] = inspect_eval(steer_me(format=fmt, element="multi", data_dir=str(data_root)),
                         model=mock_model(lambda last, all_: ("B", {"B": 1.0})),
                         log_dir=str(tmp_path), display="none")
    t = _turns(log, "multi/m0")
    R, A = P["reasoning"], P["answer"]
    if fmt == "shown":
        assert t == ["Q: multi m0 part0\nA. no\nB. yes\n" + R, A,
                     "Q: multi m0 part1\nC. 1\nD. 2\nE. 3\n" + R, A]
    else:
        assert t == ["Q: multi m0 part0\n" + R, "A. no\nB. yes\n" + A,
                     "Q: multi m0 part1\n" + R, "C. 1\nD. 2\nE. 3\n" + A]


def test_custom_prompts_file(data_root, tmp_path):
    f = tmp_path / "terse.json"
    f.write_text(json.dumps({"answer": "\nLetter only:", "hidden_options_newline": True}))
    name, strings = load_prompts(str(f))
    assert name == "terse" and strings["reasoning"] == P["reasoning"]  # missing keys: open2025
    assert prompts_hash(strings) != prompts_hash(P)
    [s] = [x for x in build_samples("steer_me", element="tiny", prompts=str(f), data_dir=str(data_root))
           if x.id == "tiny/q0"]
    assert s.input == "Q: tiny q0\nA. x\nB. y\nC. z\n\nLetter only:"
    [log] = inspect_eval(steer_me(element="tiny", format="hidden", prompts=str(f), data_dir=str(data_root)),
                         model=mock_model(lambda last, all_: ("A", {"A": 1.0})),
                         log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    assert log.eval.metadata["prompts"] == "terse"
    assert log.eval.metadata["prompts_hash"] == prompts_hash(strings)
    assert _turns(log, "tiny/q0")[1] == "\nA. x\nB. y\nC. z\n\nLetter only:"


def test_bad_prompts(tmp_path):
    with pytest.raises(ValueError):
        load_prompts("paper")  # only the 2025 set ships
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"answr": "x"}))
    with pytest.raises(ValueError):
        load_prompts(str(f))
