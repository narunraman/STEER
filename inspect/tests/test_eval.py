"""End-to-end runs of the tasks against scripted mockllm models."""

import math
import re

import pytest
from conftest import mock_model, question_index
from inspect_ai import eval as inspect_eval

from steer_bench import steer, steer_me
from steer_bench.data import NONE_OPTION
from steer_bench.prompts import EXPLANATION_SUFFIXES, FREE_ANSWER_SUFFIX, MC_ANSWER_SUFFIX


def run(task, model, tmp_path, **kw):
    [log] = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none", **kw)
    assert log.status == "success", log.error
    return log


def metrics(log):
    return {k: v.value for k, v in log.results.scores[0].metrics.items()}


def always(letter_or_text, top=None):
    return lambda last, all_: (letter_or_text, top)


# ------------------------------------------------------------------ every format runs
@pytest.mark.parametrize("fmt", ["mc", "shown", "hidden", "none", "free"])
def test_each_format_runs(fmt, data_root, tmp_path):
    def policy(last, all_):
        if last.endswith(MC_ANSWER_SUFFIX):
            return "A", {"A": 0.6, "B": 0.3, "C": 0.1}
        if FREE_ANSWER_SUFFIX in last:
            return "Some reasoning. ANSWER: 10.123", None
        return "Let me reason about it.", None

    log = run(steer_me(format=fmt, element="consumer_surplus", data_dir=str(data_root)),
              mock_model(policy), tmp_path, limit=4)
    m = metrics(log)
    assert m["all/no_answer_rate"] == 0.0
    assert not math.isnan(m["all/exact_match"])
    assert ("all/ece" in m) == (fmt != "free")
    s = log.samples[0]
    users = [x.text for x in s.messages if x.role == "user"]
    n_turns = {"mc": 1, "none": 1, "free": 1, "shown": 2, "hidden": 2}[fmt]
    assert len(users) == n_turns and len(s.messages) == 2 * n_turns
    if fmt == "shown":
        assert "\nA. " in users[0] and users[0].endswith(EXPLANATION_SUFFIXES[2])
        assert users[1] == MC_ANSWER_SUFFIX
    if fmt == "hidden":
        assert "\nA. " not in users[0] and users[0].endswith("\n" + EXPLANATION_SUFFIXES[2])
        assert users[1].startswith("\nA. ") and users[1].endswith("\n" + MC_ANSWER_SUFFIX)
    if fmt == "none":
        assert NONE_OPTION in users[0]
    if fmt == "free":
        assert "\nA. " not in users[0] and users[0].endswith(FREE_ANSWER_SUFFIX)
        assert m["all/exact_match"] == 0.25  # only question 0 has gold 10.123


def test_steer_task_also_runs(data_root, tmp_path):
    log = run(steer(element="tiny", data_dir=str(data_root)), mock_model(always("A", {"A": 1.0})), tmp_path)
    assert log.eval.task.endswith("steer") and metrics(log)["all/exact_match"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------ hand-computed metrics
def tiny_policy(last, all_):
    i = question_index(last, "tiny q")
    return {
        0: ("A", {"A": 0.7, "B": 0.2, "The": 0.1}),          # gold A: right
        1: ("A", {"A": 0.5, " A": 0.05, "B": 0.4}),          # gold B: wrong
        2: ("C", {"C": 0.95, "A": 0.05}),                     # gold C: right, conf in top bin
    }[i]


# conditioning: q0 cond (7/9, 2/9, 0), q1 (5/9, 4/9, 0), q2 (0.05, 0, 0.95)
P_COND = [7 / 9, 4 / 9, 0.95]
CONF_COND = [7 / 9, 5 / 9, 0.95]
TOP_OK = [1, 0, 1]
BRIER_COND = [(2 / 9) ** 2 + (2 / 9) ** 2, (5 / 9) ** 2 + (5 / 9) ** 2, 0.05**2 + 0.05**2]
# mixing (alpha = 0.9, 0.9, 1.0): mix = alpha * cond + (1 - alpha) / 3
MIX = [[0.9 * c + 0.1 / 3 for c in (7 / 9, 2 / 9, 0)],
       [0.9 * c + 0.1 / 3 for c in (5 / 9, 4 / 9, 0)],
       [0.05, 0.0, 0.95]]
GOLD = [0, 1, 2]


def _ece(conf, ok):
    bins = {}
    for c, y in zip(conf, ok):
        b = min(int(c * 10), 9)
        bins.setdefault(b, [0.0, 0.0])
        bins[b][0] += c
        bins[b][1] += y
    return sum(abs(a - c) for c, a in bins.values()) / len(conf)


def test_metrics_hand_computed_conditioning(data_root, tmp_path):
    log = run(steer_me(element="tiny", data_dir=str(data_root)), mock_model(tiny_policy), tmp_path)
    m = metrics(log)
    assert m["tiny/exact_match"] == pytest.approx(2 / 3)
    assert m["tiny/normalized_accuracy"] == pytest.approx((1 - 0.5 + 1) / 3)   # wrong costs 1/(3-1)
    assert m["tiny/epa"] == pytest.approx(sum(P_COND) / 3)
    assert m["tiny/brier"] == pytest.approx(sum(BRIER_COND) / 3)
    # bins 7, 5 and 9: (|1 - 7/9| + |0 - 5/9| + |1 - 0.95|) / 3
    assert m["tiny/ece"] == pytest.approx((2 / 9 + 5 / 9 + 0.05) / 3)
    assert m["tiny/ece"] == pytest.approx(_ece(CONF_COND, TOP_OK))
    assert m["all/epa"] == m["tiny/epa"]


def test_metrics_hand_computed_mixing(data_root, tmp_path):
    log = run(steer_me(element="tiny", prob_mode="mix", data_dir=str(data_root)), mock_model(tiny_policy), tmp_path)
    m = metrics(log)
    p_mix = [MIX[i][GOLD[i]] for i in range(3)]
    brier_mix = [sum(((j == GOLD[i]) - MIX[i][j]) ** 2 for j in range(3)) for i in range(3)]
    conf_mix = [max(x) for x in MIX]
    assert m["tiny/epa"] == pytest.approx(sum(p_mix) / 3)
    assert m["tiny/brier"] == pytest.approx(sum(brier_mix) / 3)
    assert m["tiny/ece"] == pytest.approx(_ece(conf_mix, TOP_OK))
    # accuracy does not depend on the probability mode
    assert m["tiny/exact_match"] == pytest.approx(2 / 3)
    # both variants are stored per sample regardless of prob_mode
    v = log.samples[0].scores["steer_scorer"].value
    assert v["p_correct_cond"] == pytest.approx(7 / 9) and v["p_correct_mix"] == pytest.approx(MIX[0][0])


def test_provider_without_logprobs_reports_accuracy_only(data_root, tmp_path):
    log = run(steer_me(element="tiny", data_dir=str(data_root)),
              mock_model(tiny_policy, provider_logprobs=False), tmp_path)
    m = metrics(log)
    assert m["tiny/exact_match"] == pytest.approx(2 / 3)
    assert m["tiny/normalized_accuracy"] == pytest.approx(0.5)
    for k in ("ece", "brier", "epa"):
        assert math.isnan(m[f"tiny/{k}"]) and math.isnan(m[f"all/{k}"])
    assert all(s.scores["steer_scorer"].value["has_probs"] == 0 for s in log.samples)


def test_logprobs_off_drops_calibration_metrics(data_root, tmp_path):
    log = run(steer_me(element="tiny", logprobs=False, data_dir=str(data_root)), mock_model(always("B")), tmp_path)
    m = metrics(log)
    assert set(m) == {f"{g}/{k}" for g in ("tiny", "all") for k in ("exact_match", "normalized_accuracy", "no_answer_rate")}
    assert m["tiny/exact_match"] == pytest.approx(1 / 3)


def test_unanswered_excluded_from_accuracy(data_root, tmp_path):
    def policy(last, all_):
        i = question_index(last, "tiny q")
        return ("A", {"A": 1.0}) if i == 0 else ("I am not sure.", {"I": 0.9, "B": 0.1})

    m = metrics(run(steer_me(element="tiny", data_dir=str(data_root)), mock_model(policy), tmp_path))
    assert m["tiny/no_answer_rate"] == pytest.approx(2 / 3)
    assert m["tiny/exact_match"] == 1.0          # over answered questions only (steer-scoring n)
    assert m["tiny/epa"] == 1.0


# ------------------------------------------------------------------ consistency and multi-part
def pairs_policy(last, all_):
    g = question_index(last, "pairs g")
    part1 = "part1" in last
    # g0: B then B (0, 1) consistent; g1: A then C (1, 0) inconsistent; g2: A then A (1, 2) consistent
    choice = {(0, False): "A", (0, True): "D", (1, False): "B", (1, True): "C",
              (2, False): "B", (2, True): "E"}[(g, part1)]
    return choice, {choice: 0.9}


def test_consistency_pair(data_root, tmp_path):
    log = run(steer_me(element="pairs", data_dir=str(data_root)), mock_model(pairs_policy), tmp_path)
    m = metrics(log)
    assert m["pairs/exact_match"] == pytest.approx(2 / 3)
    # chance = 2 allowed / (2 * 3) = 1/3, so a violation costs (1/3) / (2/3) = 0.5
    assert m["pairs/normalized_accuracy"] == pytest.approx((1 - 0.5 + 1) / 3)
    assert math.isnan(m["pairs/epa"])          # no correct option: no calibration
    by_id = {s.id: s for s in log.samples}
    s = by_id["pairs/g1"]
    assert s.scores["steer_scorer"].value["correct"] == 0
    assert s.scores["steer_scorer"].answer == "B|C"
    users = [x.text for x in s.messages if x.role == "user"]
    assert len(users) == 2 and len(s.messages) == 4                       # one conversation, in order
    assert "pairs g1 part0" in users[0] and "\nA. B\nB. A\n" in users[0]
    assert "pairs g1 part1" in users[1] and "\nC. C\nD. B\nE. A\n" in users[1]  # letters continue
    assert [r["chosen"] for r in s.metadata["responses"]] == [1, 0]


def test_multi_part_answer_graded(data_root, tmp_path):
    def policy(last, all_):
        g = question_index(last, "multi m")
        if "part0" in last:
            return "B", {"B": 0.8, "A": 0.2}
        return ("D", {"D": 1.0}) if g == 0 else ("C", {"C": 1.0})

    log = run(steer_me(element="multi", data_dir=str(data_root)), mock_model(policy), tmp_path)
    m = metrics(log)
    assert m["multi/exact_match"] == 0.5
    # penalty = max over parts of 1/(n-1) = max(1, 1/2) = 1
    assert m["multi/normalized_accuracy"] == pytest.approx((1 - 1) / 2)
    # probabilities from the first part only (p(B) = 0.8 in both)
    assert m["multi/epa"] == pytest.approx(0.8)


# ------------------------------------------------------------------ free text
def test_free_text_sig_figs(data_root, tmp_path):
    def policy(last, all_):
        i = question_index(last, "cs question")
        gold = 10 + i + 0.123
        answers = {0: f"ANSWER: {gold:.4f}",                  # 10.1230 -> right
                   1: "I get 11.1 so ANSWER: 11.1",            # 11.1 vs 11.123 at 3 s.f. (11.1) -> right
                   2: "ANSWER: 12.2",                          # wrong
                   3: "the answer is 13.12"}                   # no ANSWER:, last number -> right
        return answers.get(i, "no idea"), None

    log = run(steer_me(format="free", element="consumer_surplus", data_dir=str(data_root)),
              mock_model(policy), tmp_path, limit=5)
    m = metrics(log)
    assert m["all/no_answer_rate"] == pytest.approx(1 / 5)
    assert m["all/exact_match"] == pytest.approx(3 / 4)
    assert m["all/normalized_accuracy"] == pytest.approx(3 / 4)
    log2 = run(steer_me(format="free", element="consumer_surplus", sig_figs=5, data_dir=str(data_root)),
               mock_model(policy), tmp_path, limit=5)
    assert metrics(log2)["all/exact_match"] == pytest.approx(1 / 4)


# ------------------------------------------------------------------ logprobs request and log content
def test_logprobs_requested_only_on_answer_turn(data_root, tmp_path):
    calls = []

    def fn(messages, tools, tool_choice, config):
        from conftest import output
        calls.append((messages[-1].text, bool(config.logprobs), config.top_logprobs))
        last = messages[-1].text
        return output("B", {"B": 0.9}, True) if last.endswith(MC_ANSWER_SUFFIX) else output("reasoning", None, False)

    from inspect_ai.model import get_model
    run(steer_me(format="hidden", element="tiny", data_dir=str(data_root)),
        get_model("mockllm/model", custom_outputs=fn), tmp_path)
    assert sorted((lp, k) for _, lp, k in calls) == [(False, None)] * 3 + [(True, 20)] * 3
    for text, lp, _ in calls:
        assert lp == text.endswith(MC_ANSWER_SUFFIX)


def test_log_records_task_args_and_responses(data_root, tmp_path):
    log = run(steer_me(element="tiny", format="none", data_dir=str(data_root)), mock_model(tiny_policy), tmp_path)
    assert log.eval.task_args["format"] == "none"
    assert log.eval.task_version == "0.1.0"
    assert log.eval.metadata["benchmark"] == "steer_me"
    r = log.samples[0].metadata["responses"][0]
    assert r["letter"] in "ABC" and isinstance(r["top_logprobs"], list)
    assert re.match(r"tiny/q\d", log.samples[0].id)
