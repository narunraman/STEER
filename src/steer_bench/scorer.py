"""The STEER scorer: one dict-valued Score per question group."""

from __future__ import annotations

import math
from typing import Any

from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from .scoring import (
    NAN,
    consistency_chance,
    distribution_stats,
    extract_number,
    letter_probs,
    parse_gold_number,
    parse_rule,
    sigfig_equal,
)

# Every Score.value has exactly these keys (NaN where not applicable), so epoch reduction and
# the metrics see the same shape on every sample.
VALUE_KEYS = (
    "answered",        # 1 if every part has a usable answer, else 0
    "correct",         # exact match (answer-graded: all parts right; consistency: rule holds); NaN if unanswered
    "norm",            # normalized accuracy contribution; NaN if unanswered
    "norm_strict",     # same, with an unanswered question counted as wrong
    "penalty",         # what a wrong answer costs in normalized accuracy
    "has_probs",       # 1 if option probabilities are available for the first part
    "alpha", "invalid_mass",
    "p_correct_cond", "p_correct_mix",
    "conf_cond", "conf_mix",
    "top_correct",
    "brier_cond", "brier_mix",
)


def _empty() -> dict[str, float]:
    return {k: NAN for k in VALUE_KEYS}


def score_group(md: dict[str, Any], fmt: str, sig_figs: int = 3) -> tuple[dict[str, float], str]:
    """Score one packed sample from its metadata (``parts`` + solver ``responses``).

    Returns (value dict, answer string). Pure function, also used by tests.
    """
    parts, resp = md["parts"], md["responses"]
    v = _empty()
    v["has_probs"] = 0.0

    if fmt == "free":
        golds = [parse_gold_number(p["options"][p["correct_index"]]) for p in parts]
        values = [extract_number(r["completion"]) for r in resp]
        answered = all(x is not None for x in values)
        correct = answered and all(sigfig_equal(x, g, sig_figs) for x, g in zip(values, golds))
        v.update(answered=float(answered), penalty=0.0)
        # free text: chance of guessing is ~0, so a wrong answer costs nothing (norm = exact match)
        v["correct"] = float(correct) if answered else NAN
        v["norm"] = float(correct) if answered else NAN
        v["norm_strict"] = float(correct)
        return v, "|".join("" if x is None else repr(x) for x in values)

    chosen = [r["chosen"] for r in resp]
    answered = all(c is not None for c in chosen)
    answer_str = "|".join(r["letter"] or "-" for r in resp)
    v["answered"] = float(answered)

    if md["grading"] == "consistency":
        _, allowed = parse_rule(md["consistency_rule"])
        n_opts = [len(p["options"]) for p in parts]
        chance = consistency_chance(allowed, n_opts)
        # Generalizes 1/(n-1): a wrong answer costs p/(1-p), so uniform random choice scores 0.
        penalty = chance / (1.0 - chance) if chance < 1 else 0.0
        ok = answered and tuple(chosen) in set(allowed)
        v["penalty"] = penalty
        v["correct"] = float(ok) if answered else NAN
        v["norm"] = (1.0 if ok else -penalty) if answered else NAN
        v["norm_strict"] = 1.0 if ok else -penalty
        return v, answer_str

    golds = [p["correct_index"] for p in parts]
    penalty = max(1.0 / (len(p["options"]) - 1) if len(p["options"]) > 1 else 1.0 for p in parts)
    ok = answered and list(chosen) == golds
    v["penalty"] = penalty
    v["correct"] = float(ok) if answered else NAN
    v["norm"] = (1.0 if ok else -penalty) if answered else NAN
    v["norm_strict"] = 1.0 if ok else -penalty

    top = resp[0].get("top_logprobs")
    if answered and top is not None:
        letters = resp[0]["letters"]
        raw = letter_probs([tuple(t) for t in top], letters)
        v.update(distribution_stats(raw, letters, golds[0]))
        v["has_probs"] = 1.0
    return v, answer_str


@scorer(metrics=[])  # metrics are attached on the Task
def steer_scorer(format: str = "mc", sig_figs: int = 3) -> Any:
    """Exact match, normalized accuracy and per-question calibration quantities.

    Both the conditioning and the mixing variants are stored on every sample; the task's
    ``prob_mode`` only picks which one the headline metrics report.
    """

    async def score(state: TaskState, target: Target) -> Score:
        value, answer = score_group(state.metadata, format, sig_figs)
        expl = "unanswered" if value["answered"] == 0 else (
            "correct" if value["correct"] == 1 else "incorrect")
        if not math.isnan(value["p_correct_cond"]):
            expl += f"; p(correct) cond={value['p_correct_cond']:.3f} mix={value['p_correct_mix']:.3f}"
        return Score(value=value, answer=answer, explanation=expl)

    return score
