"""List-level metrics over the dict-valued scores (definitions as steer-scoring ``metrics.py``).

All means skip NaN, so:
* accuracy metrics are over answered questions (``n``; unanswered ones are reported by
  ``no_answer_rate``), as steer-scoring drops "No Answer" rows;
* calibration metrics are over answered questions that have option probabilities. A run
  whose provider returns no logprobs gets NaN, never a made-up value.
"""

from __future__ import annotations

import math

from inspect_ai.scorer import Metric, SampleScore, grouped, metric

from .scoring import N_BINS, ece_bin

NAN = float("nan")


def _value(s: SampleScore, key: str) -> float:
    v = s.score.value
    x = v.get(key) if isinstance(v, dict) else None
    if x is None:
        return NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _mean(scores: list[SampleScore], key: str) -> float:
    xs = [x for x in (_value(s, key) for s in scores) if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else NAN


@metric(name="exact_match")
def exact_match() -> Metric:
    """Fraction of answered questions answered correctly (consistency groups: rule satisfied)."""
    def m(scores: list[SampleScore]) -> float:
        return _mean(scores, "correct")
    return m


@metric(name="normalized_accuracy")
def normalized_accuracy() -> Metric:
    """Mean of +1 (correct) / -1/(n-1) (wrong) over answered questions."""
    def m(scores: list[SampleScore]) -> float:
        return _mean(scores, "norm")
    return m


@metric(name="no_answer_rate")
def no_answer_rate() -> Metric:
    """Fraction of questions with no usable answer (excluded from the accuracy metrics)."""
    def m(scores: list[SampleScore]) -> float:
        a = _mean(scores, "answered")
        return NAN if math.isnan(a) else 1.0 - a
    return m


@metric(name="epa")
def epa(prob_mode: str = "condition") -> Metric:
    """Expected probability assignment: mean probability on the correct option."""
    key = "p_correct_" + _suffix(prob_mode)
    def m(scores: list[SampleScore]) -> float:
        return _mean(scores, key)
    return m


@metric(name="brier")
def brier(prob_mode: str = "condition") -> Metric:
    """Mean over questions of sum over options of (1[o = gold] - p(o))^2."""
    key = "brier_" + _suffix(prob_mode)
    def m(scores: list[SampleScore]) -> float:
        return _mean(scores, key)
    return m


@metric(name="ece")
def ece(prob_mode: str = "condition") -> Metric:
    """Expected calibration error, 10 equal-width bins [i/10, (i+1)/10) (1.0 in the top bin):
    sum over bins of |sum top_correct - sum confidence| / n."""
    key = "conf_" + _suffix(prob_mode)
    def m(scores: list[SampleScore]) -> float:
        sum_conf = [0.0] * N_BINS
        sum_acc = [0.0] * N_BINS
        n = 0
        for s in scores:
            c, y = _value(s, key), _value(s, "top_correct")
            if math.isnan(c) or math.isnan(y):
                continue
            b = ece_bin(c)
            sum_conf[b] += c
            sum_acc[b] += y
            n += 1
        if n == 0:
            return NAN
        return sum(abs(a - c) for a, c in zip(sum_acc, sum_conf)) / n
    return m


def _suffix(prob_mode: str) -> str:
    if prob_mode in ("condition", "cond", "conditioning"):
        return "cond"
    if prob_mode in ("mix", "mixing"):
        return "mix"
    raise ValueError(f"prob_mode must be 'condition' or 'mix', got {prob_mode!r}")


def by_element(m: Metric, name: str) -> Metric:
    return grouped(m, "element", all="samples", all_label=f"all/{name}",
                   name_template="{group_name}/" + name)


def task_metrics(prob_mode: str = "condition", calibration: bool = True) -> list[Metric]:
    """Metrics attached to the tasks, each per element plus an ``all/<metric>`` aggregate."""
    out = [
        by_element(exact_match(), "exact_match"),
        by_element(normalized_accuracy(), "normalized_accuracy"),
        by_element(no_answer_rate(), "no_answer_rate"),
    ]
    if calibration:
        out += [
            by_element(ece(prob_mode), "ece"),
            by_element(brier(prob_mode), "brier"),
            by_element(epa(prob_mode), "epa"),
        ]
    return out
