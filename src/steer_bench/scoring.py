"""Pure scoring functions (no Inspect objects).

Definitions (the same as in the scoring behind the results website):

* option probabilities: tokens at the answer position, cleaned of non-alphanumerics, kept if
  they are a valid option letter, max over duplicates (``"C"`` vs ``" C"``), ``exp(logprob)``;
  ``alpha`` = total probability on valid options;
* conditioning: ``p / alpha`` (uniform if ``alpha == 0``);
* mixing: ``alpha * p_cond + (1 - alpha) / |O|`` over all options;
* top option, confidence = max probability, ``top_correct`` = top option is the gold option;
* Brier = sum over options of ``(1[o = gold] - p(o))^2``;
* probability metrics use the first part of multi-part questions only;
* normalized accuracy: +1 if correct, else ``-max over parts of 1/(n_options - 1)``.
"""

from __future__ import annotations

import math
import re
from string import ascii_uppercase
from typing import Any

NAN = float("nan")
N_BINS = 10

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def clean_token(tok: str) -> str:
    # spaces are dropped too, so " C" counts as "C"
    return _NON_ALNUM.sub("", tok)


# ---------------------------------------------------------------- chosen letter from the text
_LEAD = re.compile(r"^\s*(?:\*\*)?\(?\s*([A-Z])\s*(?:\)|\.|:|,|\*\*|\s|$)")
_ANSWER_IS = re.compile(r"(?:answer|ANSWER|Answer)\s*(?:is|:)?\s*(?:option\s*)?\(?\*{0,2}([A-Z])\b")


def parse_letter(text: str, letters: list[str]) -> str | None:
    """The option letter the model wrote, or None.

    Accepts a leading letter (``"B"``, ``" B."``, ``"(B)"``, ``"B: ..."``), else the last
    ``answer is B`` / ``ANSWER: B``. Only letters valid for this part count.
    """
    text = text or ""
    m = _LEAD.match(text)
    if m and m.group(1) in letters:
        return m.group(1)
    found = [x for x in _ANSWER_IS.findall(text) if x in letters]
    return found[-1] if found else None


# ---------------------------------------------------------------- option probabilities
def answer_position_top_logprobs(logprobs_content: list[Any] | None, letters: list[str]) -> list[tuple[str, float]] | None:
    """(token, logprob) pairs of ``top_logprobs`` at the answer position.

    The answer position is the first generated token that is a valid option letter after
    cleaning (with the paper prompt this is the first token); if there is none, the first token.
    Returns None if the provider returned no logprobs.
    """
    if not logprobs_content:
        return None
    pos = next((lp for lp in logprobs_content if clean_token(lp.token) in letters), logprobs_content[0])
    tops = pos.top_logprobs or []
    pairs = [(t.token, float(t.logprob)) for t in tops]
    if not any(clean_token(tok) == clean_token(pos.token) for tok, _ in pairs):
        pairs.append((pos.token, float(pos.logprob)))
    return pairs


def letter_probs(top: list[tuple[str, float]], letters: list[str]) -> dict[str, float]:
    """Raw probability per valid letter (max over duplicate tokens)."""
    best: dict[str, float] = {}
    for tok, lp in top:
        c = clean_token(tok)
        if c in letters:
            best[c] = max(best.get(c, -math.inf), lp)
    raw = {k: math.exp(v) for k, v in best.items()}
    s = sum(raw.values())
    if s > 1.0:  # numerical guard
        raw = {k: v / s for k, v in raw.items()}
    return raw


def distribution_stats(raw: dict[str, float], letters: list[str], gold: int) -> dict[str, float]:
    """Calibration quantities for one question part (both conditioning and mixing)."""
    k = len(letters)
    alpha = sum(raw.values())
    if alpha > 0:
        cond = [raw.get(L, 0.0) / alpha for L in letters]
    else:
        cond = [1.0 / k] * k
    mix = [alpha * c + (1.0 - alpha) / k for c in cond]
    top = max(range(k), key=lambda i: cond[i])  # same argmax for both
    onehot = [1.0 if i == gold else 0.0 for i in range(k)]
    return {
        "alpha": alpha,
        "invalid_mass": 1.0 - alpha,
        "p_correct_cond": cond[gold],
        "p_correct_mix": mix[gold],
        "conf_cond": cond[top],
        "conf_mix": max(mix),
        "top_correct": float(top == gold),
        "brier_cond": sum((o - p) ** 2 for o, p in zip(onehot, cond)),
        "brier_mix": sum((o - p) ** 2 for o, p in zip(onehot, mix)),
    }


def ece_bin(conf: float) -> int:
    """Bin i covers [i/10, (i+1)/10); conf == 1.0 goes in the top bin."""
    return min(int(conf * N_BINS), N_BINS - 1)


# ---------------------------------------------------------------- consistency rules
def parse_rule(rule: str) -> tuple[str, list[tuple[int, ...]]]:
    """``same_brand:01,12`` -> ("same_brand", [(0, 1), (1, 2)])."""
    rule_id, _, allowed = rule.partition(":")
    return rule_id, [tuple(int(c) for c in a.strip()) for a in allowed.split(",") if a.strip()]


def consistency_chance(allowed: list[tuple[int, ...]], n_options: list[int]) -> float:
    """Probability that uniform random choices satisfy the rule."""
    total = math.prod(n_options)
    valid = {a for a in allowed if len(a) == len(n_options) and all(0 <= x < n for x, n in zip(a, n_options))}
    return len(valid) / total if total else NAN


# ---------------------------------------------------------------- free-text numbers
_NUM = re.compile(r"[-+−]?\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        s = m.group(0)
        if not re.search(r"\d", s):
            continue
        s = s.replace("−", "-").replace("$", "").replace(",", "").replace(" ", "")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_number(text: str) -> float | None:
    """The final numeric answer: the first number in the last ``\\boxed{...}``, else the first
    number after the last ``ANSWER:``, else the last number in the text. Thousands separators,
    ``$`` and a unicode minus are accepted."""
    text = text or ""
    boxed = _BOXED.findall(text)
    if boxed:
        nums = _numbers(boxed[-1])
        if nums:
            return nums[0]
    idx = max(text.rfind("ANSWER:"), text.rfind("Answer:"))
    if idx >= 0:
        nums = _numbers(text[idx + 7:])
        if nums:
            return nums[0]
    nums = _numbers(text)
    return nums[-1] if nums else None


def round_sig(x: float, k: int) -> float:
    return float(f"{x:.{k}g}")


def sigfig_equal(value: float | None, gold: float, k: int) -> bool:
    """True if ``value`` and ``gold`` agree when both are rounded to ``k`` significant figures."""
    if value is None or math.isnan(value):
        return False
    return round_sig(value, k) == round_sig(gold, k)


def parse_gold_number(text: str) -> float | None:
    s = (text or "").strip().replace(",", "").replace("$", "").replace("−", "-")
    try:
        return float(s)
    except ValueError:
        return None


def letters_for(offset: int, n: int) -> list[str]:
    return list(ascii_uppercase[offset: offset + n])
