import math

import pytest

from steer_bench.scoring import (
    consistency_chance,
    distribution_stats,
    ece_bin,
    extract_number,
    letter_probs,
    parse_letter,
    parse_rule,
    sigfig_equal,
)

L4 = ["A", "B", "C", "D"]


@pytest.mark.parametrize("text,expected", [
    ("B", "B"), (" B", "B"), ("B.", "B"), ("(C)", "C"), ("D: because", "D"), ("**A**", "A"),
    ("The answer is C.", "C"), ("Reasoning...\nANSWER: B", "B"), ("answer: (D)", "D"),
    ("E", None),                # not a valid letter for this part
    ("I think so", None), ("", None), ("Apples are red", None),
])
def test_parse_letter(text, expected):
    assert parse_letter(text, L4) == expected


def test_parse_letter_continued_letters():
    # second part of a group uses letters C-E
    assert parse_letter("D", ["C", "D", "E"]) == "D"
    assert parse_letter("A", ["C", "D", "E"]) is None


def test_letter_probs_max_over_duplicates_and_cleaning():
    top = [("A", math.log(0.5)), (" A", math.log(0.05)), ("B.", math.log(0.3)), ("The", math.log(0.1)),
           ("E", math.log(0.05))]
    raw = letter_probs(top, L4)
    assert raw == pytest.approx({"A": 0.5, "B": 0.3})


def test_conditioning_and_mixing():
    raw = {"A": 0.6, "B": 0.2}            # alpha = 0.8, C and D absent
    s = distribution_stats(raw, L4, gold=1)
    cond = [0.75, 0.25, 0, 0]
    mix = [0.8 * c + 0.2 / 4 for c in cond]  # [0.65, 0.25, 0.05, 0.05]
    assert s["alpha"] == pytest.approx(0.8)
    assert s["p_correct_cond"] == pytest.approx(0.25)
    assert s["p_correct_mix"] == pytest.approx(0.25)
    assert s["conf_cond"] == pytest.approx(0.75)
    assert s["conf_mix"] == pytest.approx(0.65)
    assert s["top_correct"] == 0.0
    assert s["brier_cond"] == pytest.approx(0.75**2 + 0.75**2)
    assert s["brier_mix"] == pytest.approx(sum((o - p) ** 2 for o, p in zip([0, 1, 0, 0], mix)))
    assert sum(mix) == pytest.approx(1.0)


def test_alpha_zero_is_uniform():
    s = distribution_stats({}, L4, gold=0)
    assert s["p_correct_cond"] == pytest.approx(0.25)
    assert s["p_correct_mix"] == pytest.approx(0.25)


def test_ece_bins():
    assert ece_bin(0.0) == 0
    assert ece_bin(0.0999) == 0
    assert ece_bin(0.1) == 1
    assert ece_bin(0.95) == 9
    assert ece_bin(1.0) == 9   # top bin, not dropped


def test_consistency_rule():
    rid, allowed = parse_rule("same_brand:01,12")
    assert rid == "same_brand" and allowed == [(0, 1), (1, 2)]
    assert consistency_chance(allowed, [2, 3]) == pytest.approx(2 / 6)
    assert parse_rule("flip_endowment:1")[1] == [(1,)]


@pytest.mark.parametrize("text,value", [
    ("The surplus is 2.83", 2.83),
    ("Step 1: 3 - 1 = 2. ANSWER: 1,234.5", 1234.5),
    ("ANSWER: $-0.75", -0.75),
    ("we get 1.5e3 units", 1500.0),
    ("ANSWER: −4", -4.0),
    ("no numbers here", None),
])
def test_extract_number(text, value):
    assert extract_number(text) == value


@pytest.mark.parametrize("value,gold,k,ok", [
    (2.834, 2.83, 3, True),
    (2.836, 2.83, 3, False),     # 2.84 vs 2.83
    (1072.0, 1071.83, 3, True),  # both 1.07e3
    (1072.0, 1071.83, 6, False),
    (7.4, 7.0, 3, False),
    (0.000123456, 0.000123, 3, True),
    (None, 1.0, 3, False),
])
def test_sigfig(value, gold, k, ok):
    assert sigfig_equal(value, gold, k) is ok
