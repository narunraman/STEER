"""Brier averaged over the options (the STEER-ME paper's definition), for new and old logs."""
import pytest

from steer_bench.export import sample_row
from steer_bench.scoring import distribution_stats


def test_distribution_stats_brier_avg_is_sum_over_k():
    d = distribution_stats({"A": 0.5, "B": 0.3, "C": 0.2}, ["A", "B", "C"], 1)
    assert d["brier_avg_cond"] == pytest.approx(d["brier_cond"] / 3)
    assert d["brier_avg_mix"] == pytest.approx(d["brier_mix"] / 3)


def test_old_logs_get_brier_avg_from_first_part_options():
    value = {"answered": 1, "has_probs": 1, "correct": 1, "norm": 1, "norm_strict": 1, "conf_cond": 0.5,
             "conf_mix": 0.5, "top_correct": 1, "p_correct_cond": 0.5, "p_correct_mix": 0.5,
             "brier_cond": 0.6, "brier_mix": 0.9, "alpha": 1.0}
    md = {"parts": [{"options": ["a", "b", "c"]}, {"options": ["d", "e"]}]}
    r = sample_row(value, md, None)
    assert r["sum_brier_avg_cond"] == pytest.approx(0.2)
    assert r["sum_brier_avg_mix"] == pytest.approx(0.3)
    assert r["n_prob"] == 1
