"""Per-question values agree with steer-scoring's score_row (skipped if it is not installed).

Run with: uv run --with ~/UBC/steer-scoring pytest tests/test_parity_steer_scoring.py
"""

import math
import random
from string import ascii_uppercase

import pytest

from steer_bench.scorer import score_group

score = pytest.importorskip("steer_scoring.score")

KEYS = ["correct", "p_correct_cond", "p_correct_mix", "conf_cond", "conf_mix", "top_correct",
        "brier_cond", "brier_mix", "alpha", "invalid_mass"]


@pytest.mark.parametrize("seed", range(25))
def test_score_row_parity(seed):
    rng = random.Random(seed)
    k = rng.choice([2, 3, 4, 5])
    letters = list(ascii_uppercase[:k])
    gold = rng.randrange(k)
    # random top-k over some letters plus junk tokens; letters without spaces (steer-scoring keeps
    # spaces when cleaning tokens, this package strips them)
    toks = rng.sample(letters, rng.randint(1, k)) + ["The", "I", "\n"][: rng.randint(0, 3)]
    w = [rng.random() for _ in toks]
    total = sum(w) / rng.uniform(0.5, 1.0)
    probs = {t: x / total for t, x in zip(toks, w)}
    argmax = max(probs, key=probs.get)
    if argmax not in letters:  # steer-scoring scores the raw argmax; keep the letter on top
        probs[argmax], probs[toks[0]] = probs[toks[0]], probs[argmax]
        argmax = toks[0]

    row = {"valid_tokens": [letters], "answer": [letters[gold]], "temperature": 0.0,
           "logprobs": [{t: math.log(p) for t, p in probs.items()}]}
    ref = score.score_row(row)

    md = {"grading": "answer",
          "parts": [{"options": [str(i) for i in range(k)], "correct_index": gold}],
          "responses": [{"letters": letters, "letter": argmax, "chosen": letters.index(argmax),
                         "top_logprobs": [[t, math.log(p)] for t, p in probs.items()]}]}
    got, _ = score_group(md, "mc")
    assert got["has_probs"] == 1
    assert got["norm"] == pytest.approx(ref["norm"])
    for key in KEYS:
        assert float(got[key]) == pytest.approx(float(ref[key]), abs=1e-12), key
    assert score.ece_bin(ref["conf_cond"]) == min(int(got["conf_cond"] * 10), 9)
