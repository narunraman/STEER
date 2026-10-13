import math

from steer_bench.export import rescore


def _md(fmt_letters=("A", "B", "C", "D"), top=None, completion=""):
    part = {"options": ["1", "2", "3", "4"], "correct_index": 1, "question_text": "q"}
    return {"parts": [part], "grading": "answer", "consistency_rule": None,
            "responses": [{"completion": completion, "letters": list(fmt_letters), "letter": None,
                           "chosen": None, "top_logprobs": top}]}


def test_rescore_reads_the_answer_after_think_from_the_full_reply():
    md = _md(completion="<think>long reasoning, clipped")       # the logged completion lost the end
    value, n = rescore(md, ["thinking out loud", "<think>...</think>\n\nB"], "shown", from_logprobs=False)
    assert value["answered"] == 1 and value["correct"] == 1 and n == 0


def test_rescore_letter_from_logprobs_only_when_asked():
    top = [["To", math.log(0.95)], ["B", math.log(0.04)], ["A", math.log(0.01)]]
    md = _md(top=top, completion="To")
    off, _ = rescore(md, ["To"], "mc", from_logprobs=False)
    assert off["answered"] == 0
    on, n = rescore(md, ["To"], "mc", from_logprobs=True)
    assert on["answered"] == 1 and on["correct"] == 1 and n == 1 and on["has_probs"] == 1
