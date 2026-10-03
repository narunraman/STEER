"""Prompt text.

Everything except ``FREE_ANSWER_SUFFIX`` is copied word for word from the original harness,
STEER-evaluation ``src/steer_evaluation/steer_benchmark.py`` (commit c4ba97d),
``TestBuilder.__init__`` / ``build_question`` / ``build_prefix``.
"""

from __future__ import annotations

from string import ascii_uppercase
from typing import Any

# steer_benchmark.py TestBuilder.suffix_options
EXPLANATION_SUFFIXES = (
    "\nLet's think step by step. Explain your reasoning in at most 3 sentences.",  # explanation_length=1
    "\nLet's think step by step. Explain your reasoning in at most 5 sentences.",  # explanation_length=2
    "\nLet's think step by step. Explain your reasoning.",                          # explanation_length=3
)
MC_ANSWER_SUFFIX = (
    "\nAnswer by writing the option letter corresponding to the correct option. "
    "WRITE ONLY A SINGLE LETTER. \nA: "
)

# Not from the papers: the original harness has no free-text prompt. Added so the final answer
# can be extracted reliably. OPEN DECISION for the user.
FREE_ANSWER_SUFFIX = "\nEnd your response with 'ANSWER: ' followed by your final answer as a number."


def option_block(options: list[str], offset: int) -> str:
    """``\\nA. opt`` lines; letters continue across the parts of a group (global_option_index)."""
    return "".join(f"\n{ascii_uppercase[offset + i]}. {o}" for i, o in enumerate(options))


def explanation_suffix(explanation_length: int) -> str:
    if explanation_length not in (1, 2, 3):
        raise ValueError("explanation_length must be 1, 2 or 3")
    return EXPLANATION_SUFFIXES[explanation_length - 1]


def few_shot_prefix(example_groups: list[list[dict[str, Any]]]) -> str:
    """build_prefix: examples in the plain multiple-choice format, each followed by its answer.

    As in the original, the examples always use shown options without reasoning, are never
    altered by the "No other option is correct" transform, and the prefix ends with a blank line.
    """
    if not example_groups:
        return ""
    prefix = ""
    for parts in example_groups:
        offset, msgs = 0, []
        for p in parts:
            letter = ascii_uppercase[offset + p["correct_index"]]
            msgs.append("Q: " + p["question_text"] + option_block(p["options"], offset) + "\n"
                        + MC_ANSWER_SUFFIX + letter + "\n")
            offset += len(p["options"])
        prefix += "\n".join(msgs)
    return prefix + "\n\n"


def turns_for_part(fmt: str, part: dict[str, Any], offset: int, explanation_length: int) -> list[str]:
    """User messages for one part. The model replies after each; the last reply is the answer.

    mc / none : [question + options + answer suffix]
    shown     : [question + options + reasoning suffix, answer suffix]
    hidden    : [question + reasoning suffix, options + answer suffix]
    free      : [question + reasoning suffix + free-text answer instruction]
    """
    q = "Q: " + part["question_text"]
    opts = option_block(part["options"], offset)
    if fmt in ("mc", "none"):
        return [q + opts + "\n" + MC_ANSWER_SUFFIX]
    if fmt == "shown":
        return [q + opts + "\n" + explanation_suffix(explanation_length), MC_ANSWER_SUFFIX]
    if fmt == "hidden":
        return [q + "\n" + explanation_suffix(explanation_length), opts + "\n" + MC_ANSWER_SUFFIX]
    if fmt == "free":
        return [q + "\n" + explanation_suffix(explanation_length) + FREE_ANSWER_SUFFIX]
    raise ValueError(f"unknown format {fmt!r}")
