"""The STEER solver: asks every part of a question group in one conversation, in order."""

from __future__ import annotations

from typing import Any

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .prompts import turns_for_part
from .scoring import answer_position_top_logprobs, letters_for, parse_letter


def _logprobs_content(state: TaskState) -> list[Any] | None:
    out = state.output
    if not out or not out.choices:
        return None
    lp = out.choices[0].logprobs
    return lp.content if lp is not None and lp.content else None


@solver
def steer_solver(
    format: str = "mc",
    explanation_length: int = 3,
    logprobs: bool = True,
    top_logprobs: int = 20,
    answer_max_tokens: int | None = None,
) -> Solver:
    """Ask each part of the sample's group in sequence; record the answers in metadata.

    Each part gets the user turns from ``prompts.turns_for_part``; the model replies to every
    turn. Option letters continue across parts (A-B for part 0, C-E for part 1, ...), as in the
    original harness. Logprobs are requested only on the answer turn of the multiple-choice
    formats. Per-part results are written to ``state.metadata["responses"]`` (saved in the log):
    the chosen letter and index, and the ``top_logprobs`` at the answer position.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        state.messages = []
        responses: list[dict[str, Any]] = []
        offset = 0
        for i, part in enumerate(md["parts"]):
            turns = turns_for_part(format, part, offset, explanation_length)
            if i == 0 and md.get("prefix"):
                turns[0] = md["prefix"] + turns[0]
            for t, text in enumerate(turns):
                state.messages.append(ChatMessageUser(content=text))
                is_answer_turn = t == len(turns) - 1
                if is_answer_turn and format != "free":
                    kwargs: dict[str, Any] = {}
                    if logprobs:
                        kwargs.update(logprobs=True, top_logprobs=top_logprobs)
                    if answer_max_tokens is not None:
                        kwargs["max_tokens"] = answer_max_tokens
                    state = await generate(state, **kwargs)
                else:
                    state = await generate(state)
            completion = state.output.completion if state.output else ""
            n = len(part["options"])
            letters = letters_for(offset, n)
            if format == "free":
                responses.append({"completion": completion, "letters": None, "letter": None,
                                  "chosen": None, "top_logprobs": None})
            else:
                letter = parse_letter(completion, letters)
                top = answer_position_top_logprobs(_logprobs_content(state), letters) if logprobs else None
                responses.append({
                    "completion": completion[:2000],
                    "letters": letters,
                    "letter": letter,
                    "chosen": None if letter is None else letters.index(letter),
                    "top_logprobs": top,
                })
            offset += n
        state.metadata["responses"] = responses
        return state

    return solve
