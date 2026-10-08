"""Prompt text: every string the model sees, other than the questions themselves, is defined here.

The default prompt set, ``open2025``, is the one used by every canonical open-weight STEER-ME
run (the 2025 evaluation harness, ``vllm_model.py`` of 2025-05-19). The
strings were recovered verbatim from the raw result pickles and checked byte for byte against
7,942 original prompts. Those runs used the ``shown`` and
``hidden`` formats; the prompts of ``mc``/``none`` (same answer instruction), ``free`` (the
reasoning instruction alone; the answer is read from ``\\boxed{}``) and few-shot examples
follow the same strings.

To use your own wording, pass ``-T prompts=path/to/prompts.json``: a JSON object with the keys
of a ``PROMPT_SETS`` entry (missing keys fall back to ``open2025``). The set's name and a hash
of its strings are recorded in the log and in the converted cells.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from .scoring import part_offsets

DEFAULT_PROMPTS = "open2025"

PROMPT_SETS: dict[str, dict[str, Any]] = {
    "open2025": {
        # Strings are used exactly as written. Each turn is built as
        #   question [+ options] + "\n" + reasoning   or   [options +] "\n" + answer
        # reasoning instruction of the formats shown / hidden / free
        "reasoning": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        # the answer turn; the model's reply should start with the option letter
        "answer": ("\nAnswer by writing the option letter corresponding to the correct option. "
                   "RESPOND WITH ONLY A SINGLE LETTER.\nA:"),
        # extra text after the reasoning instruction in the free-text format
        "free_answer": "",
        # hidden: whether the options turn starts with a newline before "A. ..."
        "hidden_options_newline": False,
        # few-shot examples: text between the answer instruction and the example's letter
        "few_shot_answer_sep": " ",
    },
}
PROMPT_KEYS = tuple(PROMPT_SETS[DEFAULT_PROMPTS])


def load_prompts(prompts: str | None = None) -> tuple[str, dict[str, Any]]:
    """(name, strings) for a built-in set name or a path to a JSON file of strings."""
    prompts = prompts or DEFAULT_PROMPTS
    if prompts in PROMPT_SETS:
        return prompts, dict(PROMPT_SETS[prompts])
    path = Path(prompts).expanduser()
    if not path.is_file():
        raise ValueError(f"prompts must be one of {sorted(PROMPT_SETS)} or a JSON file, got {prompts!r}")
    custom = json.loads(path.read_text())
    if not isinstance(custom, dict) or set(custom) - set(PROMPT_KEYS):
        raise ValueError(f"{path}: expected a JSON object with keys from {PROMPT_KEYS}")
    return path.stem, {**PROMPT_SETS[DEFAULT_PROMPTS], **custom}


def prompts_hash(strings: dict[str, Any]) -> str:
    """Short hash of a prompt set's strings (recorded with the results)."""
    blob = json.dumps({k: strings[k] for k in PROMPT_KEYS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def option_block(options: list[str], offset: int) -> str:
    """``\\nA. opt`` lines; letters continue across the parts of a group (see ``scoring.part_offsets``)."""
    return "".join(f"\n{ascii_uppercase[offset + i]}. {o}" for i, o in enumerate(options))


def few_shot_prefix(example_groups: list[list[dict[str, Any]]], strings: dict[str, Any]) -> str:
    """Solved examples in the plain multiple-choice format, each followed by its answer letter.

    The examples always show the options without reasoning, are never altered by the "No other
    option is correct" transform, and the prefix ends with a blank line.
    """
    if not example_groups:
        return ""
    prefix = ""
    for parts in example_groups:
        msgs = []
        for offset, p in zip(part_offsets([len(p["options"]) for p in parts]), parts):
            letter = ascii_uppercase[offset + p["correct_index"]]
            msgs.append("Q: " + p["question_text"] + option_block(p["options"], offset) + "\n"
                        + strings["answer"] + strings["few_shot_answer_sep"] + letter + "\n")
        prefix += "\n".join(msgs)
    return prefix + "\n\n"


def turns_for_part(fmt: str, part: dict[str, Any], offset: int, strings: dict[str, Any]) -> list[str]:
    """User messages for one part. The model replies after each; the last reply is the answer.

    mc / none : [question + options + answer]
    shown     : [question + options + reasoning, answer]
    hidden    : [question + reasoning, options + answer]
    free      : [question + reasoning + free_answer]
    """
    q = "Q: " + part["question_text"]
    opts = option_block(part["options"], offset)
    reason, answer = strings["reasoning"], strings["answer"]
    if fmt in ("mc", "none"):
        return [q + opts + "\n" + answer]
    if fmt == "shown":
        return [q + opts + "\n" + reason, answer]
    if fmt == "hidden":
        hidden_opts = opts if strings["hidden_options_newline"] else opts.lstrip("\n")
        return [q + "\n" + reason, hidden_opts + "\n" + answer]
    if fmt == "free":
        return [q + "\n" + reason + strings["free_answer"]]
    raise ValueError(f"unknown format {fmt!r}")
