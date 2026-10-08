"""Inspect tasks ``steer`` and ``steer_me``."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

try:
    import steer_bench  # noqa: F401
except ImportError:
    # Loaded by file path (`inspect eval .../src/steer_bench/tasks.py@steer`) without the package
    # installed: make `steer_bench` importable from the enclosing src/ directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steer_bench import __version__
from steer_bench.data import (
    apply_none_option,
    group_rows,
    load_rows,
    none_option_selection,
    select_few_shot,
)
from steer_bench.metrics import task_metrics
from steer_bench.prompts import DEFAULT_PROMPTS, few_shot_prefix, load_prompts, prompts_hash, turns_for_part
from steer_bench.scorer import steer_scorer
from steer_bench.scoring import parse_gold_number, part_offsets
from steer_bench.solver import steer_solver

FORMATS = ("mc", "shown", "hidden", "none", "free")
TASK_VERSION = "0.1.0"

# metadata copied from the first part of a group onto the Sample (used by grouped() and the
# log converter)
SAMPLE_META = ("element", "setting", "module", "type", "domain", "perspective", "difficulty",
               "grading", "consistency_rule", "base_id", "template_id")


def build_samples(
    benchmark: str,
    format: str = "mc",
    element: str | Iterable[str] | None = None,
    module: str | Iterable[str] | None = None,
    setting: str | Iterable[str] | None = None,
    shots: int = 0,
    prompts: str | dict[str, Any] = DEFAULT_PROMPTS,
    seed: int = 42,
    data_dir: str | None = None,
    include_held: bool = False,
) -> list[Sample]:
    """One Sample per (element, base_id): all parts of a question, asked in sequence.

    ``prompts``: a prompt set name or JSON path (``prompts.load_prompts``) or its strings.
    """
    strings = prompts if isinstance(prompts, dict) else load_prompts(prompts)[1]
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
    if shots < 0:
        raise ValueError("shots must be >= 0")
    if shots and format == "free":
        raise ValueError("few-shot examples are multiple choice; shots must be 0 for format='free'")

    rows = load_rows(benchmark, "test", element, module, setting, data_dir, include_held)
    groups = group_rows(rows)
    examples = group_rows(load_rows(benchmark, "few_shot", element, module, setting, data_dir,
                                    include_held)) if shots else {}
    replace_correct = none_option_selection(groups, seed) if format == "none" else set()

    samples: list[Sample] = []
    prefixes: dict[tuple[Any, ...], tuple[str, int]] = {}
    for key, parts in groups.items():
        first = parts[0]
        grading = first["grading"]
        if format in ("none", "free") and grading != "answer":
            continue  # no correct option to replace / no number to write
        if format == "free" and any(
            parse_gold_number(p["options"][p["correct_index"]]) is None for p in parts
        ):
            continue  # free text is scored numerically; skip non-numeric answers
        if format == "none":
            parts = apply_none_option(parts, key in replace_correct, seed)

        cell = (first["element"], first["type"], first["domain"])
        if cell not in prefixes:
            chosen = select_few_shot(examples, *cell, shots, seed) if shots else []
            prefixes[cell] = (few_shot_prefix(chosen, strings), len(chosen))
        prefix, shots_used = prefixes[cell]

        packed = [{k: p[k] for k in ("question_id", "question_text", "options", "correct_index")}
                  for p in parts]
        offsets = part_offsets([len(p["options"]) for p in packed])
        if grading == "answer":
            target = ",".join(ascii_uppercase[o + p["correct_index"]] for o, p in zip(offsets, packed))
        else:
            target = first["consistency_rule"]
        meta = {k: first.get(k) for k in SAMPLE_META}
        meta.update(
            format=format,
            n_parts=len(parts),
            parts=packed,
            prefix=prefix,
            shots_used=shots_used,  # fewer than `shots` if the element has too few examples
            none_option_replaces_correct=(key in replace_correct) if format == "none" else None,
        )
        shown = (prefix + turns_for_part(format, packed[0], 0, strings)[0])
        samples.append(Sample(id=f"{key[0]}/{key[1]}", input=shown, target=target, metadata=meta))
    if not samples:
        raise ValueError(f"no {benchmark} questions match element={element!r} module={module!r} "
                         f"setting={setting!r} format={format!r}")
    return samples


def _task(benchmark: str, format: str, element: Any, module: Any, setting: Any, shots: int,
          prob_mode: str, logprobs: bool, top_logprobs: int, prompts: str,
          sig_figs: int, answer_max_tokens: int | None, seed: int, data_dir: str | None,
          include_held: bool) -> Task:
    if prob_mode not in ("condition", "mix"):
        raise ValueError(f"prob_mode must be 'condition' or 'mix', got {prob_mode!r}")
    prompts_name, strings = load_prompts(prompts)
    samples = build_samples(benchmark, format, element, module, setting, shots,
                            strings, seed, data_dir, include_held)
    calibration = logprobs and format != "free"
    return Task(
        dataset=MemoryDataset(samples, name=benchmark),
        solver=steer_solver(format, strings, logprobs, top_logprobs, answer_max_tokens),
        scorer=steer_scorer(format, sig_figs),
        metrics=task_metrics(prob_mode, calibration=calibration),
        config=GenerateConfig(temperature=0.0),
        version=TASK_VERSION,
        metadata={"benchmark": benchmark, "steer_bench_version": __version__, "format": format,
                  "shots": shots, "prob_mode": prob_mode, "seed": seed,
                  "prompts": prompts_name, "prompts_hash": prompts_hash(strings),
                  "prompt_strings": strings},
    )


@task
def steer(
    format: str = "mc",
    element: str | list[str] | None = None,
    module: str | list[str] | None = None,
    setting: str | list[str] | None = None,
    shots: int = 0,
    prob_mode: str = "condition",
    logprobs: bool = True,
    top_logprobs: int = 20,
    prompts: str = DEFAULT_PROMPTS,
    sig_figs: int = 3,
    answer_max_tokens: int | None = None,
    seed: int = 42,
    data_dir: str | None = None,
    include_held: bool = False,
) -> Task:
    """STEER (arXiv 2402.09552). See the package README for the parameters."""
    return _task("steer", format, element, module, setting, shots, prob_mode, logprobs,
                 top_logprobs, prompts, sig_figs, answer_max_tokens, seed, data_dir,
                 include_held)


@task
def steer_me(
    format: str = "mc",
    element: str | list[str] | None = None,
    module: str | list[str] | None = None,
    setting: str | list[str] | None = None,
    shots: int = 0,
    prob_mode: str = "condition",
    logprobs: bool = True,
    top_logprobs: int = 20,
    prompts: str = DEFAULT_PROMPTS,
    sig_figs: int = 3,
    answer_max_tokens: int | None = None,
    seed: int = 42,
    data_dir: str | None = None,
    include_held: bool = False,
) -> Task:
    """STEER-ME (arXiv 2502.13119). See the package README for the parameters."""
    return _task("steer_me", format, element, module, setting, shots, prob_mode, logprobs,
                 top_logprobs, prompts, sig_figs, answer_max_tokens, seed, data_dir,
                 include_held)
