"""Convert Inspect ``.eval`` logs of steer_bench tasks into the scored-cells table.

Usage::

    python -m steer_bench.export LOG_OR_DIR [...] --out OUT_DIR
    # or: steer-bench export LOG_OR_DIR [...] --out OUT_DIR

Writes three files to ``OUT_DIR``:

* ``cells.parquet``, the scored-cells table: one row per cell, a cell being one combination of
  the key columns ``run_id`` (``inspect/<model>``), ``element``, ``question_format``, ``car``,
  ``adaptation``, ``domain``, ``type``, ``perspective`` and ``difficulty``. The other columns are
  additive counts and sums over the questions in the cell, so cells from different runs or logs
  can be concatenated and any metric recomputed at any level of aggregation by summing first:

  - ``n_rows`` questions; ``n_no_answer`` without a usable answer; ``n`` answered (the
    denominator of the accuracy metrics); ``n_prob`` answered with option probabilities (the
    denominator of the calibration metrics; 0 when the provider returned no logprobs);
  - ``n_correct`` (exact match = ``n_correct / n``), ``sum_norm`` (normalized accuracy =
    ``sum_norm / n``), ``sum_norm_strict`` (the same with unanswered questions counted wrong);
  - ``sum_p_correct_{cond,mix}`` (EPA), ``sum_brier_{cond,mix}``, ``sum_conf_{cond,mix}``,
    ``n_top_correct``, ``sum_invalid_mass`` and ``n_alpha_zero`` for the two ways of handling
    probability outside the option letters (conditioning, mixing);
  - ``ece_{cond,mix}_{n,conf,acc}{0..9}``: per confidence bin [i/10, (i+1)/10), the count, the
    summed confidence and the number with the top option correct; ECE =
    ``sum_i |acc_i - conf_i| / n_prob``;
  - ``n_timed`` and ``sum_inference_time`` (seconds); ``n_last_turn`` is always 0 here.

* ``models.parquet``: one row per run (model, benchmarks, prompt sets, package and Inspect
  versions, eval ids, dates, whether logprobs were present).
* ``provenance.json``: the command, the logs read or skipped, and the number of cells.

Formats map to the key columns as::

    format   question_format  car    adaptation expl=
    mc       shown_options    False  0
    none     shown_options    True   0
    shown    shown_options    False  3
    hidden   hidden_options   False  3
    free     no_options       False  3

``car`` marks the "No other option is correct" variant and ``expl=3`` the reasoning formats.
``adaptation`` records shots, reasoning and decoding, and ends with ``;prompts=<name>@<hash>``:
the prompt set and a hash of its strings (``unknown`` for logs written before prompt sets were
recorded).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scoring import N_BINS, ece_bin

KEYS = ["run_id", "element", "question_format", "car", "adaptation",
        "domain", "type", "perspective", "difficulty"]

# per-question score key -> cell column, summed over answered rows
_VALID_SUMS = {
    "correct": "n_correct",
    "norm": "sum_norm",
    "p_correct_cond": "sum_p_correct_cond",
    "p_correct_mix": "sum_p_correct_mix",
    "conf_cond": "sum_conf_cond",
    "conf_mix": "sum_conf_mix",
    "top_correct": "n_top_correct",
    "brier_cond": "sum_brier_cond",
    "brier_mix": "sum_brier_mix",
    "invalid_mass": "sum_invalid_mass",
    "alpha_zero": "n_alpha_zero",
    "last_turn": "n_last_turn",
}

STAT_COLS = (["n_rows", "n_no_answer", "n", "sum_norm_strict"] + list(_VALID_SUMS.values())
             + ["n_timed", "sum_inference_time"]
             + [f"ece_{v}_{s}{i}" for v in ("cond", "mix") for s in ("n", "conf", "acc") for i in range(N_BINS)])
EXTRA_COLS = ["n_prob"]

FORMAT_MAP = {  # format -> (question_format, car, uses explanation)
    "mc": ("shown_options", False, False),
    "none": ("shown_options", True, False),
    "shown": ("shown_options", False, True),
    "hidden": ("hidden_options", False, True),
    "free": ("no_options", False, True),
}
SCORER = "steer_scorer"
TASKS = {"steer", "steer_me"}


def _f(x: Any) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x


def _z(x: float) -> float:
    return 0.0 if math.isnan(x) else x


def _log_files(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(p.rglob("*.eval")) + sorted(p.rglob("*.json"))
        else:
            out.append(p)
    return out


def prompts_label(task_metadata: dict[str, Any] | None) -> str:
    md = task_metadata or {}
    if not md.get("prompts"):
        return "unknown"
    return f"{md['prompts']}@{md.get('prompts_hash', '')}"


def task_settings(task_args: dict[str, Any], task_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    fmt = task_args.get("format", "mc")
    qf, car, uses_expl = FORMAT_MAP[fmt]
    shots = int(task_args.get("shots", 0) or 0)
    expl = 3 if uses_expl else 0
    decoding = "logprobs" if task_args.get("logprobs", True) else "text"
    prompts = prompts_label(task_metadata)
    adaptation = (f"shots={shots};expl={expl};explprompt=0;reps=0;retries=0;decoding={decoding}"
                  f";prompts={prompts}")
    return {"format": fmt, "question_format": qf, "car": car, "adaptation": adaptation,
            "prompts": prompts}


def sample_row(value: dict[str, Any], metadata: dict[str, Any], seconds: float | None) -> dict[str, float]:
    """One sample's contribution to its cell (the per-question columns, already summed form)."""
    v = {k: _f(x) for k, x in value.items()}
    answered = v.get("answered", 0.0) == 1.0
    has_probs = answered and v.get("has_probs", 0.0) == 1.0
    r: dict[str, float] = {c: 0.0 for c in STAT_COLS + EXTRA_COLS}
    r["n_rows"] = 1
    r["n_no_answer"] = 0 if answered else 1
    r["n"] = 1 if answered else 0
    r["sum_norm_strict"] = _z(v.get("norm_strict", math.nan))
    if answered:
        for src, dst in _VALID_SUMS.items():
            if src == "alpha_zero":
                r[dst] = 1.0 if has_probs and v.get("alpha", 1.0) <= 0 else 0.0
            elif src == "last_turn":
                r[dst] = 0.0
            else:
                r[dst] = _z(v.get(src, math.nan))
        if seconds is not None and not math.isnan(seconds):
            r["n_timed"] = 1
            r["sum_inference_time"] = seconds
    if has_probs:
        r["n_prob"] = 1
        for var in ("cond", "mix"):
            c = v[f"conf_{var}"]
            b = ece_bin(c)
            r[f"ece_{var}_n{b}"] = 1
            r[f"ece_{var}_conf{b}"] = c
            r[f"ece_{var}_acc{b}"] = v["top_correct"]
    return r


def cells_from_logs(paths: Iterable[str], tasks: Iterable[str] | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """(cells DataFrame, models DataFrame, provenance dict) for the given logs/directories.

    ``tasks``: keep only logs of these tasks (``steer`` / ``steer_me``; default both).
    """
    keep = set(tasks) if tasks is not None else TASKS
    import pandas as pd
    from inspect_ai.log import read_eval_log, read_eval_log_samples

    rows: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {}
    logs_used, skipped = [], []
    for path in _log_files(paths):
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception as e:  # noqa: BLE001 - anything unreadable is skipped and reported
            skipped.append({"path": str(path), "reason": f"unreadable: {e}"})
            continue
        task_name = header.eval.task.split("/")[-1]
        if task_name not in keep:
            skipped.append({"path": str(path), "reason": f"task {header.eval.task}"})
            continue
        settings = task_settings(header.eval.task_args or {}, header.eval.metadata)
        model = header.eval.model
        run_id = f"inspect/{model}"
        # A log written by `inspect eval-retry` can hold the same sample twice (the copy started
        # by the interrupted run and the finished one); keep the last copy of each (id, epoch).
        by_sample: dict[tuple[Any, int], dict[str, Any]] = {}
        for s in read_eval_log_samples(str(path), all_samples_required=False):
            if not s.scores or SCORER not in s.scores:
                continue
            md = s.metadata or {}
            value = s.scores[SCORER].value
            if not isinstance(value, dict):
                continue
            seconds = s.working_time if s.working_time is not None else s.total_time
            r = sample_row(value, md, seconds)
            r.update(
                run_id=run_id, element=md.get("element"),
                question_format=settings["question_format"], car=settings["car"],
                adaptation=settings["adaptation"], domain=md.get("domain"), type=md.get("type"),
                perspective=md.get("perspective"), difficulty=md.get("difficulty"),
            )
            by_sample[(s.id, s.epoch)] = r
        rows.extend(by_sample.values())
        n_samples = len(by_sample)
        m = models.setdefault(run_id, {
            "run_id": run_id, "model": model, "source": "inspect", "benchmark": set(),
            "has_logprobs": False, "n_rows": 0, "eval_ids": [], "task_versions": set(),
            "steer_bench_versions": set(), "inspect_versions": set(), "revisions": set(),
            "prompts": set(),
            "run_date_min": None, "run_date_max": None,
        })
        m["benchmark"].add(task_name)
        m["n_rows"] += n_samples
        m["eval_ids"].append(header.eval.eval_id)
        m["task_versions"].add(str(header.eval.task_version))
        m["prompts"].add(settings["prompts"])
        m["steer_bench_versions"].add(str((header.eval.metadata or {}).get("steer_bench_version")))
        m["inspect_versions"].add(str(header.eval.packages.get("inspect_ai")) if header.eval.packages else "")
        if header.eval.revision is not None:
            m["revisions"].add(f"{header.eval.revision.origin}@{header.eval.revision.commit}")
        created = header.eval.created
        m["run_date_min"] = min(filter(None, [m["run_date_min"], created]))
        m["run_date_max"] = max(filter(None, [m["run_date_max"], created]))
        logs_used.append({"path": str(path), "eval_id": header.eval.eval_id, "task": header.eval.task,
                          "model": model, "status": header.status, "n_samples": n_samples,
                          "task_args": header.eval.task_args})

    if rows:
        df = pd.DataFrame(rows)
        cells = df.groupby(KEYS, dropna=False, sort=True)[STAT_COLS + EXTRA_COLS].sum().reset_index()
    else:
        cells = pd.DataFrame(columns=KEYS + STAT_COLS + EXTRA_COLS)
    for c in STAT_COLS + EXTRA_COLS:
        # counts are int32, sums float64
        is_int = c.startswith(("n", "ece_cond_n", "ece_mix_n")) or "_acc" in c
        cells[c] = cells[c].astype("int32" if is_int else "float64")
    # has_logprobs: any answered row with option probabilities
    if len(cells):
        with_probs = set(cells.loc[cells["n_prob"] > 0, "run_id"])
        for rid, m in models.items():
            m["has_logprobs"] = rid in with_probs
    models_df = pd.DataFrame([
        {**m, **{k: ";".join(sorted(m[k])) for k in
                 ("benchmark", "task_versions", "steer_bench_versions", "inspect_versions", "revisions",
                  "prompts")},
         "eval_ids": ";".join(m["eval_ids"])}
        for m in models.values()
    ])
    provenance = {
        "schema": "steer-me-scored-cells/1 (+n_prob)",
        "built_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "logs": logs_used,
        "skipped": skipped,
        "n_cells": len(cells),
    }
    return cells, models_df, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", help=".eval log files or directories (searched recursively)")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)
    cells, models, prov = cells_from_logs(a.logs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out / "cells.parquet", index=False)
    models.to_parquet(out / "models.parquet", index=False)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"{len(cells)} cells from {len(prov['logs'])} logs ({len(prov['skipped'])} skipped) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
