"""Per-response rows from Inspect ``.eval`` logs of steer_bench tasks: the public results dataset.

Usage::

    python -m steer_bench.results LOG_OR_DIR [...] --model NAME --out FILE.parquet [--rescore [--letter-from-logprobs]]
    # or: steer-bench results ...

One row per model, question and format, with the question (its parts, options and correct option), the model's
answer (the letter it gave, the option it chose, and the probabilities it put on the option letters when the run
recorded them), the transcript, and the scores. The rows are counted and scored exactly as the scored-cells table
(steer_bench.export, sharing ``read_logs``), so summing them gives the same cells; ``check_against_cells`` compares.

The model column is the name given with ``--model``: the model id in a log can name how the model was served, which
does not belong in the public data, and neither the log's run metadata nor its task arguments (paths) are copied.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .export import STAT_COLS, cells_from_logs, read_logs, rescored_responses
from .scorer import score_group
from .scoring import letter_probs

BENCHMARKS = {"steer": "STEER", "steer_me": "STEER-ME"}
# the columns that identify a row
KEY = ["benchmark", "model", "format", "question_id"]


def _f(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _messages(sample) -> list[dict[str, Any]]:
    """The transcript: role, text, and for assistant turns the reasoning the server returned separately."""
    out = []
    for m in sample.messages:
        reasoning = None
        if m.role == "assistant" and isinstance(m.content, list):
            parts = [c.reasoning for c in m.content if getattr(c, "type", None) == "reasoning" and c.reasoning]
            reasoning = "\n\n".join(parts) or None
        out.append({"role": m.role, "content": m.text, "reasoning": reasoning})
    return out


def _parts(md: dict[str, Any], responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, part in enumerate(md.get("parts") or []):
        r = responses[i] if i < len(responses) else {}
        letters = r.get("letters")
        probs = None
        if letters and r.get("top_logprobs"):
            raw = letter_probs([tuple(t) for t in r["top_logprobs"]], letters)
            probs = [raw.get(L, 0.0) for L in letters]
        out.append({
            "question_id": part.get("question_id"),
            "question": part.get("question_text"),
            "options": list(part.get("options") or []),
            "correct_index": _int(part.get("correct_index")),
            "letter": r.get("letter"),
            "chosen_index": _int(r.get("chosen")),
            "option_probs": probs,   # raw probabilities on the option letters, in option order
        })
    return out


def rows_from_logs(paths: Iterable[str], model: str, rescore_logs: bool = False, from_logprobs: bool = False,
                   tasks: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """The rows of the given logs (see the module docstring), all under the public model name ``model``."""
    rows = []
    for _, header, settings, samples, _ in read_logs(paths, tasks, rescore_logs, from_logprobs):
        task_name = header.eval.task.split("/")[-1]
        args = header.eval.task_args or {}
        meta = header.eval.metadata or {}
        gen = header.plan.config if header.plan else None
        common = {
            "benchmark": BENCHMARKS.get(task_name, task_name), "model": model, "format": settings["format"],
            "data_subset": meta.get("data_subset") or "full", "prompts": settings["prompts"],
            "shots": _int(args.get("shots")) or 0, "logprobs": bool(args.get("logprobs", True)),
            "temperature": _f(getattr(gen, "temperature", None)), "max_tokens": _int(getattr(gen, "max_tokens", None)),
            "steer_bench_version": str(meta.get("steer_bench_version") or ""),
            "inspect_version": str((header.eval.packages or {}).get("inspect_ai", "")),
            "run_id": header.eval.eval_id, "run_date": str(header.eval.created)[:10],
        }
        for s, value in samples.values():
            md = s.metadata or {}
            if rescore_logs and md.get("responses"):
                replies = [m.text for m in s.messages if m.role == "assistant"]
                responses, _ = rescored_responses(md, replies, settings["format"], from_logprobs)
            else:
                responses = md.get("responses") or []
            parts = _parts(md, responses)
            # the answer as the scorer read it: the letter(s), or the free-text value
            _, answer = score_group({**md, 'responses': responses}, settings['format'],
                                    int(args.get('sig_figs', 3) or 3))
            v = {k: _f(x) for k, x in value.items()}
            answered = v.get("answered") == 1.0
            has_probs = answered and v.get("has_probs") == 1.0
            k = len(parts[0]["options"]) if parts and parts[0]["options"] else None
            brier = v.get("brier_avg_cond")
            if brier is None and v.get("brier_cond") is not None and k:
                brier = v["brier_cond"] / k
            usage = s.output.usage if s.output and s.output.usage else None
            rows.append({
                **common,
                "question_id": str(s.id), "element": md.get("element"), "setting": md.get("setting"),
                "module": md.get("module"), "type": md.get("type"), "domain": md.get("domain"),
                "perspective": md.get("perspective"), "difficulty": _int(md.get("difficulty")),
                "template_id": md.get("template_id"), "grading": md.get("grading"),
                "parts": parts, "answer": answer, "messages": _messages(s),
                "answered": answered, "correct": (v.get("correct") == 1.0) if answered else None,
                "normalized_score": v.get("norm") if answered else None,
                "normalized_score_strict": v.get("norm_strict"),
                "p_correct": v.get("p_correct_cond") if has_probs else None,
                "confidence": v.get("conf_cond") if has_probs else None,
                "brier": brier if has_probs else None,
                "input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None),
                "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
                "seconds": _f(s.working_time if s.working_time is not None else s.total_time),
            })
    return rows


def check_against_cells(rows: list[dict[str, Any]], paths: Iterable[str], rescore_logs: bool = False,
                        from_logprobs: bool = False) -> list[str]:
    """Differences between the rows and the cells exporter on the same logs (empty when they agree): question
    counts, answered, correct, normalized score, and the calibration sums."""
    cells, _, _ = cells_from_logs(paths, rescore_logs=rescore_logs, from_logprobs=from_logprobs)
    s = cells[STAT_COLS + ["n_prob"]].sum() if len(cells) else None
    got = {
        "n_rows": len(rows), "n": sum(r["answered"] for r in rows), "n_correct": sum(bool(r["correct"]) for r in rows),
        "sum_norm": sum(r["normalized_score"] or 0.0 for r in rows),
        "n_prob": sum(r["p_correct"] is not None for r in rows),
        "sum_p_correct_cond": sum(r["p_correct"] or 0.0 for r in rows),
        "sum_brier_avg_cond": sum(r["brier"] or 0.0 for r in rows),
    }
    out = []
    for k, x in got.items():
        want = float(s[k]) if s is not None else 0.0
        if abs(x - want) > 1e-6 * max(1.0, abs(want)):
            out.append(f"{k}: rows {x} != cells {want}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", help=".eval log files or directories (searched recursively)")
    ap.add_argument("--model", required=True, help="the model's public name for the model column")
    ap.add_argument("--out", required=True, help="output Parquet file")
    ap.add_argument("--rescore", action="store_true", help="as steer_bench.export --rescore")
    ap.add_argument("--letter-from-logprobs", action="store_true", help="as steer_bench.export")
    ap.add_argument("--no-check", action="store_true", help="skip the comparison with the cells exporter")
    a = ap.parse_args(argv)
    if a.letter_from_logprobs and not a.rescore:
        ap.error("--letter-from-logprobs needs --rescore")
    import pandas as pd

    rows = rows_from_logs(a.logs, a.model, a.rescore, a.letter_from_logprobs)
    if not a.no_check:
        diff = check_against_cells(rows, a.logs, a.rescore, a.letter_from_logprobs)
        if diff:
            print("rows disagree with the cells:", *diff, sep="\n  ", file=sys.stderr)
            return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"{len(rows)} rows ({a.model}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
