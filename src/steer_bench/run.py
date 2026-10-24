"""Run a benchmark on several models and write a report card for each, with an index comparing them.

``steer-bench run steer --models models.txt --out cards/`` runs every model through Inspect's
``eval_set`` (models run side by side; an interrupted or failed run is resumed by running the same
command again) and then writes ``cards/<model>.html`` and ``cards/index.html``.

Models are Inspect model names, one per line in a file (``#`` starts a comment) or comma-separated:
``openai/gpt-4o-mini``, ``vllm/Qwen/Qwen2.5-7B-Instruct``, or ``sllm/qwen-3-8b`` for a model that
slurm-llm serves on a Slurm cluster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

BENCHMARKS = ("steer", "steer_me")


def parse_models(spec: str) -> list[str]:
    """Models from a file (one per line, # comments) or a comma-separated list."""
    path = Path(spec).expanduser()
    if path.is_file():
        lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
        models = [m for m in lines if m]
    else:
        models = [m.strip() for m in spec.split(",") if m.strip()]
    if not models:
        raise ValueError(f"no models in {spec!r}")
    return models


def parse_args_list(pairs: list[str], flag: str) -> dict[str, Any]:
    """key=value options (values parsed as YAML, as Inspect's -T and -M do)."""
    import yaml

    out = {}
    for pair in pairs:
        key, eq, value = pair.partition("=")
        if not eq or not key:
            raise ValueError(f"{flag} takes key=value, not {pair!r}")
        out[key] = yaml.safe_load(value)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="steer-bench run", description=__doc__.split("\n\n")[0])
    ap.add_argument("benchmark", choices=BENCHMARKS)
    ap.add_argument("--models", required=True, help="file with one model per line, or a comma-separated list")
    ap.add_argument("--out", default="cards", help="directory for the report cards and index.html (default: cards)")
    ap.add_argument("--log-dir", default="logs", help="Inspect logs; run again with the same one to resume (default: logs)")
    ap.add_argument("-T", dest="task_args", action="append", default=[], metavar="KEY=VALUE",
                    help="task option, e.g. -T format=mc -T module=... (see the README)")
    ap.add_argument("-M", dest="model_args", action="append", default=[], metavar="KEY=VALUE",
                    help="model option for every model, e.g. -M keep=false for sllm/ models")
    ap.add_argument("--limit", type=int, help="questions per model (to try things out)")
    ap.add_argument("--max-tasks", type=int,
                    help="models run at once (default: Inspect's); with sllm/ models, each one is a GPU job")
    ap.add_argument("--retry-attempts", type=int, help="times to retry a failed model (default: Inspect's)")
    a = ap.parse_args(argv)
    try:
        models = parse_models(a.models)
        task_args = parse_args_list(a.task_args, "-T")
        model_args = parse_args_list(a.model_args, "-M")
    except ValueError as e:
        ap.error(str(e))

    from inspect_ai import eval_set

    from .report import load_inputs, write_cards

    print(f"running {a.benchmark} on {len(models)} model(s); logs in {a.log_dir}")
    options = {k: v for k, v in (("limit", a.limit), ("max_tasks", a.max_tasks), ("retry_attempts", a.retry_attempts))
               if v is not None}
    success, logs = eval_set(tasks=f"steer_bench/{a.benchmark}", model=models, model_args=model_args,
                             task_args=task_args, log_dir=a.log_dir, **options)
    done = [log for log in logs if log.status == "success"]
    if done:
        cells, model_info, bench = load_inputs([a.log_dir], a.benchmark)
        index = write_cards(cells, model_info, bench, Path(a.out))
        print(f"wrote {len(set(cells['run_id']))} report cards; compare them in {index}")
    if not success:
        failed = sorted({log.eval.model for log in logs if log.status != "success"})
        print(f"not finished: {', '.join(failed)}; run the same command again to resume")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
