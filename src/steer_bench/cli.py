"""``steer-bench`` command: ``steer-bench report ...``, ``steer-bench export ...`` and ``steer-bench results ...``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = ("usage: steer-bench {report,export,results} ...\n"
             "  report   report card (HTML/Markdown) from .eval logs or converted cells\n"
             "  export   convert .eval logs to the scored-cells table\n"
             "  results  convert .eval logs to per-response rows (the public results dataset)")
    if not argv or argv[0] in ("-h", "--help"):
        print(usage)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "report":
        from .report import main as run
    elif cmd == "export":
        from .export import main as run
    elif cmd == "results":
        from .results import main as run
    else:
        print(usage, file=sys.stderr)
        return 2
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
