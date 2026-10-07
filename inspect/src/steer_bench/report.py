"""Report cards: one model's results per setting, module and element.

Usage::

    steer-bench report LOGS_OR_CELLS [...] --out report.html [--out report.md]

Input is Inspect ``.eval`` logs (files or directories) of the ``steer`` / ``steer_me`` tasks, or
the output of ``steer-bench export`` (a directory with ``cells.parquet``, or the parquet file).
Everything is computed from the additive scored-cells table, so a report from logs and one from
their converted cells are identical.

The report card has, for each question format that was run (columns):

* a summary: exact match, normalized accuracy, no-answer rate and, where the provider returned
  logprobs, ECE, Brier score and EPA (conditioning by default);
* a heatmap table of normalized accuracy for every element, grouped by setting and module in the
  benchmark's taxonomy order (``taxonomy.json``, a derived copy of the website's content), with
  module and setting averages;
* per format, a table of every metric per element, plus the worst domain (lowest normalized
  accuracy over the element's domains).

Module, setting and overall scores are unweighted means over elements (each element counts
once, whatever its number of questions), as in the papers' summaries.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from .scoring import N_BINS

BENCH_NAMES = {"steer": "STEER", "steer_me": "STEER-ME"}

# column order and labels of the question formats
FORMAT_LABELS = {
    "mc": "Multiple choice",
    "shown": "Reasoning, options shown",
    "hidden": "Reasoning, options hidden",
    "none": "“No other option is correct”",
    "free": "Free text",
}
NAN = float("nan")


# ------------------------------------------------------------------ inputs
def load_taxonomy() -> dict[str, Any]:
    return json.loads(resources.files("steer_bench").joinpath("taxonomy.json").read_text())


def _is_cells(p: Path) -> bool:
    return (p.suffix == ".parquet") or (p.is_dir() and (p / "cells.parquet").exists())


def load_inputs(paths: Iterable[str], benchmark: str | None = None):
    """(cells, models, benchmark) from logs or converted cells."""
    import pandas as pd

    from .export import cells_from_logs

    paths = [Path(p).expanduser() for p in paths]
    if all(_is_cells(p) for p in paths):
        cells, models = [], []
        for p in paths:
            f = p if p.suffix == ".parquet" else p / "cells.parquet"
            cells.append(pd.read_parquet(f))
            if (f.parent / "models.parquet").exists():
                models.append(pd.read_parquet(f.parent / "models.parquet"))
        c = pd.concat(cells, ignore_index=True)
        m = pd.concat(models, ignore_index=True) if models else pd.DataFrame()
        benches = set()
        if len(m) and "benchmark" in m:
            for b in m["benchmark"].dropna():
                benches |= set(str(b).split(";"))
        if benchmark is None:
            benchmark = _infer_benchmark(c, benches)
        return c, m, benchmark
    if any(_is_cells(p) for p in paths):
        raise ValueError("give either .eval logs or converted cells, not both")
    if benchmark is None:
        found = _log_benchmarks(paths)
        if len(found) > 1:
            raise ValueError(f"the logs cover several benchmarks {sorted(found)}; pick one with --benchmark")
        benchmark = next(iter(found), "steer_me")
    cells, models, _ = cells_from_logs([str(p) for p in paths], tasks=[benchmark])
    return cells, models, benchmark


def _log_benchmarks(paths: list[Path]) -> set[str]:
    from inspect_ai.log import read_eval_log

    from .export import TASKS, _log_files

    out = set()
    for f in _log_files([str(p) for p in paths]):
        try:
            t = read_eval_log(str(f), header_only=True).eval.task.split("/")[-1]
        except Exception:  # noqa: BLE001, S112 - unreadable logs are reported by the converter
            continue
        if t in TASKS:
            out.add(t)
    return out


def _infer_benchmark(cells, benches: set[str]) -> str:
    if len(benches) == 1:
        return next(iter(benches))
    tax = load_taxonomy()["benchmarks"]
    els = set(cells["element"])
    cover = {b: len(els & {e["element"] for s in v["settings"] for m in s["modules"] for e in m["elements"]})
             for b, v in tax.items()}
    return max(cover, key=cover.get)


# ------------------------------------------------------------------ metrics from cells
def format_key(question_format: str, car: bool, adaptation: str) -> str:
    expl = int(re.search(r"expl=(-?\d+)", adaptation or "").group(1)) if "expl=" in (adaptation or "") else 0
    if car:
        base = "none"
    elif question_format == "hidden_options":
        base = "hidden"
    elif question_format == "no_options":
        base = "free"
    else:
        base = "shown" if expl > 0 else "mc"
    shots = re.search(r"shots=(\d+)", adaptation or "")
    if shots and int(shots.group(1)) > 0:
        base += f", {shots.group(1)}-shot"
    return base


def prompts_of(adaptation: str) -> str:
    m = re.search(r"prompts=([^;]+)", adaptation or "")
    return m.group(1) if m else "unknown"


def metrics_from_sums(s: dict[str, float], var: str = "cond") -> dict[str, float]:
    """Metrics from summed cell columns (calibration over n_prob; NaN without logprobs)."""
    n, rows = s.get("n", 0), s.get("n_rows", 0)
    n_prob = s.get("n_prob", n)
    out = {
        "n_rows": rows,
        "n": n,
        "exact_match": s["n_correct"] / n if n else NAN,
        "normalized_accuracy": s["sum_norm"] / n if n else NAN,
        "no_answer_rate": s["n_no_answer"] / rows if rows else NAN,
        "n_prob": n_prob,
        "ece": NAN, "brier": NAN, "epa": NAN,
    }
    if n_prob:
        ece = sum(abs(s[f"ece_{var}_acc{i}"] - s[f"ece_{var}_conf{i}"]) for i in range(N_BINS))
        out.update(ece=ece / n_prob, brier=s[f"sum_brier_{var}"] / n_prob, epa=s[f"sum_p_correct_{var}"] / n_prob)
    return out


def _nanmean(xs: Iterable[float]) -> float:
    v = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(v) / len(v) if v else NAN


def build_report(cells, models, benchmark: str, run_id: str | None = None, prob_mode: str = "condition") -> dict[str, Any]:
    """Everything the renderers need, as plain Python data."""
    var = "mix" if prob_mode in ("mix", "mixing") else "cond"
    runs = sorted(set(cells["run_id"]))
    if run_id is None:
        if len(runs) != 1:
            raise ValueError(f"several models in the input {runs}; pick one with --run")
        run_id = runs[0]
    c = cells[cells["run_id"] == run_id].copy()
    if c.empty:
        raise ValueError(f"no cells for run {run_id!r} (have {runs})")
    c["fmt"] = [format_key(q, bool(car), a) for q, car, a in zip(c["question_format"], c["car"], c["adaptation"])]
    c["prompts"] = [prompts_of(a) for a in c["adaptation"]]
    if c["prompts"].nunique() > 1:  # same format under two prompt sets: keep them apart
        c["fmt"] = c["fmt"] + " [" + c["prompts"].str.split("@").str[0] + "]"
    order = list(FORMAT_LABELS)
    fmts = sorted(set(c["fmt"]), key=lambda f: (next((i for i, k in enumerate(order) if f.startswith(k)), 99), f))
    stat = [x for x in c.columns if x.startswith(("n", "sum_", "ece_"))]

    def sums(df) -> dict[str, float]:
        return {k: float(v) for k, v in df[stat].sum().items()}

    per: dict[tuple[str, str], dict[str, Any]] = {}
    for (el, f), g in c.groupby(["element", "fmt"]):
        m = metrics_from_sums(sums(g), var)
        worst = None
        if g["domain"].notna().any():
            dm = [(d, metrics_from_sums(sums(gd), var)) for d, gd in g.groupby("domain")]
            dm = [(d, x) for d, x in dm if x["n"] > 0]
            if len(dm) > 1:
                d, x = min(dm, key=lambda t: t[1]["normalized_accuracy"])
                worst = {"domain": d, "normalized_accuracy": x["normalized_accuracy"], "n": x["n"]}
        m["worst_domain"] = worst
        per[(el, f)] = m

    # taxonomy order; elements not in the taxonomy go to "Other"
    tax = load_taxonomy()["benchmarks"].get(benchmark, {"settings": []})
    present = set(c["element"])
    settings, seen = [], set()
    for s in tax["settings"]:
        mods = []
        for mod in s["modules"]:
            els = [e for e in mod["elements"] if e["element"] in present]
            if els:
                mods.append({"number": mod["number"], "name": mod["name"], "elements": els})
                seen |= {e["element"] for e in els}
        if mods:
            settings.append({"number": s["number"], "name": s["name"], "modules": mods})
    other = sorted(present - seen)
    if other:
        settings.append({"number": "", "name": "Other (not in the taxonomy)", "modules": [
            {"number": "", "name": "", "elements": [{"element": e, "name": e, "id": None, "held": False}
                                                    for e in other]}]})

    def agg(elements: list[str], f: str) -> float:
        return _nanmean(per[(e, f)]["normalized_accuracy"] for e in elements if (e, f) in per)

    for s in settings:
        for mod in s["modules"]:
            names = [e["element"] for e in mod["elements"]]
            mod["avg"] = {f: agg(names, f) for f in fmts}
        all_s = [e["element"] for mod in s["modules"] for e in mod["elements"]]
        s["avg"] = {f: agg(all_s, f) for f in fmts}

    summary = {}
    for f in fmts:
        m = metrics_from_sums(sums(c[c["fmt"] == f]), var)
        m["macro_normalized_accuracy"] = agg(sorted(present), f)
        m["n_elements"] = sum(1 for e in present if (e, f) in per)
        summary[f] = m

    model = run_id.split("/", 1)[1] if run_id.startswith("inspect/") else run_id
    info: dict[str, Any] = {}
    if models is not None and len(models) and "run_id" in models:
        r = models[models["run_id"] == run_id]
        if len(r):
            info = {k: r.iloc[0][k] for k in r.columns if k not in ("run_id",)}
    return {
        "model": model, "run_id": run_id, "benchmark": benchmark,
        "benchmark_name": BENCH_NAMES.get(benchmark, benchmark),
        "prompts": sorted(set(c["prompts"])), "prob_mode": "mixing" if var == "mix" else "conditioning",
        "formats": fmts, "summary": summary, "settings": settings, "per": per, "info": info,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "steer_bench_version": __version__,
    }


def format_label(f: str) -> str:
    base, _, rest = f.partition(",")
    base, _, br = base.partition(" [")
    lab = FORMAT_LABELS.get(base, base)
    if rest:
        lab += "," + rest
    if br:
        lab += " [" + br
    return lab


# ------------------------------------------------------------------ rendering helpers
def _fmt(x: float, nd: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _hex(c: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0, min(255, v))):02x}" for v in c)


def _rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _lum(c) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# diverging blue (good) <-> red (bad), neutral gray midpoint; light and dark steps
_DIV = {"light": ("#d03b33", "#f0efec", "#2a78d6"), "dark": ("#e0524a", "#383835", "#3987e5")}


def cell_colors(v: float, mode: str) -> tuple[str, str]:
    """(background, text) for a normalized accuracy in [-1, 1]."""
    neg, mid, pos = (_rgb(h) for h in _DIV[mode])
    if v is None or math.isnan(v):
        return ("transparent", "inherit")
    t = max(-1.0, min(1.0, v))
    end = pos if t >= 0 else neg
    a = abs(t) ** 0.8
    bg = tuple(m + (e - m) * a for m, e in zip(mid, end))
    fg = "#ffffff" if _lum(bg) < 0.30 else "#0b0b0b"
    return _hex(bg), fg


def _heat_td(v: float, sub: str = "", title: str = "", strong: bool = False) -> str:
    bl, fl = cell_colors(v, "light")
    bd, fd = cell_colors(v, "dark")
    style = f"--bl:{bl};--fl:{fl};--bd:{bd};--fd:{fd}"
    cls = "heat strong" if strong else "heat"
    sub_html = f"<small>{html.escape(sub)}</small>" if sub else ""
    return (f'<td class="{cls}" style="{style}" title="{html.escape(title)}">'
            f"<span>{_fmt(v)}</span>{sub_html}</td>")


# ------------------------------------------------------------------ HTML
CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--rule:#e2e1dc;--head:#f4f3f0;--neg:#d03b33;--mid:#f0efec;
--pos:#2a78d6;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--fg:#f4f3ef;--fg2:#c3c2b7;
--rule:#3a3a37;--head:#242422;--neg:#e0524a;--mid:#383835;--pos:#3987e5;color-scheme:dark}
:root:not([data-theme=light]) td.heat{background:var(--bd);color:var(--fd)}}
:root[data-theme=dark]{--bg:#1a1a19;--fg:#f4f3ef;--fg2:#c3c2b7;--rule:#3a3a37;--head:#242422;--neg:#e0524a;
--mid:#383835;--pos:#3987e5;color-scheme:dark}
:root[data-theme=dark] td.heat{background:var(--bd);color:var(--fd)}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
margin:0;padding:24px 16px 48px}
main{max-width:1100px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}h3{font-size:15px;margin:20px 0 6px}
p.meta,.note{color:var(--fg2)}p.meta{margin:0 0 16px}
.wrap{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{font-weight:600;background:var(--head);position:sticky;top:0}
td.l,th.l{text-align:left;white-space:normal}
tr.set td{background:var(--head);font-weight:700;text-align:left}
tr.mod td.l{font-weight:600;padding-left:16px}
tr.el td.l{padding-left:32px}
td.heat{background:var(--bl);color:var(--fl);text-align:center;border:2px solid var(--bg);min-width:84px}
td.heat span{font-weight:600}td.heat small{display:block;font-size:11px;opacity:.85}
td.heat.strong span{font-weight:800}
.legend{display:flex;align-items:center;gap:8px;color:var(--fg2);font-size:12px;margin:6px 0 10px}
.legend .bar{width:220px;height:10px;border-radius:2px;background:linear-gradient(90deg,var(--neg),var(--mid),var(--pos))}
details{margin:8px 0}summary{cursor:pointer;font-weight:600}
code{font-size:12px}
"""


def render_html(r: dict[str, Any]) -> str:
    fmts = r["formats"]
    esc = html.escape
    title = f"{r['model']} on {r['benchmark_name']}"
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>{esc(r['benchmark_name'])} report card: {esc(r['model'])}</title>",
           f"<style>{CSS}</style></head><body><main>",
           f"<h1>{esc(title)}</h1>",
           (f"<p class='meta'>{esc(r['benchmark_name'])} report card · prompts {esc(', '.join(r['prompts']))} · "
            f"calibration by {esc(r['prob_mode'])} · steer-bench {esc(r['steer_bench_version'])} · "
            f"generated {esc(r['generated'])}</p>")]

    # summary
    out.append("<h2>Summary</h2><div class='wrap'><table><tr><th class='l'>format</th><th>elements</th>"
               "<th>questions</th><th>normalized accuracy<br><small>mean over elements</small></th>"
               "<th>normalized accuracy<br><small>pooled</small></th><th>exact match</th>"
               "<th>no answer</th><th>ECE</th><th>Brier</th><th>EPA</th></tr>")
    for f in fmts:
        s = r["summary"][f]
        out.append(f"<tr><td class='l'>{esc(format_label(f))}</td><td>{s['n_elements']}</td>"
                   f"<td>{int(s['n_rows'])}</td><td><b>{_fmt(s['macro_normalized_accuracy'])}</b></td>"
                   f"<td>{_fmt(s['normalized_accuracy'])}</td><td>{_fmt(s['exact_match'])}</td>"
                   f"<td>{_fmt(s['no_answer_rate'])}</td><td>{_fmt(s['ece'])}</td><td>{_fmt(s['brier'])}</td>"
                   f"<td>{_fmt(s['epa'])}</td></tr>")
    out.append("</table></div>")
    if all(math.isnan(r["summary"][f]["ece"]) for f in fmts):
        out.append("<p class='note'>No option probabilities in these results (the provider returned no "
                   "logprobs, or the run used <code>logprobs=false</code>), so there are no calibration "
                   "metrics.</p>")

    # settings overview
    out.append("<h2>By setting</h2><div class='wrap'><table><tr><th class='l'>setting</th>"
               + "".join(f"<th>{esc(format_label(f))}</th>" for f in fmts) + "</tr>")
    for s in r["settings"]:
        name = f"{s['number']}. {s['name']}" if s["number"] != "" else s["name"]
        out.append(f"<tr><td class='l'>{esc(name)}</td>"
                   + "".join(_heat_td(s["avg"][f], title=f"{s['name']}: mean normalized accuracy over elements")
                             for f in fmts) + "</tr>")
    out.append("</table></div>")

    # element heatmap
    out.append("<h2>Elements</h2><div class='legend'><span>−1</span><span class='bar'></span><span>+1</span>"
               "<span>normalized accuracy (0 = random guessing); small figure: exact match</span></div>")
    out.append("<div class='wrap'><table><tr><th class='l'>setting / module / element</th><th>questions</th>"
               + "".join(f"<th>{esc(format_label(f))}</th>" for f in fmts) + "</tr>")
    for s in r["settings"]:
        name = f"{s['number']}. {s['name']}" if s["number"] != "" else s["name"]
        out.append(f"<tr class='set'><td colspan='{len(fmts) + 2}'>{esc(name)}</td></tr>")
        for mod in s["modules"]:
            if mod["name"]:
                out.append(f"<tr class='mod'><td class='l'>{esc(str(mod['number']))} {esc(mod['name'])}</td><td></td>"
                           + "".join(_heat_td(mod["avg"][f], title=f"{mod['name']}: mean over elements", strong=True)
                                     for f in fmts) + "</tr>")
            for e in mod["elements"]:
                el = e["element"]
                n = max((int(r["per"][(el, f)]["n_rows"]) for f in fmts if (el, f) in r["per"]), default=0)
                label = f"{e['id']} {e['name']}" if e.get("id") else e["name"]
                tds = []
                for f in fmts:
                    m = r["per"].get((el, f))
                    if m is None:
                        tds.append("<td></td>")
                        continue
                    tip = (f"{e['name']} · {format_label(f)}: n={int(m['n_rows'])}, exact match {_fmt(m['exact_match'])}, "
                           f"normalized accuracy {_fmt(m['normalized_accuracy'])}, no answer {_fmt(m['no_answer_rate'])}")
                    tds.append(_heat_td(m["normalized_accuracy"], sub=f"EM {_fmt(m['exact_match'])}", title=tip))
                out.append(f"<tr class='el'><td class='l' title='{esc(el)}'>{esc(label)}</td><td>{n}</td>"
                           + "".join(tds) + "</tr>")
    out.append("</table></div>")

    # per-format detail
    out.append("<h2>All metrics by format</h2>")
    for f in fmts:
        out.append(f"<details><summary>{esc(format_label(f))}</summary><div class='wrap'><table>"
                   "<tr><th class='l'>element</th><th>questions</th><th>answered</th><th>exact match</th>"
                   "<th>normalized accuracy</th><th>no answer</th><th>ECE</th><th>Brier</th><th>EPA</th>"
                   "<th class='l'>worst domain</th></tr>")
        for s in r["settings"]:
            for mod in s["modules"]:
                for e in mod["elements"]:
                    m = r["per"].get((e["element"], f))
                    if m is None:
                        continue
                    w = m["worst_domain"]
                    wd = f"{esc(str(w['domain']))} ({_fmt(w['normalized_accuracy'])}, n={int(w['n'])})" if w else "–"
                    out.append(f"<tr><td class='l'>{esc(e['name'])}</td><td>{int(m['n_rows'])}</td><td>{int(m['n'])}</td>"
                               f"<td>{_fmt(m['exact_match'])}</td><td>{_fmt(m['normalized_accuracy'])}</td>"
                               f"<td>{_fmt(m['no_answer_rate'])}</td><td>{_fmt(m['ece'])}</td>"
                               f"<td>{_fmt(m['brier'])}</td><td>{_fmt(m['epa'])}</td><td class='l'>{wd}</td></tr>")
        out.append("</table></div></details>")

    out.append("<h2>Definitions</h2><p class='note'>"
               "<b>Exact match</b>: fraction of answered questions answered correctly (multi-part: every part; "
               "consistency-graded: the answers satisfy the element's rule). "
               "<b>Normalized accuracy</b>: +1 for a correct answer, −1/(n−1) for a wrong one with n options "
               "(consistency groups: −p/(1−p), p the chance that random answers are consistent), so random "
               "guessing scores 0 and a perfect score is 1. "
               "<b>No answer</b>: fraction of questions without a usable answer, excluded from the other metrics. "
               "<b>ECE</b> (10 equal-width bins), <b>Brier</b> and <b>EPA</b> (mean probability on the correct "
               "option) use the option probabilities from the answer token's logprobs, first part only, and are "
               "blank when the provider returned none. Setting and module scores are unweighted means over their "
               "elements. <b>Worst domain</b>: the element's domain with the lowest normalized accuracy.</p>")
    info = r.get("info") or {}
    if info:
        keep = ["model", "benchmark", "prompts", "task_versions", "steer_bench_versions", "inspect_versions",
                "revisions", "eval_ids", "run_date_min", "run_date_max", "has_logprobs"]
        rows = "".join(f"<tr><td class='l'>{esc(k)}</td><td class='l'><code>{esc(str(info[k]))}</code></td></tr>"
                       for k in keep if k in info)
        out.append(f"<h2>Provenance</h2><div class='wrap'><table>{rows}</table></div>")
    out.append("</main></body></html>")
    return "\n".join(out)


# ------------------------------------------------------------------ Markdown
def render_markdown(r: dict[str, Any]) -> str:
    fmts = r["formats"]
    L = [f"# {r['model']} on {r['benchmark_name']}", "",
         (f"{r['benchmark_name']} report card · prompts {', '.join(r['prompts'])} · calibration by "
          f"{r['prob_mode']} · steer-bench {r['steer_bench_version']} · generated {r['generated']}"), "",
         "## Summary", "",
         ("| format | elements | questions | norm. acc. (mean over elements) | norm. acc. (pooled) | exact match "
          "| no answer | ECE | Brier | EPA |"), "|---|" + "---:|" * 9]
    for f in fmts:
        s = r["summary"][f]
        L.append(f"| {format_label(f)} | {s['n_elements']} | {int(s['n_rows'])} | **{_fmt(s['macro_normalized_accuracy'])}** "
                 f"| {_fmt(s['normalized_accuracy'])} | {_fmt(s['exact_match'])} | {_fmt(s['no_answer_rate'])} "
                 f"| {_fmt(s['ece'])} | {_fmt(s['brier'])} | {_fmt(s['epa'])} |")
    L += ["", "## Elements", "",
          ("Normalized accuracy (0 = random guessing, 1 = perfect), with exact match in parentheses. "
           "Module and setting rows are unweighted means over their elements."), "",
          "| setting / module / element | questions | " + " | ".join(format_label(f) for f in fmts) + " |",
          "|---|---:|" + "---:|" * len(fmts)]
    for s in r["settings"]:
        name = f"{s['number']}. {s['name']}" if s["number"] != "" else s["name"]
        L.append(f"| **{name}** | | " + " | ".join(f"**{_fmt(s['avg'][f])}**" for f in fmts) + " |")
        for mod in s["modules"]:
            if mod["name"]:
                L.append(f"| *{mod['number']} {mod['name']}* | | " + " | ".join(f"*{_fmt(mod['avg'][f])}*" for f in fmts) + " |")
            for e in mod["elements"]:
                el = e["element"]
                n = max((int(r["per"][(el, f)]["n_rows"]) for f in fmts if (el, f) in r["per"]), default=0)
                cells = []
                for f in fmts:
                    m = r["per"].get((el, f))
                    cells.append("" if m is None else f"{_fmt(m['normalized_accuracy'])} ({_fmt(m['exact_match'])})")
                label = f"{e['id']} {e['name']}" if e.get("id") else e["name"]
                L.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    if not all(math.isnan(r["summary"][f]["ece"]) for f in fmts):
        L += ["", "## Calibration", "", "ECE / Brier / EPA per element (first part's option probabilities).", "",
              "| element | " + " | ".join(format_label(f) for f in fmts) + " |", "|---|" + "---|" * len(fmts)]
        for s in r["settings"]:
            for mod in s["modules"]:
                for e in mod["elements"]:
                    row = []
                    for f in fmts:
                        m = r["per"].get((e["element"], f))
                        row.append("" if m is None or math.isnan(m["ece"]) else
                                   f"{_fmt(m['ece'])} / {_fmt(m['brier'])} / {_fmt(m['epa'])}")
                    L.append(f"| {e['name']} | " + " | ".join(row) + " |")
    L += ["", "Definitions: see the HTML report card or the package README (Metrics).", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="steer-bench report", description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help=".eval logs or directories, or steer-bench export output")
    ap.add_argument("--out", action="append", required=True, help="output file, .html or .md (repeatable)")
    ap.add_argument("--benchmark", choices=sorted(BENCH_NAMES), help="default: inferred from the input")
    ap.add_argument("--run", help="run_id to report (default: the only one in the input)")
    ap.add_argument("--prob-mode", default="condition", choices=["condition", "mix"])
    a = ap.parse_args(argv)
    cells, models, bench = load_inputs(a.inputs, a.benchmark)
    rep = build_report(cells, models, bench, a.run, a.prob_mode)
    for o in a.out:
        p = Path(o)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in (".html", ".htm"):
            p.write_text(render_html(rep))
        elif p.suffix.lower() in (".md", ".markdown"):
            p.write_text(render_markdown(rep))
        else:
            ap.error(f"--out must end in .html or .md: {o}")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
