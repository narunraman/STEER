"""Loading STEER / STEER-ME rows and packing them into Inspect samples.

Every assumption about the dataset schema lives in this module, in ``normalize_row`` and the
source functions below. The rest of the package only sees the canonical row dict that
``normalize_row`` returns (keys in ``ROW_KEYS``).

Data sources:

* By default, the Hugging Face Hub: ``HF_REPOS`` at the commits pinned in ``HF_REVISIONS``. Each
  dataset has one config per element (``data/<element>/{test,few_shot}.parquet``) plus
  ``default`` (all elements); ``setting`` and ``module`` are columns.
* ``data_dir`` (or the ``STEER_BENCH_DATA_DIR`` environment variable): a local copy of a dataset
  repository, i.e. a directory with ``elements.csv`` and ``data/<element>/<split>.parquet``, or a
  parent holding one directory per benchmark (``steer``, ``steer-me``; case, ``-`` and ``_`` are
  ignored). The pre-release staging layout ``data/<setting>/<module>/<element>/<split>.parquet``
  is also read.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BENCHMARKS = ("steer", "steer_me")

HF_REPOS: dict[str, str] = {"steer": "narunraman/steer", "steer_me": "narunraman/steer_me"}
# Commit SHAs of the dataset repositories this package version is evaluated on. None means the
# latest revision (with a warning); tests/test_hub.py fails for a release version with any None.
HF_REVISIONS: dict[str, str | None] = {"steer": None, "steer_me": None}

DATA_DIR_ENV = "STEER_BENCH_DATA_DIR"

# Held back from release (status "held" in the site's elements.json, autosteer
# web/content, October 2026). Loaded only when named explicitly or with include_held.
HELD_ELEMENTS = {
    "steer": {"monotone_sc", "sc_axioms", "sc_dictatorship", "sc_monotonicity", "sc_pareto",
              "sc_transitivity"},
    "steer_me": set(),
}

# Canonical row keys used by the rest of the package.
ROW_KEYS = (
    "element", "question_id", "base_id", "part", "question_text", "options", "correct_index",
    "grading", "consistency_group", "consistency_rule", "type", "domain", "perspective", "tags",
    "difficulty", "template_id", "setting", "module", "explanation",
)


# --------------------------------------------------------------------------------------------
# Schema: staged parquet -> canonical row
# --------------------------------------------------------------------------------------------
def _clean(x: Any) -> Any:
    """None for NaN/None, plain Python lists for numpy arrays."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        return x.tolist()
    return x


def normalize_row(raw: dict[str, Any], setting: str | None = None, module: str | None = None) -> dict[str, Any]:
    """Map one dataset record to the canonical row.

    Schema assumptions (staged packages of 2026-09-28):
    * ``setting``/``module`` are not columns in the staged files; they come from the file path
      (``setting``/``module`` arguments). If a future schema adds the columns, they win.
    * ``source``/``repair`` (staged) vs ``generator``/``seed`` (proposed) are provenance only
      and are not used for evaluation.
    * ``correct_index`` is null for consistency-graded rows; ``consistency_rule`` is
      ``<rule_id>:<allowed>`` with one digit (0-based option index within the part) per part.
    """
    r = {k: _clean(raw.get(k)) for k in raw}
    ci = r.get("correct_index")
    row = {
        "element": r["element"],
        "question_id": r.get("question_id") or f"{r['element']}/{r['base_id']}/{r.get('part', 0)}",
        "base_id": str(r["base_id"]),
        "part": int(r.get("part") or 0),
        "question_text": r["question_text"],
        "options": [str(o) for o in (r.get("options") or [])],
        "correct_index": None if ci is None else int(ci),
        "grading": r.get("grading") or ("answer" if ci is not None else "consistency"),
        "consistency_group": r.get("consistency_group"),
        "consistency_rule": r.get("consistency_rule"),
        "type": r.get("type") or "default",
        "domain": r.get("domain"),
        "perspective": r.get("perspective"),
        "tags": list(r.get("tags") or []),
        "difficulty": None if r.get("difficulty") is None else int(r["difficulty"]),
        "template_id": r.get("template_id"),
        "setting": r.get("setting") or setting,
        "module": r.get("module") or module,
        "explanation": r.get("explanation"),
    }
    return row


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------
def _as_list(x: str | Iterable[str] | None) -> list[str] | None:
    if x is None:
        return None
    if isinstance(x, str):
        items = [s.strip() for s in x.split(",")]
    else:
        items = [str(s).strip() for s in x]
    items = [s for s in items if s]
    return items or None


def _norm_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _slug(name: str | None) -> str | None:
    if not name:
        return None
    return "_".join("".join(c if c.isalnum() else " " for c in name.lower()).split())


def resolve_data_dir(benchmark: str, data_dir: str | None) -> Path | None:
    """The local dataset directory for ``benchmark``, or None to use the Hub."""
    d = data_dir or os.environ.get(DATA_DIR_ENV)
    if not d:
        return None
    p = Path(d).expanduser()
    if (p / "elements.csv").exists():
        return p
    for child in sorted(p.iterdir()) if p.is_dir() else []:
        if child.is_dir() and _norm_name(child.name) == benchmark and (child / "elements.csv").exists():
            return child
    raise FileNotFoundError(
        f"{p} is neither a {benchmark} dataset directory (no elements.csv) nor a directory containing "
        f"one (expected a subdirectory named like '{benchmark.replace('_', '-')}')."
    )


def _element_table(pkg: Path) -> dict[str, dict[str, str]]:
    with open(pkg / "elements.csv", newline="") as f:
        return {r["element"]: r for r in csv.DictReader(f)}


def _matches(value: str | None, names: Iterable[str | None], wanted: list[str] | None) -> bool:
    """True if ``wanted`` is None or names one of ``value``/``names`` (as given or as a slug)."""
    if wanted is None:
        return True
    have = {n.lower() for n in [value, *names] if n} | {_slug(n) for n in [value, *names] if n}
    return any(w.lower() in have or _slug(w) in have for w in wanted)


def _local_files(pkg: Path, split: str) -> list[tuple[str, Path]]:
    """(element, path) for every ``<split>.parquet``: ``data/<element>/`` (release layout) or
    ``data/<setting>/<module>/<element>/`` (pre-release staging layout)."""
    files = sorted(pkg.glob(f"data/*/{split}.parquet")) + sorted(pkg.glob(f"data/*/*/*/{split}.parquet"))
    return [(path.parent.name, path) for path in files]


def _keep(row: dict[str, Any], elements: list[str] | None, modules: list[str] | None,
          settings: list[str] | None, held: set[str]) -> bool:
    if elements and row["element"] not in elements:
        return False
    if row["element"] in held:
        return False
    return _matches(row["module"], [], modules) and _matches(row["setting"], [], settings)


def load_rows(
    benchmark: str,
    split: str = "test",
    element: str | Iterable[str] | None = None,
    module: str | Iterable[str] | None = None,
    setting: str | Iterable[str] | None = None,
    data_dir: str | None = None,
    include_held: bool = False,
) -> list[dict[str, Any]]:
    """Canonical rows of one split, filtered by element / module / setting.

    ``element``, ``module`` and ``setting`` take a name or a comma-separated list. Module and
    setting match the full name (``Comparative Statics of Demand``) or its slug
    (``comparative_statics_of_demand``), case-insensitively.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}, got {benchmark!r}")
    elements, modules, settings = _as_list(element), _as_list(module), _as_list(setting)
    held = set() if include_held else HELD_ELEMENTS[benchmark] - set(elements or [])

    pkg = resolve_data_dir(benchmark, data_dir)
    if pkg is None:
        return load_rows_hub(benchmark, split, elements, modules, settings, held)

    import pyarrow.parquet as pq

    table = _element_table(pkg)
    known = {e for e, _ in _local_files(pkg, "test")} | set(table)
    if elements:
        unknown = [e for e in elements if e not in known]
        if unknown:
            raise ValueError(f"unknown {benchmark} element(s): {unknown}")
    rows: list[dict[str, Any]] = []
    for el, path in _local_files(pkg, split):
        if (elements and el not in elements) or el in held:
            continue
        meta = table.get(el, {})
        # setting/module: columns in the release files; else elements.csv; else the staging path
        staged = path.parent.parent.parent.name if path.parent.parent.name != "data" else None
        default_setting = meta.get("setting") or staged
        default_module = meta.get("module") or (path.parent.parent.name if staged else None)
        extra_modules = [meta.get("module_slug"), path.parent.parent.name if staged else None]
        extra_settings = [meta.get("setting_slug"), staged]
        for raw in pq.read_table(path).to_pylist():
            row = normalize_row(raw, setting=default_setting, module=default_module)
            if (_matches(row["module"], extra_modules, modules)
                    and _matches(row["setting"], extra_settings, settings)):
                rows.append(row)
    return rows


def load_rows_hub(
    benchmark: str,
    split: str = "test",
    elements: list[str] | None = None,
    modules: list[str] | None = None,
    settings: list[str] | None = None,
    held: set[str] | None = None,
    repo: str | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Rows from the Hugging Face dataset (``repo``/``revision`` default to ``HF_REPOS`` and
    ``HF_REVISIONS``; ``repo`` may also be a local copy of the repository).

    Named elements are loaded from their own configs; otherwise the ``default`` config. An
    element without a ``few_shot`` split contributes no few-shot rows.
    """
    import warnings

    import datasets

    repo = repo or HF_REPOS[benchmark]
    if revision is None and repo == HF_REPOS[benchmark]:
        revision = HF_REVISIONS[benchmark]
        if revision is None:
            warnings.warn(f"{repo}: no pinned revision in steer_bench.data.HF_REVISIONS; "
                          "loading the latest one", stacklevel=2)
    held = held or set()
    rows: list[dict[str, Any]] = []
    for config in elements or ["default"]:
        try:
            ds = datasets.load_dataset(repo, name=config, split=split, revision=revision)
        except ValueError as e:
            if split != "test" and "split" in str(e).lower():
                continue  # this element has no few-shot examples
            if elements and ("config" in str(e).lower() or "builderconfig" in str(e).lower()):
                raise ValueError(f"unknown {benchmark} element {config!r} ({repo})") from e
            raise
        for raw in ds:
            row = normalize_row(raw)
            if _keep(row, elements, modules, settings, held):
                rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
# Grouping, "No other option is correct", few-shot selection
# --------------------------------------------------------------------------------------------
NONE_OPTION = "No other option is correct."


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Rows of each (element, base_id), sorted by part. Insertion order follows the rows."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["element"], r["base_id"])].append(r)
    for parts in groups.values():
        parts.sort(key=lambda r: r["part"])
        gradings = {p["grading"] for p in parts}
        if len(gradings) != 1:
            raise ValueError(f"mixed grading within {parts[0]['question_id']}: {gradings}")
    return groups


def _stable_hash(*parts: object) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def none_option_selection(groups: dict[tuple[str, str], list[dict[str, Any]]], seed: int) -> set[tuple[str, str]]:
    """Which answer-graded groups get the correct option replaced by "No other option is correct.".

    Within each (element, number of options of the first part) bucket, groups are ranked by a
    seeded hash of their id and the first round(count / n) are selected, so exactly 1 in n
    questions (up to rounding) has the correct option replaced, as in the original harness
    (which sampled frac = 1/n of the questions). The choice depends only on the seed and the
    element's full question set, not on --limit, shuffling or module/setting filters.
    """
    buckets: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for key, parts in groups.items():
        if parts[0]["grading"] != "answer":
            continue
        buckets[(key[0], len(parts[0]["options"]))].append(key)
    chosen: set[tuple[str, str]] = set()
    for (element, n), keys in buckets.items():
        keys = sorted(keys, key=lambda k: (_stable_hash(seed, "none", k[0], k[1]), k[1]))
        chosen.update(keys[: math.floor(len(keys) / n + 0.5)])
    return chosen


def apply_none_option(parts: list[dict[str, Any]], replace_correct: bool, seed: int) -> list[dict[str, Any]]:
    """Copy of the parts with one option per part replaced by NONE_OPTION.

    If ``replace_correct``, the correct option is replaced (the sentence becomes the correct
    answer, at the same index); otherwise a wrong option chosen with an RNG seeded from
    (seed, question_id).
    """
    out = []
    for p in parts:
        q = dict(p, options=list(p["options"]))
        ci = q["correct_index"]
        if replace_correct:
            j = ci
        else:
            rng = random.Random(_stable_hash(seed, "none-wrong", q["question_id"]))
            j = rng.choice([i for i in range(len(q["options"])) if i != ci])
        q["options"][j] = NONE_OPTION
        q["none_option_index"] = j
        out.append(q)
    return out


def select_few_shot(
    examples: dict[tuple[str, str], list[dict[str, Any]]],
    element: str,
    type_: str | None,
    domain: str | None,
    shots: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """``shots`` few-shot example groups for one (element, type, domain) cell.

    The original harness drew examples matching the cell's metadata, shuffled with the seed,
    and used the same prefix for every question in the cell. Here: candidates with the same
    element, type and domain; if fewer than ``shots``, same element and type; then same element.
    Consistency-graded groups are never used as examples (they have no correct answer).
    """
    if shots <= 0:
        return []
    cands = [g for (el, _), g in examples.items() if el == element and g[0]["grading"] == "answer"]
    tiers = [
        [g for g in cands if g[0]["type"] == type_ and g[0]["domain"] == domain],
        [g for g in cands if g[0]["type"] == type_],
        cands,
    ]
    pool = next((t for t in tiers if len(t) >= shots), cands)
    pool = sorted(pool, key=lambda g: g[0]["question_id"])
    random.Random(_stable_hash(seed, "shots", element, type_, domain)).shuffle(pool)
    return pool[:shots]
