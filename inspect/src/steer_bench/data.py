"""Loading STEER / STEER-ME rows and packing them into Inspect samples.

Every assumption about the dataset schema lives in this module, in ``normalize_row`` and
``_local_files``. The rest of the package only sees the canonical row dict that
``normalize_row`` returns (keys in ``ROW_KEYS``).

Data sources:

* ``data_dir`` (or the ``STEER_BENCH_DATA_DIR`` environment variable): a staged copy of the
  Hugging Face package, laid out as ``<dir>/elements.csv`` and
  ``<dir>/data/<setting>/<module>/<element>/<split>.parquet``. ``data_dir`` may point at the
  package itself or at a parent that holds ``STEER/`` and ``steer_me/``.
* Otherwise the Hugging Face Hub, at ``HF_REPOS`` / ``HF_REVISIONS`` (not published yet).
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

# TODO(release): the Hugging Face datasets are not published yet. Fill in the repo ids and pin
# each revision to a commit SHA of the published dataset before registering the eval.
HF_REPOS: dict[str, str] = {"steer": "narunraman/steer", "steer_me": "narunraman/steer_me"}
HF_REVISIONS: dict[str, str | None] = {"steer": None, "steer_me": None}  # TODO(release): commit SHAs

DATA_DIR_ENV = "STEER_BENCH_DATA_DIR"

# Staged but held back from release (steer-drafts/README.md). Loaded only when named explicitly.
HELD_ELEMENTS = {"steer": {"enforceability"}, "steer_me": {"tfp_shocks"}}

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

    Schema assumptions (staged packages of 2026-09-28, see steer-drafts/schema_spec.md):
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


def resolve_data_dir(benchmark: str, data_dir: str | None) -> Path | None:
    """The staged package directory for ``benchmark``, or None to use the Hub."""
    d = data_dir or os.environ.get(DATA_DIR_ENV)
    if not d:
        return None
    p = Path(d).expanduser()
    if (p / "elements.csv").exists():
        return p
    for child in p.iterdir() if p.is_dir() else []:
        if child.is_dir() and child.name.lower() == benchmark and (child / "elements.csv").exists():
            return child
    raise FileNotFoundError(
        f"{p} is neither a {benchmark} package (no elements.csv) nor a directory containing one "
        f"(expected a '{benchmark}' subdirectory, case-insensitive)."
    )


def _element_table(pkg: Path) -> dict[str, dict[str, str]]:
    with open(pkg / "elements.csv", newline="") as f:
        return {r["element"]: r for r in csv.DictReader(f)}


def _matches(value: str | None, names: Iterable[str | None], wanted: list[str] | None) -> bool:
    if wanted is None:
        return True
    names = {n.lower() for n in names if n}
    return any(w.lower() in names for w in wanted) or (value or "").lower() in {w.lower() for w in wanted}


def _local_files(pkg: Path, split: str) -> list[tuple[str, str, str, Path]]:
    """(element, setting_slug, module_slug, path) for every ``<split>.parquet`` in the package."""
    out = []
    for path in sorted(pkg.glob(f"data/*/*/*/{split}.parquet")):
        element, module, setting = path.parent.name, path.parent.parent.name, path.parent.parent.parent.name
        out.append((element, setting, module, path))
    return out


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
    setting match either the slug (``comparative_statics_of_demand``) or the full name
    (``Comparative Statics of Demand``), case-insensitively.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}, got {benchmark!r}")
    elements, modules, settings = _as_list(element), _as_list(module), _as_list(setting)
    held = set() if include_held else HELD_ELEMENTS[benchmark] - set(elements or [])

    pkg = resolve_data_dir(benchmark, data_dir)
    if pkg is None:
        return _load_rows_hf(benchmark, split, elements, modules, settings, held)

    import pyarrow.parquet as pq

    table = _element_table(pkg)
    known = {e for e, _, _, _ in _local_files(pkg, "test")} | set(table)
    if elements:
        unknown = [e for e in elements if e not in known]
        if unknown:
            raise ValueError(f"unknown {benchmark} element(s): {unknown}")
    rows: list[dict[str, Any]] = []
    for el, setting_slug, module_slug, path in _local_files(pkg, split):
        meta = table.get(el, {})
        if elements and el not in elements:
            continue
        if el in held:
            continue
        if not _matches(module_slug, [meta.get("module"), meta.get("module_slug")], modules):
            continue
        if not _matches(setting_slug, [meta.get("setting"), meta.get("setting_slug")], settings):
            continue
        for raw in pq.read_table(path).to_pylist():
            rows.append(normalize_row(raw, setting=setting_slug, module=module_slug))
    return rows


def _load_rows_hf(
    benchmark: str,
    split: str,
    elements: list[str] | None,
    modules: list[str] | None,
    settings: list[str] | None,
    held: set[str],
) -> list[dict[str, Any]]:
    """Load from the Hugging Face Hub.

    TODO(release): untested until the datasets are published. Assumes the recommended final
    layout (schema_spec.md section 4): one config per element plus ``default``, and
    ``setting``/``module`` as columns.
    """
    from inspect_ai.dataset import Sample, hf_dataset

    repo, revision = HF_REPOS[benchmark], HF_REVISIONS[benchmark]
    if revision is None:
        raise RuntimeError(
            f"The {benchmark} dataset is not published on Hugging Face yet ({repo}). Point "
            f"-T data_dir=... or ${DATA_DIR_ENV} at a staged copy of the package."
        )
    configs = elements or ["default"]
    rows: list[dict[str, Any]] = []
    for config in configs:
        ds = hf_dataset(
            repo, split=split, name=config, revision=revision,
            sample_fields=lambda rec: Sample(input="-", metadata={"raw": rec}),
        )
        for s in ds:
            row = normalize_row((s.metadata or {})["raw"])
            if row["element"] in held:
                continue
            if not _matches(row["module"], [], modules) or not _matches(row["setting"], [], settings):
                continue
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
