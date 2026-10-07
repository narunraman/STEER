"""Fixtures: a tiny dataset in the staged Hugging Face layout, and scripted mock models.

No test calls a model API: every model is ``mockllm/model`` with scripted outputs.
"""

from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.model._model_output import Logprob, Logprobs, TopLogprob

STAGED_ROOT = Path(os.environ.get("STEER_BENCH_STAGED", "~/UBC/steer-release/hf")).expanduser()

# element -> (setting slug, setting name, module slug, module name)
ELEMENTS = {
    "consumer_surplus": ("equilibria", "Evaluating Equilibria", "welfare", "Welfare Analysis"),
    "words": ("consumption", "Consumption Decisions", "demand", "Comparative Statics of Demand"),
    "tiny": ("consumption", "Consumption Decisions", "demand", "Comparative Statics of Demand"),
    "pairs": ("single_agent", "Single Agent", "axioms", "Utility Axioms"),
    "multi": ("single_agent", "Single Agent", "mech", "Mechanism Properties"),
    "sc_axioms": ("behalf", "Behalf of Others", "social", "Social Choice"),  # held (STEER)
}


def _row(element, base_id, part, text, options, correct, grading="answer", rule=None,
         type_="t1", domain="d1", perspective="anon"):
    return {
        "element": element, "question_id": f"{element}/{base_id}/{part}", "base_id": str(base_id),
        "part": part, "question_text": text, "options": options,
        "correct_index": correct, "grading": grading,
        "consistency_group": f"{element}/{base_id}" if grading == "consistency" else None,
        "consistency_rule": rule, "type": type_, "domain": domain, "perspective": perspective,
        "tags": [], "difficulty": 0, "template_id": None, "source": "test@0", "repair": [],
        "explanation": None,
    }


def make_rows() -> dict[str, dict[str, list[dict]]]:
    rows: dict[str, dict[str, list[dict]]] = {e: {"test": [], "few_shot": []} for e in ELEMENTS}
    # consumer_surplus: 40 numeric 4-option questions; gold value = 10 + i + 0.123
    for i in range(40):
        gold = f"{10 + i + 0.123:.3f}"
        opts = [gold, f"{i + 0.5}", f"{i + 1.5}", f"{i + 2.5}"]
        k = i % 4
        opts[0], opts[k] = opts[k], opts[0]
        rows["consumer_surplus"]["test"].append(_row(
            "consumer_surplus", f"cs{i:02d}", 0, f"cs question {i}", opts, k,
            type_="linear" if i % 2 else "quadratic", domain="food" if i < 20 else "energy"))
    for i in range(6):
        rows["consumer_surplus"]["few_shot"].append(_row(
            "consumer_surplus", f"ex{i}", 0, f"cs example {i}", ["1", "2", "3", "4"], i % 4,
            type_="linear" if i % 2 else "quadratic", domain="food"))
    # words: non-numeric answers
    for i in range(9):
        rows["words"]["test"].append(_row("words", f"w{i}", 0, f"words question {i}",
                                          ["Yes", "No", "Maybe"], i % 3))
    # tiny: the hand-computed metrics example (3 questions, gold A, B, C)
    for i in range(3):
        rows["tiny"]["test"].append(_row("tiny", f"q{i}", 0, f"tiny q{i}", ["x", "y", "z"], i))
    # pairs: consistency groups, 2 then 3 options, rule same:01,12
    for g in range(3):
        rows["pairs"]["test"].append(_row("pairs", f"g{g}", 0, f"pairs g{g} part0",
                                          ["B", "A"], None, "consistency", "same:01,12"))
        rows["pairs"]["test"].append(_row("pairs", f"g{g}", 1, f"pairs g{g} part1",
                                          ["C", "B", "A"], None, "consistency", "same:01,12"))
    # multi: answer-graded two-part questions (gold B then C -> letters B, then D among C-E)
    for g in range(2):
        rows["multi"]["test"].append(_row("multi", f"m{g}", 0, f"multi m{g} part0", ["no", "yes"], 1))
        rows["multi"]["test"].append(_row("multi", f"m{g}", 1, f"multi m{g} part1", ["1", "2", "3"], 1))
    rows["sc_axioms"]["test"].append(_row("sc_axioms", "h0", 0, "held", ["1", "2"], 0))
    return rows


def write_package(root: Path, name: str = "steer_me") -> Path:
    pkg = root / name
    rows = make_rows()
    pkg.mkdir(parents=True, exist_ok=True)
    with open(pkg / "elements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["element", "setting", "setting_slug", "module", "module_slug"])
        for el, (ss, sn, ms, mn) in ELEMENTS.items():
            w.writerow([el, sn, ss, mn, ms])
    for el, (ss, _, ms, _) in ELEMENTS.items():
        d = pkg / "data" / ss / ms / el
        d.mkdir(parents=True, exist_ok=True)
        for split, rs in rows[el].items():
            table = pa.Table.from_pylist(rs, schema=_schema())
            pq.write_table(table, d / f"{split}.parquet")
    return pkg


def _schema() -> pa.Schema:
    return pa.schema([
        ("element", pa.string()), ("question_id", pa.string()), ("base_id", pa.string()),
        ("part", pa.int32()), ("question_text", pa.string()), ("options", pa.list_(pa.string())),
        ("correct_index", pa.int32()), ("grading", pa.string()), ("consistency_group", pa.string()),
        ("consistency_rule", pa.string()), ("type", pa.string()), ("domain", pa.string()),
        ("perspective", pa.string()), ("tags", pa.list_(pa.string())), ("difficulty", pa.int32()),
        ("template_id", pa.string()), ("source", pa.string()), ("repair", pa.list_(pa.string())),
        ("explanation", pa.string()),
    ])


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("hf")
    write_package(root, "steer_me")
    write_package(root, "STEER")
    return root


# ------------------------------------------------------------------ scripted mock models
Policy = Callable[[str, str], "tuple[str, dict[str, float] | None]"]


def output(text: str, top: dict[str, float] | None, with_logprobs: bool) -> ModelOutput:
    out = ModelOutput.from_content(model="mockllm", content=text)
    if with_logprobs and top is not None:
        first = text.split()[0] if text.strip() else text
        lp = math.log(top.get(first, max(top.values())))
        out.choices[0].logprobs = Logprobs(content=[Logprob(
            token=first, logprob=lp,
            top_logprobs=[TopLogprob(token=t, logprob=math.log(p)) for t, p in top.items()])])
    return out


def mock_model(policy: Policy, provider_logprobs: bool = True):
    """mockllm/model whose reply is ``policy(last_user_text, all_user_text)``.

    The returned ``(text, top)`` gives the completion and, for answer turns, the first-token
    top_logprobs as probabilities. ``provider_logprobs=False`` mimics a provider that ignores
    the logprobs request (e.g. Anthropic).
    """
    def fn(messages, tools, tool_choice, config):
        users = [m.text for m in messages if isinstance(m, ChatMessageUser)]
        if not config.logprobs:
            text, _ = policy(users[-1], "\n".join(users))
            return output(text, None, False)
        text, top = policy(users[-1], "\n".join(users))
        return output(text, top, provider_logprobs)
    return get_model("mockllm/model", custom_outputs=fn)


def question_index(text: str, prefix: str) -> int | None:
    m = re.search(prefix + r"\s*(\d+)", text)
    return int(m.group(1)) if m else None
