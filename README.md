<img src="steer_small.png" alt="STEER" width="120" align="right">

# STEER and STEER-ME

**STEER** (ICML 2024) tests whether large language models make economically rational choices. It
breaks rationality into fine-grained elements, from arithmetic and probability through
single-agent decisions, games and choices made on behalf of others, each tested with
multiple-choice questions generated across domains, question types and perspectives.

**STEER-ME** (NeurIPS 2025, Datasets and Benchmarks Track) extends this to microeconomic
reasoning: deriving demand, comparative statics, production decisions, market equilibrium and
welfare, with questions that need several steps of calculation.

This repository holds `steer-bench`, the [Inspect](https://inspect.aisi.org.uk) evaluation of
both benchmarks: it runs a model on the questions, scores accuracy and calibration, converts
results to the table behind the website's Evals page and makes per-element report cards.

- Website: <https://steer-benchmark.cs.ubc.ca> (benchmarks, elements, results)
- Papers: STEER, [arXiv 2402.09552](https://arxiv.org/abs/2402.09552);
  STEER-ME, [arXiv 2502.13119](https://arxiv.org/abs/2502.13119)
- Data (CC BY 4.0): [`narunraman/steer`](https://huggingface.co/datasets/narunraman/steer) and
  [`narunraman/steer-me`](https://huggingface.co/datasets/narunraman/steer-me) on Hugging Face
- Looking for the 2024 data? See tag [`v1-icml2024`](https://github.com/narunraman/STEER/tree/v1-icml2024).

## Quickstart

```bash
uv tool install --with-executables-from inspect-ai git+https://github.com/narunraman/STEER
#   add --with vllm for local open-weight models; or, in a virtualenv: pip install git+https://github.com/narunraman/STEER
export OPENAI_API_KEY=...                                               # the usual Inspect provider variables
inspect eval steer_bench/steer_me --model openai/gpt-4o-mini -T element=consumer_surplus --limit 50
steer-bench report logs/ --out card.html --out card.md
```

The tasks are `steer_bench/steer` and `steer_bench/steer_me`. Any
[Inspect provider](https://inspect.aisi.org.uk/providers.html) works; Inspect options such as
`--limit`, `--sample-id`, `--max-connections` and `--epochs` apply. Logs go to `./logs`
(`inspect view` opens them). More in [`examples/`](examples/): an API model, a Hugging Face model
with vLLM in one Slurm job, a model served separately (for example with
[slurm-llm](https://github.com/narunraman/slurm-llm)), and a report card.

Until the Hugging Face datasets are public, point the tasks at a local copy with
`-T data_dir=PATH` or `STEER_BENCH_DATA_DIR=PATH` (a benchmark directory with `elements.csv` and
`data/<setting>/<module>/<element>/test.parquet`, or a parent holding `STEER/` and `steer_me/`).

## Open-weight models with vLLM

```bash
# Inspect starts vLLM (one GPU job)
inspect eval steer_bench/steer_me --model vllm/Qwen/Qwen2.5-7B-Instruct -M generation_config=vllm \
  -T format=shown -T answer_max_tokens=1 --max-tokens 4096 --max-connections 256

# or a server you started (any OpenAI-compatible vLLM endpoint)
vllm serve Qwen/Qwen2.5-7B-Instruct --generation-config vllm --api-key local --port 8000 &
VLLM_API_KEY=local inspect eval steer_bench/steer_me --model vllm/Qwen/Qwen2.5-7B-Instruct \
  --model-base-url http://127.0.0.1:8000/v1 -T format=shown -T answer_max_tokens=1 --max-tokens 4096
```

**Use `--generation-config vllm`** (or `-M generation_config=vllm` when Inspect starts the
server). Otherwise vLLM ≥ 0.8 applies the sampling defaults in the model's
`generation_config.json` (for Qwen, `repetition_penalty=1.05`) and the numbers are not comparable
with the website. The website's open-weight runs used the default prompts, temperature 0 (the
task default), `--max-tokens 4096` for reasoning and `-T answer_max_tokens=1 -T top_logprobs=20`
for the answer; a parity check on Qwen2.5-7B-Instruct reproduced them within decoding noise.

**Logprobs.** Calibration metrics (ECE, Brier, EPA) need the probabilities of the option
letters, which the task requests on the answer turn (top 20). vLLM, SGLang, Hugging Face,
llama.cpp, Together, Grok and OpenAI non-reasoning models return them. Anthropic, Bedrock and
Mistral do not: those runs report accuracy only and the calibration metrics are NaN. OpenAI
reasoning models reject the request: run them with `-T logprobs=false`.

## Formats and task arguments

| `-T format=` | the model sees |
|---|---|
| `mc` (default) | the question and options, then the answer instruction; it replies with a letter |
| `shown` | question, options and a reasoning instruction; then the answer instruction |
| `hidden` | question and the reasoning instruction; then the options and the answer instruction |
| `none` | as `mc`, with one option replaced by "No other option is correct." (the correct one in 1 of every n questions) |
| `free` | no options; the number in the last `\boxed{}` is compared with the answer at `sig_figs` significant figures (numeric elements only) |

Multi-part questions are one sample, asked in sequence in one conversation with option letters
continuing across parts; consistency-graded elements are correct when the chosen options satisfy
the element's rule.

| argument | default | meaning |
|---|---|---|
| `element`, `module`, `setting` | all released | a name or comma-separated list (slug or full name) |
| `prompts` | `open2025` | prompt set: built-in name or a JSON file (below) |
| `shots` | `0` | solved examples prepended to the first question |
| `logprobs` / `top_logprobs` | `true` / `20` | request option probabilities on the answer turn |
| `prob_mode` | `condition` | `condition` (renormalize over the letters) or `mix` (blend in the missing mass uniformly) |
| `answer_max_tokens` | none | `max_tokens` of the answer turn |
| `sig_figs`, `seed` | `3`, `42` | free-text precision; seed for `none` and few-shot selection |
| `data_dir`, `include_held` | env, `false` | local data; include elements held back from release |

**Metrics**, per element and overall: `exact_match`; `normalized_accuracy` (+1 correct,
−1/(n−1) wrong, so random guessing scores 0); `no_answer_rate` (unanswered questions are left
out of the others); and with logprobs `ece` (10 bins), `brier` and `epa` (mean probability on the
correct option). Definitions: <https://steer-benchmark.cs.ubc.ca/scoring>.

## Prompts

All instructions are in one dict, `PROMPT_SETS` in
[`src/steer_bench/prompts.py`](src/steer_bench/prompts.py). The default, `open2025`, is the set
used by the open-weight runs on the website:

| key | text |
|---|---|
| `reasoning` | `\nPlease reason step by step, and put your final answer within \boxed{}.` |
| `answer` | `\nAnswer by writing the option letter corresponding to the correct option. RESPOND WITH ONLY A SINGLE LETTER.\nA:` |
| `free_answer` | empty (extra text after the reasoning instruction in `free`) |
| `hidden_options_newline` | `false` (in `hidden`, the options turn starts directly with `A. ...`) |
| `few_shot_answer_sep` | a space (between the answer instruction and the letter in few-shot examples) |

To use your own wording, write a JSON file with any of these keys (the rest stay as in
`open2025`) and pass `-T prompts=my_prompts.json`:

```json
{"reasoning": "\nThink it through step by step.", "answer": "\nReply with the letter of the correct option only.\nAnswer:"}
```

The set's name and a hash of its strings are stored in the log and in the converted results, so
runs with different prompts are never mixed. Only the default set gives numbers comparable with
the website.

## Results, report cards and the website

```bash
steer-bench report logs/ --out card.html --out card.md   # one model, one benchmark
steer-bench export logs/ --out cells/                    # cells.parquet, models.parquet, provenance.json
```

A **report card** shows, for every format run, a summary (accuracy, normalized accuracy,
no-answer rate, calibration), a heatmap of normalized accuracy for every element grouped by
setting and module in the benchmark's order, with module and setting averages, and all metrics
per element with the element's worst domain. The HTML is a single self-contained file.

`steer-bench export` writes the **scored-cells table** that the
[Evals page](https://steer-benchmark.cs.ubc.ca/evals) is built from: one row per model × element
× format × prompt settings × domain × type × perspective × difficulty, with additive counts (the
page's "No CoT", "Options shown", "Options hidden" and "NOTA" are `mc`, `shown`, `hidden` and
`none`). Report cards can be made from logs or from this table, with identical numbers.

## Development

```bash
uv sync
uv run pytest            # mockllm only; no model is called
```

## Citation

```bibtex
@inproceedings{ramansteer,
  author    = {Narun Krishnamurthi Raman and Taylor Lundy and
               Samuel Joseph Amouyal and Yoav Levine and
               Kevin Leyton{-}Brown and Moshe Tennenholtz},
  title     = {{STEER:} {A}ssessing the {E}conomic {R}ationality of
               {L}arge {L}anguage {M}odels},
  booktitle = {Forty-first International Conference on Machine Learning,
               {ICML} 2024, Vienna, Austria, July 21-27, 2024},
  publisher = {OpenReview.net},
  year      = {2024},
  url       = {https://openreview.net/forum?id=nU1mtFDtMX}
}

@inproceedings{raman2025steerme,
  title     = {{STEER-ME}: Assessing the Microeconomic
               Reasoning of Large Language Models},
  author    = {Raman, Narun and Lundy, Taylor and
               Amin, Thiago and Perla, Jesse and
               Leyton-Brown, Kevin},
  booktitle = {Advances in Neural Information Processing
               Systems},
  volume    = {38},
  year      = {2025},
  note      = {Datasets and Benchmarks Track},
  url       = {https://arxiv.org/abs/2502.13119}
}
```

Code: MIT ([LICENSE](LICENSE)). Data: CC BY 4.0.
