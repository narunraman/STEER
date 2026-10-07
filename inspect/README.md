# steer-bench: STEER and STEER-ME in Inspect

[Inspect](https://inspect.aisi.org.uk) tasks for two benchmarks of economic rationality in LLMs:

| task | benchmark | paper |
|---|---|---|
| `steer_bench/steer` | STEER | [arXiv 2402.09552](https://arxiv.org/abs/2402.09552) (ICML 2024) |
| `steer_bench/steer_me` | STEER-ME | [arXiv 2502.13119](https://arxiv.org/abs/2502.13119) |

Results from these tasks can be converted to the table behind the
[Evals page](https://steer-benchmark.cs.ubc.ca/evals) of the STEER website and turned into a
per-element report card.

> **Status: pre-release.** The Hugging Face datasets are not published yet, so the data has to
> come from a local copy (`-T data_dir=...`). Task version `0.1.0`.

## Install

```bash
uv pip install "git+https://github.com/narunraman/STEER#subdirectory=inspect"
# or: pip install "git+https://github.com/narunraman/STEER#subdirectory=inspect"
```

This installs `inspect-ai` and registers the tasks, so `inspect eval steer_bench/steer_me` works.
For local vLLM add `vllm` to the same environment. A PyPI release will follow. For development:
`cd inspect && uv sync`.

## Data

Until the datasets are published, point the tasks at a local copy, per run with
`-T data_dir=PATH` or once with `export STEER_BENCH_DATA_DIR=PATH`. `PATH` is a benchmark
directory (holding `elements.csv` and `data/<setting>/<module>/<element>/test.parquet`) or a
parent holding `STEER/` and `steer_me/`. Without either, the tasks load from the Hugging Face Hub
(`HF_REPOS`/`HF_REVISIONS` in `steer_bench/data.py`, pinned at release).

## Running with an API model

```bash
export OPENAI_API_KEY=...            # the usual Inspect provider variables
inspect eval steer_bench/steer_me --model openai/gpt-4o-mini -T element=consumer_surplus
inspect eval steer_bench/steer --model openai/o3-mini -T format=shown -T logprobs=false
```

Any [Inspect provider](https://inspect.aisi.org.uk/providers.html) works. Useful Inspect options:
`--limit N`, `--sample-id consumer_surplus/3561d1e66ba74c46`, `--max-connections`, `--epochs`.
Logs go to `./logs` (`--log-dir`); view them with `inspect view`.

## Running Hugging Face models with vLLM on a GPU cluster

Open-weight models run best through [vLLM](https://docs.vllm.ai), which returns the logprobs the
calibration metrics need. There are two ways.

**Let Inspect start vLLM** (one GPU job does everything):

```bash
inspect eval steer_bench/steer_me --model vllm/Qwen/Qwen2.5-7B-Instruct \
  -M generation_config=vllm \
  -T format=shown -T answer_max_tokens=1 --max-tokens 4096 --max-connections 256
```

**Use a vLLM server you started** (or any OpenAI-compatible endpoint):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --generation-config vllm --api-key local --port 8000 &
export VLLM_API_KEY=local
inspect eval steer_bench/steer_me --model vllm/Qwen/Qwen2.5-7B-Instruct \
  --model-base-url http://127.0.0.1:8000/v1 \
  -T format=shown -T answer_max_tokens=1 --max-tokens 4096 --max-connections 256
```

On Slurm, put either in an `sbatch` script that requests one GPU (a 7B model fits on a 24 GB
card) and run Inspect in the same job, or serve the model in its own job and point
`--model-base-url` at it. [slurm-llm](https://github.com/narunraman/slurm-llm) is one way to
serve models on Slurm behind an OpenAI-compatible endpoint. Set `HF_HOME` to a large disk and,
on compute nodes without internet, `HF_HUB_OFFLINE=1` after downloading the weights.

**`--generation-config vllm` is required for numbers comparable with the website.** Since vLLM
0.8, `vllm serve` applies the sampling defaults in the model's `generation_config.json` (for
Qwen: `repetition_penalty=1.05`, `top_p`, `top_k`) to every request field the request leaves
unset. The website's open-weight results were produced without them. Pass
`--generation-config vllm` to `vllm serve`, or `-M generation_config=vllm` when Inspect starts
the server.

**Settings of the website's open-weight runs:** the default prompts, temperature 0 (the task
default), `--max-tokens 4096` for the reasoning turn, `-T answer_max_tokens=1 -T top_logprobs=20`
for the answer turn, and the model's default chat template. A parity check with
Qwen2.5-7B-Instruct on four STEER-ME elements reproduced the website's numbers within decoding
noise (exact match 0.765 vs 0.770 with options shown, 0.447 vs 0.440 with options hidden).

## Logprobs

The calibration metrics (ECE, Brier, EPA) need the probability the model puts on each option
letter. The task requests `top_logprobs` (default 20, the most providers return) on the answer
turn only.

- vLLM, SGLang, Hugging Face, llama.cpp, Together, Grok and OpenAI non-reasoning models return
  them. Google depends on the model.
- Anthropic, Bedrock and Mistral return none: the run reports exact match, normalized accuracy
  and no-answer rate, and the calibration metrics are NaN (never estimated from the text).
- OpenAI reasoning models reject the parameter: run them with `-T logprobs=false` (calibration
  metrics are then not reported at all).
- Letters outside the top 20 count as missing probability mass; the probabilities are
  renormalized over the valid letters (`prob_mode=condition`) or mixed with a uniform
  distribution in proportion to the missing mass (`prob_mode=mix`).

## Task arguments (`-T name=value`)

Both tasks take the same arguments.

| argument | default | meaning |
|---|---|---|
| `format` | `mc` | `mc`, `shown`, `hidden`, `none` or `free` (below) |
| `element` | all released | element name, or a comma-separated list |
| `module` | all | module slug (`comparative_statics_of_demand`) or name, or a list |
| `setting` | all | setting slug or name, or a list |
| `prompts` | `open2025` | prompt set: a built-in name or a JSON file (below) |
| `shots` | `0` | few-shot examples prepended to the first question (from the `few_shot` split) |
| `prob_mode` | `condition` | which option distribution the calibration metrics report: `condition` or `mix` |
| `logprobs` | `true` | request `top_logprobs` on the answer turn; `false` for providers that reject it |
| `top_logprobs` | `20` | how many alternatives to request |
| `answer_max_tokens` | none | `max_tokens` for the answer turn (`1` as in the website's open-weight runs) |
| `sig_figs` | `3` | significant figures for `free` answers |
| `seed` | `42` | seed for the `none` transform and few-shot selection |
| `data_dir` | `$STEER_BENCH_DATA_DIR` | local data (see Data) |
| `include_held` | `false` | include elements held back from release (six STEER social-choice elements, e.g. `sc_axioms`); naming one in `element` also includes it |

## Formats

- **`mc`, multiple choice.** One turn: `Q: <question>`, the options as `A. ...` lines, then the
  answer instruction. The model replies with a letter.
- **`shown`, reasoning first, options shown.** Turn 1: question, options and the reasoning
  instruction; the model reasons. Turn 2: the answer instruction.
- **`hidden`, reasoning first, options hidden.** Turn 1: question and the reasoning instruction.
  Turn 2: the options and the answer instruction.
- **`none`, "No other option is correct".** As `mc`, with one option replaced by
  `No other option is correct.` In 1 of every n questions (n options) it replaces the correct
  option, which then becomes the correct answer; otherwise a wrong option chosen at random. The
  choice depends only on `seed` and the question ids (not on `--limit`, shuffling or filters).
  Consistency-graded elements are left out.
- **`free`, free text.** No options: the question and the reasoning instruction. The answer is
  the number in the last `\boxed{}` (else after the last `ANSWER:`, else the last number); it is
  correct if it equals the correct option's value at `sig_figs` significant figures. Only
  answer-graded questions with a numeric answer are included.

**Multi-part questions and consistency groups.** All parts of a question (rows with the same
`element` and `base_id`) are one Inspect sample, asked in sequence in one conversation; option
letters continue across parts (A-B, then C-E). An answer-graded question is correct if every part
is; a consistency-graded group is correct if the chosen options satisfy its `consistency_rule`.

**Few-shot.** `shots=k` prepends k solved examples in the `mc` format, each ending with the answer
instruction and the correct letter, then a blank line. Examples come from the same element, type
and domain when there are enough, else the same element and type, else the same element.

## Prompts

Every instruction the model sees is in one dict, `PROMPT_SETS` in
[`src/steer_bench/prompts.py`](src/steer_bench/prompts.py). The default set, `open2025`, is the
one used by every open-weight run on the website:

| key | text |
|---|---|
| `reasoning` | `\nPlease reason step by step, and put your final answer within \boxed{}.` |
| `answer` | `\nAnswer by writing the option letter corresponding to the correct option. RESPOND WITH ONLY A SINGLE LETTER.\nA:` |
| `free_answer` | empty: extra text after the reasoning instruction in `free` |
| `hidden_options_newline` | `false`: in `hidden`, the options turn starts directly with `A. ...` |
| `few_shot_answer_sep` | a space between the answer instruction and the letter in few-shot examples |

To try other wording, write a JSON file with any of these keys (the rest stay as in `open2025`)
and pass it as `-T prompts=my_prompts.json`:

```json
{"reasoning": "\nThink it through step by step.", "answer": "\nReply with the letter of the correct option only.\nAnswer:"}
```

The set's name (the file name without `.json`) and a hash of its strings are stored in the log
(`eval.metadata.prompts`, `prompts_hash`, `prompt_strings`) and in the converted results, so
results with different prompts are never mixed. Only the default prompts give numbers comparable
with the website.

## Metrics

Reported per element (`<element>/<metric>`) and over all samples (`all/<metric>`). Definitions
follow the scoring pipeline behind the website; see also the website's
[scoring page](https://steer-benchmark.cs.ubc.ca/scoring).

| metric | definition |
|---|---|
| `exact_match` | fraction of answered questions answered correctly (consistency groups: rule satisfied) |
| `normalized_accuracy` | mean of +1 for a correct answer and −1/(n−1) for a wrong one (n options; multi-part: the largest penalty of its parts), so random guessing scores 0. Consistency groups: −p/(1−p), p the chance that random choices satisfy the rule. Free text: wrong costs 0 |
| `no_answer_rate` | fraction of questions with no usable answer; these are left out of the other metrics |
| `ece` | expected calibration error over 10 equal-width confidence bins (confidence = the top option's probability) |
| `brier` | mean over questions of Σ_options (1[option is correct] − p(option))² |
| `epa` | expected probability assignment: mean probability on the correct option |

**Option probabilities** come from the `top_logprobs` at the answer position (the first generated
token that is a valid letter): tokens are stripped of non-alphanumerics, kept if they are one of
the question's letters, and the largest probability is taken over duplicates (`"C"`, `" C"`).
Multi-part questions use the first part's probabilities. Consistency groups and `free` have no
calibration metrics.

**The chosen answer** is the letter the model wrote (a leading `B`, `(B)`, `B.`, else the last
`answer is B` / `ANSWER: B`).

## Report cards

```bash
steer-bench report logs/ --out card.html --out card.md
steer-bench report cells_out/ --out card.html          # or from converted results
```

A report card covers one model on one benchmark (`--benchmark`, `--run` pick one when the input
has several). For every format that was run it shows exact match, normalized accuracy, no-answer
rate and, where logprobs exist, ECE, Brier and EPA; a heatmap of normalized accuracy for every
element, grouped by setting and module in the benchmark's order with module and setting averages
(unweighted means over elements); and a table of all metrics per element, with the element's
worst domain. The HTML file is self-contained (inline CSS, no external assets). The taxonomy
(names and order) is bundled as `src/steer_bench/taxonomy.json`, a copy derived from the website
content by `tools/build_taxonomy.py`.

## Results and the website's Evals page

The [Evals page](https://steer-benchmark.cs.ubc.ca/evals) is built from a *scored-cells table*:
one row per model × element × question format × prompt settings × domain × type × perspective ×
difficulty, with additive counts and sums. `steer-bench export` writes the same table from Inspect
logs:

```bash
steer-bench export logs/ --out cells_out/     # cells.parquet, models.parquet, provenance.json
```

The columns are those of the website's pipeline (`run_id = inspect/<model>`; `question_format` is
`shown_options` for `mc`, `none` and `shown`, `hidden_options` for `hidden`, `no_options` for
`free`; `car` is true for `none`; `adaptation` records shots, reasoning and
`prompts=<name>@<hash>`), plus `n_prob`, the number of answered questions that had option
probabilities. Divide calibration sums by `n_prob`; `n_prob = 0` means no logprobs. Metrics
computed from the table equal the ones Inspect reports, and the report card is computed from it.
The page's variants map to the formats as "No CoT" = `mc`, "Options shown" = `shown`,
"Options hidden" = `hidden` and "NOTA" = `none`.

## Tests

```bash
uv run pytest                     # mockllm only; no model is called
uv run pytest -m "not slow"       # skip the test that installs the package and runs the CLI
```

## Citation

```bibtex
@inproceedings{ramansteer,
  title={STEER: Assessing the Economic Rationality of Large Language Models},
  author={Raman, Narun Krishnamurthi and Lundy, Taylor and Amouyal, Samuel Joseph and Levine, Yoav and Leyton-Brown, Kevin and Tennenholtz, Moshe},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
