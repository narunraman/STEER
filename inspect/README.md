# steer-bench: STEER and STEER-ME in Inspect

[Inspect](https://inspect.aisi.org.uk) tasks for two benchmarks of economic rationality in LLMs:

| task | benchmark | paper |
|---|---|---|
| `steer_bench/steer` | STEER | [arXiv 2402.09552](https://arxiv.org/abs/2402.09552) (ICML 2024) |
| `steer_bench/steer_me` | STEER-ME | [arXiv 2502.13119](https://arxiv.org/abs/2502.13119) |

> **Status: not yet published.** The package is not on PyPI and the Hugging Face datasets are not
> released yet, so for now the data has to come from a local copy of the staged packages
> (`-T data_dir=...`). Task version `0.1.0`; numbers may still change before the release.

## Install

From a checkout of this repository:

```bash
cd inspect
uv sync                       # or: uv pip install -e .
```

or straight from git (once pushed): `uv pip install "git+https://github.com/narunraman/STEER#subdirectory=inspect"`.
Loading from the Hugging Face Hub (after the release) needs the `hf` extra: `uv pip install "steer-bench[hf]"`.

## Data

Until the datasets are published, point the tasks at a staged copy of the Hugging Face packages,
either per run with `-T data_dir=PATH` or once with the environment variable `STEER_BENCH_DATA_DIR`.
`PATH` can be the package itself (the directory with `elements.csv` and `data/`) or a parent
holding `STEER/` and `steer_me/`:

```bash
export STEER_BENCH_DATA_DIR=~/UBC/steer-release/hf
```

Without either, the tasks load from the Hub at the repo ids and revisions in
`steer_bench/data.py` (`HF_REPOS`, `HF_REVISIONS`, marked TODO until the release).

## Running

```bash
# plain multiple choice (default), one element
inspect eval steer_bench/steer_me -T element=consumer_surplus --model openai/gpt-4o-mini

# the formats
inspect eval steer_bench/steer_me -T format=mc     ...   # options shown, answer with one letter
inspect eval steer_bench/steer_me -T format=shown  ...   # reason first with the options shown, then answer
inspect eval steer_bench/steer_me -T format=hidden ...   # reason first without the options, then see them and answer
inspect eval steer_bench/steer_me -T format=none   ...   # one option replaced by "No other option is correct."
inspect eval steer_bench/steer_me -T format=free   ...   # no options; numeric answer in free text

# slices and few-shot
inspect eval steer_bench/steer -T setting=single_agent -T shots=5 ...
inspect eval steer_bench/steer -T module=utility_axioms_deterministic ...
inspect eval steer_bench/steer -T element=independence,transitivity ...

# without installing
inspect eval inspect/src/steer_bench/tasks.py@steer_me -T element=law_of_demand ...
```

Useful Inspect options: `--limit N`, `--sample-id consumer_surplus/3561d1e66ba74c46`,
`--epochs`, `--max-connections`, `--temperature` (the task default is 0).

## Parameters (`-T name=value`)

Both tasks take the same parameters.

| parameter | default | meaning |
|---|---|---|
| `format` | `mc` | `mc`, `shown`, `hidden`, `none` or `free` (below) |
| `element` | all | element name, or a comma-separated list |
| `module` | all | module slug (`comparative_statics_of_demand`) or name (`Comparative Statics of Demand`), or a list |
| `setting` | all | setting slug (`consumption`) or name, or a list |
| `shots` | `0` | few-shot examples prepended to the first question (from the `few_shot` split) |
| `prob_mode` | `condition` | which option distribution the calibration metrics report: `condition` or `mix` |
| `logprobs` | `true` | request `top_logprobs` on the answer turn; set `false` for providers that reject the parameter (e.g. OpenAI reasoning models) |
| `top_logprobs` | `20` | how many alternatives to request (20 is the most that providers return) |
| `explanation_length` | `3` | reasoning instruction for `shown`/`hidden`/`free`: 1 = at most 3 sentences, 2 = at most 5, 3 = unrestricted |
| `sig_figs` | `3` | significant figures for `free` answers |
| `answer_max_tokens` | none | `max_tokens` for the answer turn (none = provider default) |
| `seed` | `42` | seed for the `none` transform and few-shot selection |
| `data_dir` | `$STEER_BENCH_DATA_DIR` | staged data (see Data) |
| `include_held` | `false` | include elements staged but held back from release (six STEER social-choice elements, e.g. `sc_axioms`); naming one in `element` also includes it |

## Formats

The prompts are the papers' prompts, copied word for word from the original harness
(`STEER-evaluation/src/steer_evaluation/steer_benchmark.py`). See also the website page on
[question formats](https://steer-benchmark.cs.ubc.ca/guide/formats).

- **`mc`, multiple choice.** One user turn: `Q: <question>`, the options as `A. ...` lines, then
  `Answer by writing the option letter corresponding to the correct option. WRITE ONLY A SINGLE LETTER.`
  and the cue `A: `.
- **`shown`, reasoning first, options shown.** Turn 1: question, options and
  `Let's think step by step. Explain your reasoning.`; the model reasons. Turn 2: the answer instruction.
- **`hidden`, reasoning first, options hidden.** Turn 1: question and the reasoning instruction.
  Turn 2: the options and the answer instruction.
- **`none`, "No other option is correct".** As `mc`, with one option replaced by
  `No other option is correct.`. In 1 of every n questions (n = number of options) it replaces the
  correct option, which then becomes the correct answer; otherwise it replaces a wrong option chosen
  at random. The choice is made at load time from `seed` and the question ids: within each element
  (and number of options) the questions are ranked by a seeded hash and the first round(count/n)
  get the correct option replaced, so exactly 1 in n are, and the result does not depend on
  `--limit`, shuffling or module/setting filters. Consistency-graded elements are left out.
- **`free`, free text.** No options: the question, the reasoning instruction and
  `End your response with 'ANSWER: ' followed by your final answer as a number.` (this last
  sentence is new; the original harness had no free-text prompt). The answer is the first number
  after the last `ANSWER:`, else the last number in the response; it is correct if it equals the
  correct option's value when both are rounded to `sig_figs` significant figures. Only
  answer-graded questions whose correct option is a number are included.

**Multi-part questions and consistency groups.** Every part of a question (all rows with the same
`element` and `base_id`) is packed into one Inspect sample and asked in sequence in one
conversation, as the original harness did; the model answers each part before seeing the next,
and option letters continue across parts (A-B, then C-E). An answer-graded question is correct
if every part is; a consistency-graded group is correct ("consistent") if the tuple of chosen
options is one of those its `consistency_rule` allows.

**Few-shot.** `shots=k` prepends k solved examples from the `few_shot` split, in the plain
multiple-choice format with the correct letter after `A: `, followed by a blank line. Examples
are drawn from the same element, type and domain when there are enough, else the same element
and type, else the same element; the same prefix is used for every question of that cell. The
`none` transform never touches the examples. Elements without examples get none (the number
used is recorded per sample as `shots_used`). Not available for `free`.

## Metrics

Definitions follow the scoring pipeline behind the results website (`steer-scoring`) and the
website page on [metrics](https://steer-benchmark.cs.ubc.ca/guide/metrics). Each metric is
reported per element (`<element>/<metric>`) and over all samples (`all/<metric>`).

| metric | definition |
|---|---|
| `exact_match` | fraction of answered questions answered correctly (consistency groups: rule satisfied) |
| `normalized_accuracy` | mean of +1 for a correct answer and −1/(n−1) for a wrong one (n options; for a multi-part question the largest penalty of its parts), so random guessing scores 0. Consistency groups: −p/(1−p), where p is the chance that random choices satisfy the rule. Free text: wrong costs 0 |
| `no_answer_rate` | fraction of questions with no usable answer; these are left out of the other metrics (as in `steer-scoring`) |
| `ece` | expected calibration error over 10 equal-width confidence bins [i/10, (i+1)/10) (1.0 in the top bin): Σ_bins \|Σ correct − Σ confidence\| / n, where confidence is the top option's probability and correct means the top option is the correct one |
| `brier` | mean over questions of Σ_options (1[option is correct] − p(option))² |
| `epa` | expected probability assignment: mean probability on the correct option |

**Option probabilities.** On the answer turn the task asks for `top_logprobs`. At the answer
position (the first generated token that is a valid option letter) tokens are stripped of
non-alphanumeric characters, kept if they are one of the question's letters, and the largest
probability is taken over duplicates (`"C"` and `" C"`). α is the total probability on valid
letters. *Conditioning* (default) renormalizes: p/α (uniform if α = 0). *Mixing* blends with the
uniform distribution in proportion to the missing mass: α·p_cond + (1 − α)/n. Both are stored for
every sample; `prob_mode` picks the one the metrics report. For multi-part questions only the
first part's probabilities are used. Consistency groups and `free` have no calibration metrics.

**The chosen answer** is the option letter the model wrote (a leading letter such as `B`, `(B)`,
`B.`, else the last `answer is B` / `ANSWER: B`). At temperature 0 this is the most probable
first token, which is what `steer-scoring` uses.

## Providers without logprobs

Calibration needs token probabilities. Anthropic, Bedrock and Mistral return none, and OpenAI
reasoning models reject the request (run them with `-T logprobs=false`). For such runs the task
reports the accuracy metrics only: `ece`, `brier` and `epa` are NaN (never estimated from the
text), and with `logprobs=false` they are not reported at all. Providers return at most 20
alternatives, so probability on letters outside the top 20 counts as missing mass (it lowers α).

## Converting logs for the results website

`steer-bench-export` (or `python -m steer_bench.export`) reads `.eval` logs and writes the
scored-cells table used by `steer-scoring` and the website: one row per run × element × question
format × "No other option" flag × prompt settings × domain × type × perspective × difficulty,
with additive counts and sums, including the ECE bin sums.

```bash
steer-bench-export logs/ --out cells_out/
# cells_out/cells.parquet, models.parquet, provenance.json
```

The columns are those of `steer-scoring` (`run_id = inspect/<model>`; `question_format` is
`shown_options` for `mc`, `none` and `shown`, `hidden_options` for `hidden`, `no_options` for
`free`; `car` is true for `none`; `adaptation` records shots and the reasoning length), plus
`n_prob`, the number of answered rows that had option probabilities. Divide calibration sums by
`n_prob` rather than `n` for Inspect runs; `n_prob = 0` means the provider gave no logprobs.

## Tests

```bash
uv run pytest                     # all tests, mockllm only; no model API is called
uv run pytest -m "not slow"       # skip the test that installs the package and runs the CLI
uv run --with ../../steer-scoring pytest tests/test_parity_steer_scoring.py tests/test_export.py
                                  # per-question parity with steer-scoring, if you have it
```

## Citation

```bibtex
@inproceedings{ramansteer,
  title={STEER: Assessing the Economic Rationality of Large Language Models},
  author={Raman, Narun Krishnamurthi and Lundy, Taylor and Amouyal, Samuel Joseph and Levine, Yoav and Leyton-Brown, Kevin and Tennenholtz, Moshe},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
