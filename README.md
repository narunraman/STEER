<h1><img src="steer_logo.svg" alt="" height="40" align="absmiddle"> STEER</h1>

This repository contains two benchmarks of economic reasoning in language models, together with the code to run a model on them, score it, and diagnose where it fails.

## The benchmarks

We use **STEER** both for the project and for our first benchmark, which tests rationality in economic tasks that require strategic reasoning <a href="https://arxiv.org/abs/2402.09552"><img src="docs/arxiv_logo.svg" height="14" alt=""> ICML 2024</a>. **STEER-ME**, the second benchmark in the STEER family, focuses on non-strategic microeconomic reasoning <a href="https://arxiv.org/abs/2502.13119"><img src="docs/arxiv_logo.svg" height="14" alt=""> NeurIPS 2025</a>.

Both are built on a taxonomy of economic *elements*, where each element is one concept in economic reasoning. The papers describe the taxonomy, the elements and how questions are created; the <a href="https://steer-benchmark.cs.ubc.ca/"><img src="steer_logo.svg" height="14" alt=""> website</a> lets you browse every element interactively.

> [!NOTE]
> Our benchmarks are *dynamic*: questions are generated from templates, so you can draw fresh questions at any time from the [API](https://steer-benchmark.cs.ubc.ca/reference). For a static test set, each benchmark also has a fixed dataset on Hugging Face (CC BY 4.0):
> <a href="https://huggingface.co/datasets/narunraman/steer"><img src="docs/hf_logo.svg" height="14" alt=""> <code>narunraman/steer</code></a> and <a href="https://huggingface.co/datasets/narunraman/steer_me"><img src="docs/hf_logo.svg" height="14" alt=""> <code>narunraman/steer_me</code></a>.

## Evaluating a model

This repository is `steer-bench`, a Python package that runs a language model on STEER or STEER-ME questions and scores it. It is built on [Inspect](https://inspect.aisi.org.uk), so any model Inspect supports works: API models, or open-weight models you run yourself.

### Install

```bash
uv tool install --with-executables-from inspect-ai git+https://github.com/narunraman/STEER
```

(`pip install git+https://github.com/narunraman/STEER` works too.) The questions are downloaded from Hugging Face on first use.

### Run a model

An API model (set the provider's key first, e.g. `OPENAI_API_KEY`):

```bash
inspect eval steer_bench/steer_me --model openai/gpt-4o-mini -T element=consumer_surplus --limit 50
```

An open-weight model with [vLLM](https://docs.vllm.ai), which Inspect starts for you:

```bash
inspect eval steer_bench/steer --model vllm/Qwen/Qwen2.5-7B-Instruct -M generation_config=vllm
```

The tasks are `steer_bench/steer` and `steer_bench/steer_me`; leave out `-T element=...` to run the whole benchmark. `generation_config=vllm` stops vLLM from applying the model's own sampling defaults. To use a model you already serve behind an OpenAI-compatible endpoint, add `--model-base-url`; [slurm-llm](https://github.com/narunraman/slurm-llm) is one way to serve models on a Slurm cluster. More in [`examples/`](examples/).

### Read the results

```bash
steer-bench report logs/ --out card.html
```

A report card shows the model's scores for every format it was run in: a summary, then every element grouped by setting and module, with averages. `steer-bench export logs/ --out results/` writes the same results as a table for your own analysis.

## Choosing what to run

Each question can be posed in several formats, and answers are scored for accuracy and, when the provider returns option probabilities, for calibration. The website explains both: [question formats](https://steer-benchmark.cs.ubc.ca/formats) and [metrics](https://steer-benchmark.cs.ubc.ca/metrics). Pass options to a task with `-T`:

| option | default | meaning |
|---|---|---|
| `format` | `mc` | `mc` (answer directly), `shown` (reason with the options visible), `hidden` (reason, then see the options), `none` (one option is "No other option is correct"), `free` (no options) |
| `element`, `module`, `setting` | all | a name or comma-separated list |
| `shots` | `0` | solved examples (from the `few_shot` split) before the first question |
| `prompts` | `open2025` | the instruction wording: a built-in set or your own JSON file (below) |
| `logprobs`, `top_logprobs` | `true`, `20` | request option probabilities (needed for calibration; vLLM, SGLang, Hugging Face, llama.cpp, Together and OpenAI non-reasoning models return them) |
| `prob_mode` | `condition` | `condition` (renormalize over the option letters) or `mix` |
| `answer_max_tokens` | none | token limit of the answer turn |
| `sig_figs`, `seed` | `3`, `42` | free-text precision; seed for `none` and few-shot selection |
| `data_dir` | Hugging Face | read a local copy of the data instead (also `STEER_BENCH_DATA_DIR`) |

### Prompts

The instructions live in one dict, `PROMPT_SETS` in [`src/steer_bench/prompts.py`](src/steer_bench/prompts.py). To change the wording, write a JSON file with any of its keys and pass `-T prompts=my_prompts.json`:

```json
{"reasoning": "\nThink it through step by step.", "answer": "\nReply with the letter of the correct option only.\nAnswer:"}
```

The prompt set's name and a hash are stored with every result, so runs with different prompts are never mixed.

### Multi-part questions

Some questions have several parts. They are asked in order in one conversation, and the option letters continue from one part to the next (a part that would run past Z starts again at A). For example, a two-part `independence` question in the `mc` format:

```yaml
- role: user
  content: |
    Q: There are two types of toasters available in your local home appliance store.
    Which one would you pick?
    A. Brand A: price is $448, with quality rating 41
    B. No preference between Brand A and Brand B
    C. Brand B: price is $615, with quality rating 71

    Answer by writing the option letter corresponding to the correct option.
    RESPOND WITH ONLY A SINGLE LETTER.
    A:
- role: assistant
  content: C
- role: user
  content: |
    Q: There are three types of toasters available in your local home appliance store.
    Which one would you pick?
    D. No preference between Brand A and Brand B
    E. Brand A: price is $448, with quality rating 41
    F. Brand B: price is $615, with quality rating 71
    G. Brand C: price is $553, with quality rating 31

    Answer by writing the option letter corresponding to the correct option.
    RESPOND WITH ONLY A SINGLE LETTER.
    A:
- role: assistant
  content: F
```

Some elements, like this one, have no single correct option: they are scored on whether the answers across parts are consistent. Here, preferring Brand B in both parts is consistent; adding Brand C should not change the choice between A and B.

## Citation

If you use STEER or STEER-ME, please cite the paper for the benchmark you used:

```bibtex
# STEER
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
```

```bibtex
# STEER-ME
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
