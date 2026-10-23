<h1>
  <img src=".github/assets/steer_lockup_light.svg#gh-light-mode-only" alt="STEER" height="44">
  <img src=".github/assets/steer_lockup_dark.svg#gh-dark-mode-only" alt="STEER" height="44">
</h1>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23406991.svg)](https://doi.org/10.5281/zenodo.23406991)

This repository contains two benchmarks of economic reasoning in language models, together with the code to run a model on them, score it, and diagnose where it fails.

## The benchmarks

We use **STEER** both for the project and for our first benchmark, which tests rationality in economic tasks that require strategic reasoning <a href="https://arxiv.org/abs/2402.09552"><img src=".github/assets/arxiv_logo.svg" height="14" alt=""></a> <a href="https://arxiv.org/abs/2402.09552">ICML 2024</a>. **STEER-ME**, the second benchmark in the STEER family, focuses on non-strategic microeconomic reasoning <a href="https://arxiv.org/abs/2502.13119"><img src=".github/assets/arxiv_logo.svg" height="14" alt=""></a> <a href="https://arxiv.org/abs/2502.13119">NeurIPS 2025</a>.

Both are built on a taxonomy of economic *elements*, where each element is one concept in economic reasoning. The papers describe the taxonomy, the elements and how questions are created; the <a href="https://steer-benchmark.cs.ubc.ca/"><img src=".github/assets/steer_badge.svg" height="14" alt=""></a> <a href="https://steer-benchmark.cs.ubc.ca/">website</a> lets you browse every element interactively, and its <a href="https://steer-benchmark.cs.ubc.ca/evals">Evals</a> pages show how many models score on each.

> [!NOTE]
> Our benchmarks are *dynamic*: questions are generated from templates by <a href="https://github.com/narunraman/autosteer"><img src=".github/assets/autosteer_wheel.svg" height="14" alt=""></a> <a href="https://github.com/narunraman/autosteer">autoSTEER</a>, so you can draw fresh questions at any time from the [API](https://steer-benchmark.cs.ubc.ca/reference) or with autoSTEER in Python. For a static test set, each benchmark also has a fixed dataset on Hugging Face (CC BY 4.0):
> <a href="https://huggingface.co/datasets/narunraman/steer"><img src=".github/assets/hf_logo.svg" height="14" alt=""></a> <a href="https://huggingface.co/datasets/narunraman/steer"><code>narunraman/steer</code></a> and <a href="https://huggingface.co/datasets/narunraman/steer_me"><img src=".github/assets/hf_logo.svg" height="14" alt=""></a> <a href="https://huggingface.co/datasets/narunraman/steer_me"><code>narunraman/steer_me</code></a>.

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


> [!TIP]
> A score is an average over the fixed dataset, and the report card can only show weaknesses the dataset has enough questions to reveal. Because the benchmarks are dynamic, you can instead search the space of questions for the combinations of element, domain and difficulty where a model is reliably wrong: [QuickScope](https://quickscope.narunraman.com/) does this adaptively and runs on STEER-ME out of the box.

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

To cite steer-bench itself (this DOI always points to the latest release; each release on [Zenodo](https://doi.org/10.5281/zenodo.23406991) has its own):

```bibtex
@software{raman_steer_bench,
  author    = {Raman, Narun},
  title     = {{steer-bench: STEER and STEER-ME evaluations}},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23406991},
  url       = {https://doi.org/10.5281/zenodo.23406991}
}
```
