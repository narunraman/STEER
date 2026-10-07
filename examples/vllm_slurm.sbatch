#!/bin/bash
# Run a Hugging Face model on one GPU with vLLM, in a single Slurm job.
# Inspect starts the vLLM server itself. Adjust the partition, time and paths for your cluster.
#SBATCH --job-name=steer-bench-qwen7b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=steer-bench-%j.out
set -euo pipefail
export HF_HOME=${HF_HOME:-$SCRATCH/hf}          # model weights: a large disk
# export HF_HUB_OFFLINE=1                         # on nodes without internet, after downloading

MODEL=Qwen/Qwen2.5-7B-Instruct
# -M generation_config=vllm: do not apply the model's generation_config.json sampling defaults
# (needed for numbers comparable with the website). answer_max_tokens=1 and top_logprobs=20
# match the website's open-weight runs.
for FORMAT in mc shown hidden; do
  inspect eval steer_bench/steer_me --model vllm/$MODEL -M generation_config=vllm \
    -T format=$FORMAT -T answer_max_tokens=1 -T top_logprobs=20 \
    --max-tokens 4096 --max-connections 256 --log-dir logs/$MODEL
done
steer-bench report logs/$MODEL --out cards/qwen7b-steer-me.html --out cards/qwen7b-steer-me.md
