#!/bin/bash
# Serve the model as its own Slurm job behind an OpenAI-compatible endpoint, then point Inspect at it.
# https://github.com/narunraman/slurm-llm is one way to do the serving part; any vLLM server works:
#   vllm serve Qwen/Qwen2.5-7B-Instruct --generation-config vllm --api-key "$VLLM_API_KEY" --port 8000
# Keep --generation-config vllm (see the README).
set -euo pipefail
BASE_URL=${BASE_URL:?set BASE_URL to the server, e.g. http://gpu-node:8000/v1}
export VLLM_API_KEY=${VLLM_API_KEY:?set VLLM_API_KEY to the key the server was started with}

inspect eval steer_bench/steer --model vllm/Qwen/Qwen2.5-7B-Instruct --model-base-url "$BASE_URL" \
  -T format=hidden -T answer_max_tokens=1 --max-tokens 4096 --max-connections 128

# any other OpenAI-compatible endpoint:
#   OPENAI_BASE_URL=$BASE_URL OPENAI_API_KEY=... inspect eval steer_bench/steer --model openai-api/<name>/<model> ...
