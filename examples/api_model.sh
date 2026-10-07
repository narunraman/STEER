#!/bin/bash
# Run STEER-ME on an API model. Set the provider's key in your environment first
# (e.g. OPENAI_API_KEY); see https://inspect.aisi.org.uk/providers.html.
set -euo pipefail

# multiple choice on one element, a few questions
inspect eval steer_bench/steer_me --model openai/gpt-4o-mini \
  -T element=consumer_surplus --limit 20

# reasoning with the options shown, all STEER elements of one setting
inspect eval steer_bench/steer --model openai/gpt-4o-mini \
  -T format=shown -T setting=foundations

# providers without logprobs (or that reject the parameter): accuracy metrics only
inspect eval steer_bench/steer_me --model openai/o3-mini -T format=shown -T logprobs=false
