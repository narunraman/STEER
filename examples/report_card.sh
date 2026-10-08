#!/bin/bash
# Turn one model's logs into a report card (HTML and Markdown), and into the scored-cells table.
set -euo pipefail
LOGS=${1:-logs}
steer-bench report "$LOGS" --out report_card.html --out report_card.md
# several benchmarks or models in one directory: pick one
#   steer-bench report "$LOGS" --benchmark steer --run inspect/vllm/Qwen/Qwen2.5-7B-Instruct --out card.html
steer-bench export "$LOGS" --out cells/
steer-bench report cells/ --out report_card_from_cells.html     # identical numbers
