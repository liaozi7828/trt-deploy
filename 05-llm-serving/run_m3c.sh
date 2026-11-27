#!/usr/bin/env bash
# 跑 M3c：逐位对拍 + 计时。用法：bash run_m3c.sh [rows] [steps]
set -euo pipefail
cd "$HOME/projects/p5"
ROWS="${1:-8}"
STEPS="${2:-16}"
mkdir -p logs
~/.venvs/torch/bin/python -u m3c_qwen_paged.py --rows "$ROWS" --steps "$STEPS" 2>&1 | tee "logs/m3c_${ROWS}x${STEPS}.log"
