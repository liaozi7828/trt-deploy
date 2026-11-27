#!/usr/bin/env bash
# 跑 M3d：分页 decode 上 CUDA Graph。用法：bash run_m3d.sh [rows] [steps]
set -euo pipefail
cd "$HOME/projects/p5"
ROWS="${1:-24}"
STEPS="${2:-16}"
mkdir -p logs
~/.venvs/torch/bin/python -u m3d_graph_paged.py --rows "$ROWS" --steps "$STEPS" 2>&1 \
  | tee "logs/m3d_${ROWS}x${STEPS}.log" | tail -40
