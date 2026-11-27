#!/usr/bin/env bash
# 同步 p5 并跑 M4b（分页 prefill kernel 的对拍 + 分块开销）
set -euo pipefail
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
mkdir -p "$HOME/projects/p5"
cp "$WC"/p5/*.py "$WC"/p5/*.sh "$HOME/projects/p5/" 2>/dev/null || true
cd "$HOME/projects/p5"
echo "--- 文件已同步，开始跑 ---"
mkdir -p logs
~/.venvs/torch/bin/python -u m4b_paged_prefill.py "$@" 2>&1 | tee logs/m4b.log | tail -40
