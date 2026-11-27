#!/usr/bin/env bash
# 同步 p5 并跑 M4a（chunked prefill 的阻塞 vs 切块）
set -euo pipefail
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
mkdir -p "$HOME/projects/p5"
cp "$WC"/p5/*.py "$WC"/p5/*.sh "$HOME/projects/p5/" 2>/dev/null || true
cd "$HOME/projects/p5"
echo "--- 文件已同步，开始跑 ---"
~/.venvs/torch/bin/python -u m4a_mixed.py "$@" 2>&1 | tee logs/m4a.log | tail -35
