#!/usr/bin/env bash
# 同步 p5 到 WSL 并跑 m5a（prefix caching）
set -euo pipefail
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
mkdir -p ~/projects/p5/logs
cp -r "$WC/p5/." ~/projects/p5/
cd ~/projects/p5
~/.venvs/torch/bin/python -u m5a_prefix_cache.py "$@" 2>&1 | tee logs/m5a.log | tail -40
