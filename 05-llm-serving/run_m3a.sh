#!/usr/bin/env bash
# M3a：分页 KV 的正确性 + 显存账
set -e
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
cd ~/projects/p5
cp -f "$WC"/p5/m3a_paged.py .
export HF_ENDPOINT=https://hf-mirror.com
~/.venvs/torch/bin/python -m py_compile m3a_paged.py
~/.venvs/torch/bin/python -u m3a_paged.py --reqs 12 --steps 16 --prompt 64 2>&1 | tail -25
