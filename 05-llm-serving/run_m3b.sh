#!/usr/bin/env bash
# M3b：分页 attention kernel 的正确性与速度
set -e
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
cd ~/projects/p5
cp -f "$WC"/p5/m3b_paged_attn.py .
~/.venvs/torch/bin/python -m py_compile m3b_paged_attn.py
~/.venvs/torch/bin/python -u m3b_paged_attn.py --rows 24 --seq 192 2>&1 | tail -20
echo "--- 换一个长度再跑一次 ---"
~/.venvs/torch/bin/python -u m3b_paged_attn.py --rows 24 --seq 512 2>&1 | tail -12
