#!/usr/bin/env bash
# 画 M4 汇总图并回传 Windows 工作区
set -euo pipefail
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
cp -r "$WC/p5/." ~/projects/p5/
cd ~/projects/p5
~/.venvs/torch/bin/python plot_m4.py --out m4_summary.png
cp m4_summary.png "$WC/p5/"
