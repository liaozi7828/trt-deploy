#!/usr/bin/env bash
set -euo pipefail
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
cp -r "$WC/p5/." ~/projects/p5/
cd ~/projects/p5
~/.venvs/torch/bin/python plot_m5.py --out m5_summary.png
cp m5_summary.png "$WC/p5/"
