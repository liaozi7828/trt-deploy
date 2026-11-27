#!/usr/bin/env bash
# 只重画 M3 的图并把产物回传到 Windows 工作区
set -euo pipefail
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
cd "$HOME/projects/p5"
cp "$WC"/p5/m3_data.json "$WC"/p5/plot_m3.py . 2>/dev/null || true
~/.venvs/torch/bin/python plot_m3.py --out m3_summary.png
cp m3_summary.png "$WC/p5/" 2>/dev/null || true
echo "已生成: m3_summary.png"
