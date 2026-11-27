#!/usr/bin/env bash
# 跑 m2b 的四组配置，每组写一个 JSON。用法：wsl -e bash <此文件>
set -e
WC=$(ls -d /mnt/c/Users/*/Documents/deepseek-harness/default-workspace | head -1)
cd ~/projects/p5
cp -f "$WC"/p5/m2b_continuous.py .
cp -f "$WC"/p5/plot_m2b.py .
export HF_ENDPOINT=https://hf-mirror.com
PY=~/.venvs/torch/bin/python
$PY -m py_compile m2b_continuous.py plot_m2b.py

echo "=========== A: 96 请求 / 24 槽，短输出 ==========="
$PY -u m2b_continuous.py --reqs 96  --slots 24 --pmin 48 --pmax 256 --newens 8,16,24,32  --out m2b_A.json 2>&1 | tail -10

echo "=========== B: 192 请求 / 24 槽，短输出（8 波）==========="
$PY -u m2b_continuous.py --reqs 192 --slots 24 --pmin 48 --pmax 256 --newens 8,16,24,32  --out m2b_B.json 2>&1 | tail -10

echo "=========== C: 48 请求 / 24 槽，长输出 ==========="
$PY -u m2b_continuous.py --reqs 48  --slots 24 --pmin 48 --pmax 256 --newens 64,96,128 --out m2b_C.json 2>&1 | tail -10

echo "=========== D: 12 请求 / 4 槽，小负载 ==========="
$PY -u m2b_continuous.py --reqs 12  --slots 4  --pmin 32 --pmax 128 --newens 2,4,6    --out m2b_D.json 2>&1 | tail -10

echo "=========== 画图 ==========="
$PY -u plot_m2b.py --out m2b_summary.png m2b_A.json m2b_B.json m2b_C.json m2b_D.json
