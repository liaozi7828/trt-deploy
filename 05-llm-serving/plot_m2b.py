#!/usr/bin/env python
"""把 m2b_continuous.py 写出的若干 JSON 画成一张对比图。

用法：
    python plot_m2b.py --out m2b_summary.png m2b_A.json m2b_B.json ...

图的设计原则（和 blog/plot_int8_tradeoff.py 一致）：
  * 所有柱状图的 y 轴从 0 起，不截断。
  * 三个面板分别回答三个问题：吞吐多少、步数多少、尾延迟多高。
  * 相对差值写进 x 轴刻度标签，不画浮动箭头（柱子高度差只有百分之几，箭头一定盖住标签）。
"""
from __future__ import annotations

import argparse
import json


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_of(d: dict) -> str:
    c = d["config"]
    new = c["newens"]
    rng = f"{min(new)}-{max(new)}" if len(set(new)) > 1 else str(new[0])
    return f'{c["reqs"]}/{c["slots"]}\nnew {rng}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="m2b_summary.png")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("jsons", nargs="+")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = [(label_of(d), d) for d in (load(p) for p in args.jsons)]
    # 每个 run 的两行：wave 在前、continuous 在后
    rows = []
    for lab, d in runs:
        by = {r["name"]: r for r in d["rows"]}
        rows.append((lab, by["wave"], by["continuous"]))

    x = np.arange(len(rows))
    w = 0.36
    c_wave, c_cont = "#8c8c8c", "#2f6fb3"

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

    # ---- 面板 1：吞吐 ----
    ax = axes[0]
    vw = [r[1]["tok_s"] for r in rows]
    vc = [r[2]["tok_s"] for r in rows]
    b1 = ax.bar(x - w / 2, vw, w, label="wave (no refill)", color=c_wave)
    b2 = ax.bar(x + w / 2, vc, w, label="continuous (refill)", color=c_cont)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    top1 = max(max(vw), max(vc)) * 1.38
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_title("Throughput (tokens/s, higher is better)\ngrey = wave, blue = continuous", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _, _ in rows], fontsize=8)
    ax.set_xlabel("requests / slots   (x = continuous ÷ wave)", fontsize=8)
    ax.set_ylim(0, top1)
    ax.grid(axis="y", alpha=.25)
    for i in range(len(rows)):
        ax.annotate(f"x{vc[i] / vw[i]:.2f}", (i, top1 * 0.95),
                    ha="center", va="center", fontsize=9.5, color="#b03030")

    # ---- 面板 2：解码步数（越少越好）----
    ax = axes[1]
    sw = [r[1]["steps"] for r in rows]
    sc = [r[2]["steps"] for r in rows]
    ax.bar(x - w / 2, sw, w, color=c_wave)
    ax.bar(x + w / 2, sc, w, color=c_cont)
    top = max(sw) * 1.38
    for i, (swi, sci) in enumerate(zip(sw, sc)):
        ax.annotate(f"{swi}", (i - w / 2, swi), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{sci}", (i + w / 2, sci), ha="center", va="bottom", fontsize=8)
        ax.annotate(f'{(1 - sci / swi) * 100:.0f}%', (i, top * 0.95),
                    ha="center", va="center", fontsize=9.5, color="#b03030")
    ax.set_title("Decode steps (fewer is better)\neach step re-reads all 942 MB of weights", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _, _ in rows], fontsize=8)
    ax.set_xlabel("requests / slots", fontsize=8)
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=.25)

    # ---- 面板 3：尾延迟 ----
    ax = axes[2]
    pw = [r[1]["p95"] for r in rows]
    pc = [r[2]["p95"] for r in rows]
    ax.bar(x - w / 2, pw, w, color=c_wave)
    ax.bar(x + w / 2, pc, w, color=c_cont)
    for i, (a, b) in enumerate(zip(pw, pc)):
        ax.annotate(f"{a:.0f}", (i - w / 2, a), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{b:.0f}", (i + w / 2, b), ha="center", va="bottom", fontsize=8)
    ax.set_title("p95 end-to-end latency (ms, lower is better)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _, _ in rows], fontsize=8)
    ax.set_xlabel("requests / slots", fontsize=8)
    ax.set_ylim(0, max(max(pw), max(pc)) * 1.22)
    ax.grid(axis="y", alpha=.25)

    fig.suptitle("Continuous batching cuts decode steps by ~25% but barely moves throughput\n"
                 "Qwen2.5-0.5B, FP16, one captured decode graph, same process / same run",
                 fontsize=10.5)
    fig.text(0.5, 0.005,
             "prefill still takes ~56% of the wall clock, and continuous refills one request at a time, "
             "so the saved steps are paid back as extra prefill calls",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"已生成: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
