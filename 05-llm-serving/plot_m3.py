"""M3 writeup 的三面板图：数据全部来自 p5/m3_data.json（原始值见 logs/）。

用法：python plot_m3.py [--out m3_summary.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_DENSE = "#8c8c8c"
C_GATHER = "#c98f3a"
C_KERNEL = "#3d9a4a"
C_EAGER = "#8c8c8c"
C_GRAPH = "#2f6fb5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="m3_data.json")
    ap.add_argument("--out", default="m3_summary.png")
    args = ap.parse_args()
    d = json.loads(Path(args.data).read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))
    fig.suptitle("M3: paged KV cache is about fixed shapes, not raw speed\n"
                 "Qwen2.5-0.5B, FP16, RTX 5060 Laptop — same process, same run",
                 fontsize=13)

    # ---------------- 面板 1：attention 内核本身 ----------------
    ax = axes[0]
    b = d["m3b"]
    seqs = b["seq"]
    x = np.arange(len(seqs))
    w = 0.26
    ax.bar(x - w, b["dense_sdpa_us"], w, color=C_DENSE)
    ax.bar(x, b["gather_sdpa_us"], w, color=C_GATHER)
    ax.bar(x + w, b["paged_kernel_us"], w, color=C_KERNEL)
    top = max(b["gather_sdpa_us"]) * 1.42
    for i in range(len(seqs)):
        for dx, v in ((-w, b["dense_sdpa_us"][i]), (0, b["gather_sdpa_us"][i]),
                      (w, b["paged_kernel_us"][i])):
            ax.annotate(f"{v:.1f}", (i + dx, v), ha="center", va="bottom", fontsize=8)
        ax.annotate(f'{b["gather_sdpa_us"][i] / b["paged_kernel_us"][i]:.2f}x vs gather',
                    (i, top * 0.90), ha="center", va="center", fontsize=9.5, color="#b03030")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seq = {s}" for s in seqs], fontsize=9)
    ax.set_ylim(0, top)
    ax.set_ylabel("per-call time (us)", fontsize=9)
    ax.set_title("1. Attention kernel only — paged is 1.07x / 0.98x dense time\n"
                 "grey = dense SDPA · orange = gather+SDPA · green = paged kernel",
                 fontsize=9.5)
    ax.grid(axis="y", alpha=.25)

    # ---------------- 面板 2：整步时间 vs 行数 ----------------
    ax = axes[1]
    g = d["m3d"]
    rows = g["rows"]
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, g["eager_ms"], 0.4, color=C_EAGER)
    ax.bar(idx + 0.2, g["graph_ms"], 0.4, color=C_GRAPH)
    ax.axhline(g["floor_ms"], ls="--", lw=1.4, color="#555")
    ax.annotate(f'theory floor {g["floor_ms"]:.2f} ms', (-0.45, g["floor_ms"] + 0.3),
                ha="left", va="bottom", fontsize=8, color="#555")
    top = max(g["eager_ms"]) * 1.40
    for i in idx:
        ax.annotate(f'{g["graph_ms"][i]:.2f}', (i + 0.2, g["graph_ms"][i]),
                    ha="center", va="bottom", fontsize=8)
        ax.annotate(f'{g["eager_ms"][i]:.1f}', (i - 0.2, g["eager_ms"][i]),
                    ha="center", va="bottom", fontsize=8)
        ax.annotate(f'{g["speedup"][i]:.2f}x', (i, top * 0.96),
                    ha="center", va="center", fontsize=9.5, color="#b03030")
        ax.annotate(f'{g["tok_s"][i]:.0f} tok/s', (i, top * 0.88),
                    ha="center", va="center", fontsize=8, color="#2f6fb5")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{r} rows" for r in rows], fontsize=9)
    ax.set_ylim(0, top)
    ax.set_ylabel("one decode step (ms)", fontsize=9)
    ax.set_title("2. Whole step in one graph (grey = eager, blue = graph)\n"
                 "48x the rows yet only +15% step time", fontsize=9.5)
    ax.grid(axis="y", alpha=.25)

    # ---------------- 面板 3：KV 显存 ----------------
    ax = axes[2]
    kv = d["kv_mib"]
    idx = np.arange(len(kv["labels"]))
    ax.bar(idx - 0.2, kv["dense_equiv"], 0.4, color=C_DENSE)
    ax.bar(idx + 0.2, kv["used"], 0.4, color=C_KERNEL)
    top = max(kv["dense_equiv"]) * 1.30
    for i in idx:
        ax.annotate(f'{kv["dense_equiv"][i]:.0f}', (i - 0.2, kv["dense_equiv"][i]),
                    ha="center", va="bottom", fontsize=8)
        ax.annotate(f'{kv["used"][i]:.0f}', (i + 0.2, kv["used"][i]),
                    ha="center", va="bottom", fontsize=8)
        ax.annotate(f'-{100 * (1 - kv["used"][i] / kv["dense_equiv"][i]):.0f}%',
                    (i, top * 0.92), ha="center", va="center", fontsize=10, color="#b03030")
    ax.set_xticks(idx)
    ax.set_xticklabels(kv["labels"], fontsize=9)
    ax.set_ylim(0, top)
    ax.set_ylabel("KV cache (MiB)", fontsize=9)
    ax.set_title("3. What it buys: memory (grey = dense equiv., green = paged)\n"
                 "saving shrinks when all rows have the same length", fontsize=9.5)
    ax.grid(axis="y", alpha=.25)

    fig.text(0.5, 0.015,
             "Panel 2: step time is flat in rows because ~1900 kernel launches per step dominate; "
             "the graph removes the CPU launches, not the kernels.",
             ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.92))
    fig.savefig(args.out, dpi=160)
    print(f"已生成: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
