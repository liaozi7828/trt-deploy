"""M5 汇总图：prefix caching 省了什么、又撞上了什么地板。"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_OFF = "#8c8c8c"
C_ON = "#2e8b57"
C_RED = "#b03030"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="m5_data.json")
    ap.add_argument("--out", default="m5_summary.png")
    args = ap.parse_args()
    d = json.load(open(args.data))
    off, on = d["ttft_off_ms"], d["ttft_on_ms"]
    n = len(off)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    fig.suptitle("M5: prefix caching removes the shared prefill, then hits the same "
                 "per-forward floor\n"
                 f"Qwen2.5-0.5B, FP16 — {d['prefix_tokens']}-token shared prefix + "
                 f"{d['suffix_tokens']}-token unique suffix, block={d['block_size']}",
                 fontsize=11)

    ax = axes[0]
    x = np.arange(n)
    w = 0.38
    ax.bar(x - w / 2, off, w, color=C_OFF, label="no cache")
    ax.bar(x + w / 2, on, w, color=C_ON, label="prefix cache")
    top = max(off) * 1.35
    ax.axhline(d["forward_floor_ms"], ls="--", lw=1.2, color=C_RED,
               label=f"floor = {d['forward_floor_ms']:.0f} ms "
                     f"(one eager forward)")
    for i in range(n):
        ax.annotate(f"{off[i]:.0f}", (i - w / 2, off[i]), ha="center", va="bottom",
                    fontsize=7.5)
        ax.annotate(f"{on[i]:.0f}", (i + w / 2, on[i]), ha="center", va="bottom",
                    fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["req 0\n(cold)" if i == 0 else f"req {i}"
                        for i in range(n)], fontsize=9)
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_ylabel("TTFT (ms, prefill only)", fontsize=9)
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8, loc="upper right", frameon=False, ncol=1)
    ax.set_title("1. Requests 1-7 skip 512 of 526 prefill tokens\n"
                 f"TTFT {np.mean(off[1:]):.0f} -> {np.mean(on[1:]):.0f} ms "
                 f"({np.mean(off[1:]) / np.mean(on[1:]):.1f}x); all tokens identical",
                 fontsize=9.5)

    ax = axes[1]
    labels = ["no cache", "prefix cache"]
    totals = [d["total_prefill_off_ms"], d["total_prefill_on_ms"]]
    ax.bar(labels, totals, 0.55, color=[C_OFF, C_ON])
    top2 = max(totals) * 1.32
    for i, v in enumerate(totals):
        ax.annotate(f"{v:.0f} ms", (i, v), ha="center", va="bottom", fontsize=9)
    ax.annotate(f"{totals[0] / totals[1]:.2f}x total prefill",
                (0.5, top2 * 0.90), ha="center", va="center", fontsize=9.5,
                color=C_RED)
    ax.set_ylabel("total prefill time for 8 requests (ms)", fontsize=9)
    ax.set_ylim(0, top2)
    ax.grid(axis="y", alpha=.25)
    ax.set_title(f"2. 8 requests, {d['reused_kv_tokens']} KV tokens reused\n"
                 "(1 cold + 7 hits)", fontsize=9.5)

    fig.text(0.5, 0.005,
             "The floor is the same one measured in M0-M4: one eager forward costs "
             "~17-26 ms of Python + kernel-launch overhead, independent of token count.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    fig.savefig(args.out, dpi=130)
    print(f"已生成: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
