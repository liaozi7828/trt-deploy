"""M4 汇总图：尖峰削掉多少 / 切块的代价落在哪。"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_BLOCK = "#8c8c8c"
C_CHUNK = "#3b7dd8"
C_RED = "#b03030"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="m4_data.json")
    ap.add_argument("--out", default="m4_summary.png")
    args = ap.parse_args()
    d = json.load(open(args.data))
    a, m4c = d["m4a"], d["m4c"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    fig.suptitle("M4: chunked prefill trades a latency spike for per-chunk overhead\n"
                 "Qwen2.5-0.5B, FP16, RTX 5060 Laptop — same process, same run",
                 fontsize=12)

    # ---- 面板 1：阻塞 vs 切块的尖峰 ----
    ax = axes[0]
    x = np.arange(len(a))
    w = 0.36
    sp_b = [r["blocking"]["spike"] for r in a]
    sp_c = [r["chunked"]["spike"] for r in a]
    md_b = [r["blocking"]["median"] for r in a]
    md_c = [r["chunked"]["median"] for r in a]
    ax.bar(x - w / 2, sp_b, w, color=C_BLOCK, label="blocking")
    ax.bar(x + w / 2, sp_c, w, color=C_CHUNK, label="chunked")
    for i in range(len(a)):
        ax.hlines(md_b[i], i - w / 2 - w / 2, i - w / 2 + w / 2, color="#111",
                  lw=2.2)
        ax.hlines(md_c[i], i + w / 2 - w / 2, i + w / 2 + w / 2, color="#111",
                  lw=2.2)
    top = max(sp_b) * 1.42
    for i in range(len(a)):
        ax.annotate(f"{sp_b[i]:.0f}", (i - w / 2, sp_b[i]), ha="center",
                    va="bottom", fontsize=8)
        ax.annotate(f"{sp_c[i]:.0f}", (i + w / 2, sp_c[i]), ha="center",
                    va="bottom", fontsize=8)
        ax.annotate(f"-{(1 - sp_c[i] / sp_b[i]) * 100:.0f}% spike", (i, top * 0.94),
                    ha="center", va="center", fontsize=9, color=C_RED)
        ax.annotate(f"median {md_b[i]:.0f} -> {md_c[i]:.0f}", (i, top * 0.85),
                    ha="center", va="center", fontsize=7.5, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['rows']} decoding rows" for r in a], fontsize=9)
    ax.set_ylabel("one decode step (ms)", fontsize=9)
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("1. A long prefill blocks every decoding row\n"
                 "bars = worst step (grey blocking / blue chunked), "
                 "thick line = median", fontsize=9)

    # ---- 面板 2：切块的端到端代价 ----
    ax = axes[1]
    ks = [r["nchunks"] for r in m4c]
    ms = [r["prefill_ms"] for r in m4c]
    ax.bar(range(len(m4c)), ms, 0.6, color=C_CHUNK)
    top = max(ms) * 1.35
    for i, r in enumerate(m4c):
        ax.annotate(f"{r['prefill_ms']:.0f} ms\n{r['prefill_tps']:.0f} tok/s",
                    (i, r["prefill_ms"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(m4c)))
    ax.set_xticklabels([f"{r['nchunks']} x {r['chunk']}" for r in m4c], fontsize=8)
    ax.set_xlabel("chunks x chunk size (prompt = 1024 tokens)", fontsize=8)
    ax.set_ylabel("prefill wall clock (ms)", fontsize=9)
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("2. Same prompt, same total work\n"
                 "splitting into 15 chunks costs 7.2x prefill time", fontsize=9.5)

    # ---- 面板 3：代价落在哪 ----
    ax = axes[2]
    base_ms = m4c[0]["prefill_ms"]
    base_k = d["m4b_kernel_only_ms"]["1"]              # ms
    xs, de2e, dk = [], [], []
    for r in m4c[1:]:
        n = r["nchunks"]
        k_key = str(n if n <= 8 else 8)
        xs.append(n)
        de2e.append(r["prefill_ms"] - base_ms)
        dk.append(d["m4b_kernel_only_ms"][k_key] - base_k)
    idx = np.arange(len(xs))
    ax.bar(idx - 0.2, de2e, 0.4, color=C_RED, label="end-to-end prefill Δ (ms)")
    ax.bar(idx + 0.2, [max(x, 1e-3) for x in dk], 0.4, color=C_CHUNK,
           label="attention kernel Δ (ms)")
    ax.set_yscale("log")
    for i in range(len(xs)):
        ax.annotate(f"{de2e[i]:.0f}", (i - 0.2, de2e[i]), ha="center",
                    va="bottom", fontsize=8)
        ax.annotate(f"{dk[i]:+.2f}", (i + 0.2, max(dk[i], 1e-3)), ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{n} chunks" for n in xs], fontsize=9)
    ax.set_ylabel("extra time vs 1 chunk (ms, log)", fontsize=9)
    ax.set_ylim(5e-4, 1e3)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("3. Where the penalty goes\n"
                 "red = whole prefill Δ, blue = attention kernel Δ", fontsize=9.5)

    fig.text(0.5, 0.005,
             "Panel 3: attention kernel cost is from the kernel-only benchmark (T=1024); "
             "everything else is per-forward Python/launch overhead (~25 ms per chunk).",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(args.out, dpi=130)
    print(f"已生成: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
