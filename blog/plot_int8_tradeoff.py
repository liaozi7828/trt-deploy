#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_int8_tradeoff.py —— 博客配图：YOLOv8n 的精度-延迟权衡（三栏）

设计原则（这是重点，不是代码技巧）：
  ① **y 轴一律从 0 开始**。柱状图从非零起点开始是最常见的误导手段之一，
     而这篇博客的卖点恰恰是"我的数字可信"，不能在图上翻车。
  ② 但"INT8 掉了 1.27 pt"必须**看得见** —— 解法是**参考线 + 差值箭头**，
     不是截断 y 轴。读者看到"INT8 比虚线矮一截"，同时知道绝对高度。
  ③ 颜色全局一致：FP32 灰 / FP16 蓝 / INT8 红。
  ④ 相对差值写进 x 轴刻度标签（"+17.6%"、"2.21x"），不额外占地方。

用英文标签是故意的：WSL 里通常没装中文字体，matplotlib 会把中文渲染成方块。

用法：
    pip install matplotlib
    python plot_int8_tradeoff.py --out fig-int8-tradeoff.png

数据只有 8 个数字，都在下面 PANELS 里，改起来很容易。
"""
from __future__ import annotations

import argparse

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    raise SystemExit(
        "需要 matplotlib：  pip install matplotlib\n"
        "（数据就 8 个数字，也可以自己用别的工具画。）")

# ---------------- 实测数据（COCO val2017 5000 张；延迟为同进程测量）------------
PANELS = [
    {
        "title": "Accuracy\n(mAP@0.5, COCO val2017)",
        # ★ 相对差值放进刻度标签（和另外两栏一致），**不用浮动箭头**。
        #   原因：0.5165-0.5038=0.0127，在 0~0.63 的轴上只占 2% 高度 ——
        #   箭头画出来必然看不清，而且会盖住柱子上的数值标签。
        #   让"三根柱子几乎等高"如实呈现，反而是这篇文章要说的：
        #   精度几乎没变，但延迟差很多 → 所以选 FP16。
        "cats": ["FP32\n(reference)", "FP16\n(~0)", "INT8\n(-1.27 pt)"],
        "vals": [0.5165, 0.5167, 0.5038],
        "colors": ["#9E9E9E", "#4C78A8", "#E45756"],
        "fmt": "{:.4f}",
        "ref": 0,                      # 参考线画在这个柱子的值上
        "delta": None,
        "ylim_max": 0.63,
        "note": "bars nearly equal by design: the drop is only 2.5% relative",
    },
    {
        "title": "GPU latency\n(TensorRT, ms)",
        "cats": ["FP16\n(reference)", "INT8\n(+17.6%)"],
        "vals": [2.10, 2.47],
        "colors": ["#4C78A8", "#E45756"],
        "fmt": "{:.2f}",
        "ref": 0,
        "delta": None,
        "ylim_max": 2.95,
        "note": "FP32 not measured on TRT here",
    },
    {
        "title": "CPU latency\n(onnxruntime, ms/img)",
        "cats": ["FP32\n(2.21x slower)", "FP16\n(reference)", "INT8\n(+59%)"],
        "vals": [135.0, 61.0, 97.0],
        "colors": ["#9E9E9E", "#4C78A8", "#E45756"],
        "fmt": "{:.0f}",
        "ref": 1,
        "delta": None,
        "ylim_max": 168.0,
        "note": "same process, sequential",
    },
]


def draw_panel(ax, cfg):
    vals = cfg["vals"]
    bars = ax.bar(range(len(vals)), vals, width=0.56,
                  color=cfg["colors"], edgecolor="white", linewidth=1.4, zorder=3)

    # 数值标在柱顶
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + cfg["ylim_max"] * 0.018,
                cfg["fmt"].format(v), ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", zorder=4)

    # 参考线（灰虚线，标出基准值）
    ref_val = vals[cfg["ref"]]
    ax.axhline(ref_val, color="#666666", linestyle="--", linewidth=1.1, zorder=2)

    # 不画浮动差值箭头：精度栏的差异只有轴的 2% 高，箭头必然盖住数值标签且看不清。
    # 相对差值统一放在 x 轴刻度标签里（见 PANELS 的 cats）。

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(cfg["cats"], fontsize=10.5)
    ax.set_ylim(0, cfg["ylim_max"])
    ax.set_xlim(-0.62, len(vals) - 0.38)
    ax.set_title(cfg["title"], fontsize=12, pad=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.22, linestyle=":", zorder=0)
    ax.text(0.5, -0.225, cfg["note"], transform=ax.transAxes,
            ha="center", va="top", fontsize=8.2, color="#888888")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig-int8-tradeoff.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9))
    for ax, cfg in zip(axes, PANELS):
        draw_panel(ax, cfg)

    fig.suptitle("YOLOv8n: INT8 loses accuracy and is slower on both backends",
                 fontsize=14.5, fontweight="bold", y=0.995)

    fig.legend(handles=[
        Patch(facecolor="#9E9E9E", label="FP32"),
        Patch(facecolor="#4C78A8", label="FP16 (recommended)"),
        Patch(facecolor="#E45756", label="INT8"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.005))

    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.24, wspace=0.26)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")

    print(f"已生成: {args.out}")
    print("\n数字复核（和文章表格逐项对应）:")
    for cfg in PANELS:
        head = cfg["title"].split("\n")[0]
        body = "  ".join(f"{c.split(chr(10))[0]}={cfg['fmt'].format(v)}"
                         for c, v in zip(cfg["cats"], cfg["vals"]))
        print(f"  {head:14s} {body}")
    p0, p2 = PANELS[0]["vals"], PANELS[2]["vals"]
    print(f"\n精度栏差值: {(p0[2] - p0[0]) * 100:+.2f} pt （相对 FP32）")
    print(f"CPU: FP16 比 FP32 快 {p2[0] / p2[1]:.2f}x；"
          f"INT8 比 FP16 慢 {(p2[2] / p2[1] - 1) * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
