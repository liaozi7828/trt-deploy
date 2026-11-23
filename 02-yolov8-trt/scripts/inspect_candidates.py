#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_candidates.py —— 把 8400 → 46 → 5 这个过程完整打印出来

回答三个问题：
  ① 8400 个锚点里，有多少个分数超过阈值？它们都是什么类别？
  ② 46 个候选有多大比例是"同一个物体的重复框"？
  ③ NMS 到底删掉了谁、为什么删（IoU 多少）？

用法：
    # 先让 C++ 导出原始输出
    ./build/bin/yolov8_demo --engine models/yolov8n.engine --image models/bus.ppm \
        --dump-output dump/out.bin

    python scripts/inspect_candidates.py --raw dump/out.bin --conf 0.25 --iou 0.45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light "
        "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
        "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
        "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
        "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase "
        "scissors teddy_bear hair_drier toothbrush").split()


def load_tensor_bin(path: Path) -> np.ndarray:
    """读 C++ 的 writeTensorBin：int32 ndim + ndim×int32 dims + float32"""
    data = path.read_bytes()
    ndim = int(np.frombuffer(data, dtype=np.int32, count=1)[0])
    dims = [int(x) for x in np.frombuffer(data, dtype=np.int32, count=ndim, offset=4)]
    off = 4 + 4 * ndim
    n = int(np.prod(dims))
    arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).copy()
    return arr.reshape(tuple(dims))


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(ua + ub - inter, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dump/out.bin")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--top", type=int, default=20, help="NMS 前打印多少个候选")
    args = ap.parse_args()

    y = load_tensor_bin(Path(args.raw))
    nc = y.shape[1] - 4
    n_anchor = y.shape[2]
    p = y[0]
    scores = p[4:4 + nc]                       # [80, 8400]
    boxes = p[:4]                              # [4, 8400]  cx,cy,w,h

    print("=" * 72)
    print(f"原始输出 : {tuple(y.shape)}   = 4 坐标 + {nc} 类 × {n_anchor} 锚点")
    print(f"            {n_anchor} = 80² + 40² + 20² （P3/P4/P5 三个尺度的网格）")
    print("=" * 72)

    # ---------- ① 8400 → ?  ----------
    best = scores.max(axis=0)
    best_c = scores.argmax(axis=0)

    print(f"\n① 阈值过滤（conf > {args.conf}）")
    for th in (0.9, 0.7, 0.5, 0.25, 0.1, 0.05, 0.01):
        print(f"     conf > {th:<5} → {(best > th).sum():6d} 个候选")
    keep = best > args.conf
    print(f"  实际采用 conf > {args.conf}  →  **{keep.sum()} 个候选**")

    # 类别分布
    print("\n   类别分布：")
    u, cnt = np.unique(best_c[keep], return_counts=True)
    for c, n in sorted(zip(u, cnt), key=lambda t: -t[1]):
        name = COCO[c] if c < len(COCO) else f"cls{c}"
        print(f"     {name:<14} {n:3d} 个")

    # 分数范围
    print(f"\n   分数范围：{best[keep].min():.4f} ~ {best[keep].max():.4f}")

    # ---------- ② 位出候选，准备 NMS ----------
    dets = np.stack([
        best_c[keep].astype(np.float32),
        best[keep].astype(np.float32),
        boxes[0][keep] - boxes[2][keep] * 0.5,
        boxes[1][keep] - boxes[3][keep] * 0.5,
        boxes[0][keep] + boxes[2][keep] * 0.5,
        boxes[1][keep] + boxes[3][keep] * 0.5,
    ], axis=1)
    order = np.argsort(-dets[:, 1])
    dets = dets[order]

    print(f"\n② NMS 前的候选（按分数降序，显示前 {min(args.top, len(dets))} 个）")
    print(f"   {'#':>3} {'类别':<12} {'分数':>7}   {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7}")
    for i in range(min(args.top, len(dets))):
        d = dets[i]
        name = COCO[int(d[0])] if int(d[0]) < len(COCO) else f"cls{int(d[0])}"
        print(f"   {i:>3} {name:<12} {d[1]:>7.4f}   {d[2]:>7.1f} {d[3]:>7.1f} "
              f"{d[4]:>7.1f} {d[5]:>7.1f}")
    if len(dets) > args.top:
        print(f"   ... 还有 {len(dets) - args.top} 个")

    # ---------- ③ NMS 过程 ----------
    print(f"\n③ NMS（IoU > {args.iou} 且同类别 → 抑制）")
    removed = np.zeros(len(dets), dtype=bool)
    kept = []
    suppressed_by = {}
    for i in range(len(dets)):
        if removed[i]:
            continue
        kept.append(i)
        d = dets[i]
        name_i = COCO[int(d[0])] if int(d[0]) < len(COCO) else f"cls{int(d[0])}"
        n_sup = 0
        for j in range(i + 1, len(dets)):
            if removed[j] or int(dets[j][0]) != int(d[0]):
                continue
            u = iou(d[2:6], dets[j][2:6])
            if u > args.iou:
                removed[j] = True
                suppressed_by[j] = (i, u)
                n_sup += 1
        if n_sup:
            print(f"   保留 #{i:<3} {name_i:<12} {d[1]:.4f}"
                  f"  →  抑制了 {n_sup} 个重复框")

    print(f"\n④ 结果：**{len(dets)} 个候选 → {len(kept)} 个框**"
          f"（删掉 {len(dets) - len(kept)} 个，占 "
          f"{100.0 * (len(dets) - len(kept)) / max(len(dets), 1):.0f}%）")
    print(f"\n   {'类别':<12} {'分数':>7}   {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7}")
    for i in kept:
        d = dets[i]
        name = COCO[int(d[0])] if int(d[0]) < len(COCO) else f"cls{int(d[0])}"
        print(f"   {name:<12} {d[1]:>7.4f}   {d[2]:>7.1f} {d[3]:>7.1f} "
              f"{d[4]:>7.1f} {d[5]:>7.1f}")

    # 每个保留框"吃掉"了多少重复
    print("\n   每个框吃掉的重复数：")
    for i in kept:
        d = dets[i]
        name = COCO[int(d[0])] if int(d[0]) < len(COCO) else f"cls{int(d[0])}"
        n = sum(1 for j, (src, _) in suppressed_by.items() if src == i)
        print(f"     {name:<12} {d[1]:.4f}  ←  {n} 个重复")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
