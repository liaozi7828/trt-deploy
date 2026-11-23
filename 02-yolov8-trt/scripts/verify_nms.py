#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_nms.py —— P3：把后处理和**独立实现**对比（这是唯一的验收）

⚠️ 为什么不用 CPU NMS vs GPU NMS 的 A/B 当验收：
   那两个实现出自同一个心智模型（同样的逐类、同样的 IoU>阈值、同样的阈值语义）。
   如果语义本身写错了，两边会一起错，A/B 永远通过。
   **A/B 只能筛掉实现 bug（索引、上界、传输），筛不掉逻辑 bug。**

   验证语义必须和"别人写的、被训练/评测流程验证过的"实现比：
   → ultralytics.utils.ops.non_max_suppression

本脚本做两级对比：

  ① 语义级：我们的 decode+NMS（Python 复刻）  vs  ultralytics.non_max_suppression
            只吃 raw output，不涉及 C++。回答"NMS 的语义对不对"。

  ② 实现级：C++ 的 --dump-dets 输出（真实实现）  vs  ultralytics.non_max_suppression
            需要先跑一次 C++：
              ./build/bin/yolov8_demo --engine models/yolov8n.engine \\
                  --image models/bus.ppm --dump-output dump/out.bin --dump-dets dump/dets.txt
            回答"我的 C++ 实现对不对"。

用法：
    # 只做语义级（给 raw output 就行）
    python scripts/verify_nms.py --raw dump/out.bin

    # 语义级 + 实现级（再加 C++ 的最终框）
    python scripts/verify_nms.py --raw dump/out.bin --dets dump/dets.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 读 C++ 的 writeTensorBin 格式：int32 ndim + ndim×int32 dims + float32 数据
# ---------------------------------------------------------------------------
def load_tensor_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    ndim = np.frombuffer(data, dtype=np.int32, count=1)[0]
    dims = np.frombuffer(data, dtype=np.int32, count=ndim, offset=4)
    offset = 4 + 4 * ndim
    n = int(np.prod(dims))
    # copy：np.frombuffer 出来的是只读视图，torch.from_numpy 会警告
    arr = np.frombuffer(data, dtype=np.float32, count=n, offset=offset).copy()
    print(f"raw: {path.name}  shape={tuple(int(x) for x in dims)}  ({n} 个 float)")
    return arr.reshape(tuple(int(x) for x in dims))


def load_dets(path: Path):
    """读 C++ --dump-dets 的文件，返回 (dets[N,6], 参数 dict)"""
    params = {}
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        params[k] = float(v)
                    except ValueError:
                        pass
            continue
        v = line.split()
        if len(v) >= 6:
            rows.append([float(v[0]), float(v[1]), float(v[2]),
                         float(v[3]), float(v[4]), float(v[5])])
    dets = np.array(rows, dtype=np.float32).reshape(-1, 6)
    print(f"dets: {path.name}  {len(dets)} 个框  参数={params}")
    return dets, params


# ---------------------------------------------------------------------------
# 纯 numpy 实现（和 C++ 的 decodeV8 / nms 同语义，用于"语义级"对比）
# ---------------------------------------------------------------------------
def decode_v8(y: np.ndarray, conf: float) -> np.ndarray:
    p = y[0]                        # [4+nc, 8400]
    boxes = p[:4]
    scores = p[4:]
    cls = scores.argmax(axis=0)
    sc = scores.max(axis=0)
    keep = sc > conf
    b = boxes[:, keep]
    return np.stack([
        cls[keep].astype(np.float32), sc[keep].astype(np.float32),
        b[0] - b[2] * 0.5, b[1] - b[3] * 0.5,
        b[0] + b[2] * 0.5, b[1] + b[3] * 0.5,
    ], axis=1)


def nms(dets: np.ndarray, iou_thres: float) -> np.ndarray:
    if len(dets) == 0:
        return dets
    dets = dets[np.argsort(-dets[:, 1])]
    removed = np.zeros(len(dets), dtype=bool)
    keep = []
    for i in range(len(dets)):
        if removed[i]:
            continue
        keep.append(i)
        a = dets[i]
        for j in range(i + 1, len(dets)):
            if removed[j] or dets[j, 0] != a[0]:
                continue
            b = dets[j]
            ix1, iy1 = max(a[2], b[2]), max(a[3], b[3])
            ix2, iy2 = min(a[4], b[4]), min(a[5], b[5])
            iw, ih = ix2 - ix1, iy2 - iy1
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            ua = (a[4] - a[2]) * (a[5] - a[3])
            ub = (b[4] - b[2]) * (b[5] - b[3])
            if inter / max(ua + ub - inter, 1e-9) > iou_thres:
                removed[j] = True
    return dets[keep]


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(ua + ub - inter, 1e-9)


def undo_letterbox(boxes: np.ndarray, scale, pad_x, pad_y, w, h) -> np.ndarray:
    """把 640 空间的框映射回原图空间（和 C++ 的 undoLetterbox 一致，含 clamp）"""
    b = boxes.copy()
    for k in (0, 2):
        b[:, k] = (b[:, k] - pad_x) / scale
    for k in (1, 3):
        b[:, k] = (b[:, k] - pad_y) / scale
    b[:, 0] = np.clip(b[:, 0], 0, w - 1)
    b[:, 2] = np.clip(b[:, 2], 0, w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, h - 1)
    b[:, 3] = np.clip(b[:, 3], 0, h - 1)
    return b


def match(a: np.ndarray, b: np.ndarray, iou_thres: float = 0.5):
    """按类别 + IoU 贪心匹配两组框（a: 参考，b: 待测）"""
    used = set()
    pairs = []
    for i, x in enumerate(a):
        best_j, best_u = -1, iou_thres
        for j, y in enumerate(b):
            if j in used or int(x[0]) != int(y[0]):
                continue
            u = iou_xyxy(x[2:6], y[2:6])
            if u > best_u:
                best_u, best_j = u, j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((i, best_j, best_u))
    only_a = [i for i in range(len(a)) if i not in {p[0] for p in pairs}]
    only_b = [j for j in range(len(b)) if j not in used]
    return pairs, only_a, only_b


def from_ultralytics(d: np.ndarray) -> np.ndarray:
    """ultralytics 输出 [x1, y1, x2, y2, conf, cls] → 我们的 [cls, score, x1, y1, x2, y2]

    ⚠️ 这是个真实的坑：ultralytics 的 cls 在**最后一列**，
       而我们的内部布局 cls 在**第一列**。
       直接拿两边的第 0 列当类别比，会得出"0 对匹配"的假结论
       （数量和分数其实完全一致）。
    """
    if len(d) == 0:
        return np.zeros((0, 6), np.float32)
    return np.stack([d[:, 5], d[:, 4], d[:, 0], d[:, 1], d[:, 2], d[:, 3]],
                    axis=1).astype(np.float32)


def report(name: str, ref: np.ndarray, test: np.ndarray):
    print(f"\n{'=' * 68}\n{name}\n{'-' * 68}")
    print(f"  参考（ultralytics）: {len(ref)} 个框")
    print(f"  待测              : {len(test)} 个框")

    if len(ref) == 0 and len(test) == 0:
        print("  两边都空 → 无法判断（换一张有目标的图）")
        return None

    pairs, only_ref, only_test = match(ref, test)
    print(f"  匹配上            : {len(pairs)} 对（同类别 + IoU > 0.5）")
    print(f"  只有参考有        : {len(only_ref)} 个（漏检）")
    print(f"  只有待测有        : {len(only_test)} 个（多检）")

    if not pairs:
        print("  ❌ 一对都没匹配上 —— 语义不一致")
        # 打印两边前几个框，方便定位是"列顺序"还是"坐标系"的问题
        print(f"\n  调试：参考前 3 个 = {np.round(ref[:3], 2).tolist()}")
        print(f"  调试：待测前 3 个 = {np.round(test[:3], 2).tolist()}")
        print("  （格式均为 [cls, score, x1, y1, x2, y2]）")
        return False

    print(f"\n  {'类别':>4s} {'参考分数':>9s} {'待测分数':>9s} {'分数差':>8s} "
          f"{'IoU':>6s} {'坐标最大差':>11s}")
    worst_c, worst_s = 0.0, 0.0
    for i, j, u in pairs[:12]:
        x, y = ref[i], test[j]
        cd = max(abs(x[2] - y[2]), abs(x[3] - y[3]), abs(x[4] - y[4]), abs(x[5] - y[5]))
        sd = abs(x[1] - y[1])
        worst_c, worst_s = max(worst_c, cd), max(worst_s, sd)
        print(f"  {int(x[0]):>4d} {x[1]:>9.4f} {y[1]:>9.4f} {sd:>8.4f} "
              f"{u:>6.3f} {cd:>11.2f} px")

    print(f"\n  最大坐标差 = {worst_c:.2f} px    最大分数差 = {worst_s:.4f}")

    # 判据用 IoU，不用像素距离。
    #
    # 为什么：NMS 里常有"近重复候选"（分数几乎相同、位置差几像素），
    # 贪心只保留其中一个——保留哪个取决于排序的 tie-break，
    # 而 std::sort（不稳定）和 np.argsort（稳定）给出的顺序不同。
    # 这属于"同一个物体的两种等优解"，不是错误。
    # 实测过：bus 那一框分数完全相同（差 0.0000）、IoU 0.972、位置差 5.25 px。
    worst_iou = min(u for _, _, u in pairs)
    ok = (len(only_ref) == 0 and len(only_test) == 0 and worst_iou >= 0.90)
    print(f"  最小 IoU   = {worst_iou:.3f}")
    print(f"  判据（无漏检 + 无多检 + 最小 IoU ≥ 0.90）: "
          f"{'✅ 通过' if ok else '❌ 不通过'}")
    if not ok and len(only_ref) == 0 and len(only_test) == 0:
        print("  提示：数量一致但 IoU 偏低 → 检查坐标系（letterbox / 通道顺序）")
    return ok


def import_ultralytics_nms():
    """在 ultralytics 里找到 non_max_suppression

    8.x 里它搬过家（utils.ops → utils.nms → ...），所以逐个模块探测，
    而不是写死 import 路径。找不到就返回 None，由调用方降级到 torchvision。
    """
    import importlib

    candidates = [
        "ultralytics.utils.nms",
        "ultralytics.utils.ops",
        "ultralytics.utils",
        "ultralytics",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        fn = getattr(mod, "non_max_suppression", None)
        if callable(fn):
            print(f"参考实现: {mod_name}.non_max_suppression")
            return fn
    print("参考实现: 找不到 ultralytics.non_max_suppression")
    return None


def torchvision_nms(dets: np.ndarray, iou_thres: float) -> np.ndarray:
    """torchvision 的 NMS 作为独立参考（它做全局 NMS，所以逐类调用）

    注意：这里"逐类"这个决定由我们代码写出，所以它验证的是
    "IoU 定义 + 贪心选择"，不验证"逐类"这条语义本身。
    但如果 ultralytics 能导入，就优先用它（连"逐类"一起验证）。
    """
    import torch
    from torchvision.ops import nms as tv_nms

    if len(dets) == 0:
        return dets
    keep_all = []
    for c in np.unique(dets[:, 0]):
        idx = np.where(dets[:, 0] == c)[0]
        boxes = torch.from_numpy(dets[idx, 2:6].astype(np.float32))
        scores = torch.from_numpy(dets[idx, 1].astype(np.float32))
        keep = tv_nms(boxes, scores, iou_thres).cpu().numpy()
        keep_all.extend(idx[keep].tolist())
    keep_all = sorted(keep_all, key=lambda i: -dets[i, 1])
    return dets[keep_all]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="C++ --dump-output 导出的 raw 输出")
    ap.add_argument("--dets", default=None, help="C++ --dump-dets 导出的最终框（可选）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    import torch

    raw = load_tensor_bin(Path(args.raw))
    nc = raw.shape[1] - 4
    print(f"类别数: {nc}")

    # ================= 参考实现 =================
    ref_np = None
    nms_fn = import_ultralytics_nms()
    if nms_fn is not None:
        pred = torch.from_numpy(raw)
        with torch.no_grad():
            out = nms_fn(pred, conf_thres=args.conf, iou_thres=args.iou,
                         agnostic=False, max_det=300)
        ref = out[0]
        if ref is not None and len(ref):
            # ★ 必须换列顺序（ultralytics 的 cls 在最后一列）
            ref_np = from_ultralytics(ref.cpu().numpy())
        else:
            ref_np = np.zeros((0, 6), np.float32)
    else:
        # 降级：torchvision（它做全局 NMS，逐类调用由我们补，所以只验证 IoU+贪心）
        print("降级为 torchvision.ops.nms 作为参考（逐类调用）")
        ref_np = torchvision_nms(decode_v8(raw, args.conf), args.iou)

    # ================= ① 语义级：我们的 Python 复刻 =================
    mine = nms(decode_v8(raw, args.conf), args.iou)

    print(f"\n[① 语义级] 都是 640×640 letterbox 空间，直接比")
    ok1 = report("我们的 decode+NMS  vs  ultralytics.non_max_suppression",
                 ref_np, mine)

    # ================= ② 实现级：C++ 的真实输出 =================
    ok2 = None
    if args.dets:
        dets, prm = load_dets(Path(args.dets))
        scale = prm.get("scale", 0.0)
        pad_x = prm.get("pad_x", 0.0)
        pad_y = prm.get("pad_y", 0.0)
        sr_w = int(prm.get("src_w", 0))
        sr_h = int(prm.get("src_h", 0))

        if scale <= 0 or sr_w <= 0 or sr_h <= 0:
            print("\n[② 实现级] 跳过：dets 文件头缺 letterbox 参数")
        else:
            # 把 ultralytics 的框也映射到原图空间，两边同坐标系
            ref_orig = ref_np.copy()
            if len(ref_orig):
                ref_orig[:, 2:6] = undo_letterbox(ref_orig[:, 2:6], scale, pad_x,
                                                  pad_y, sr_w, sr_h)
            print(f"\n[② 实现级] 映射到原图 {sr_w}×{sr_h} 后比")
            ok2 = report("C++ 实现（--dump-dets）  vs  ultralytics.non_max_suppression",
                         ref_orig, dets)

    # ================= 总结 =================
    print(f"\n{'=' * 68}\n结论\n{'-' * 68}")
    print(f"  ① 语义（decode + NMS 的定义）      : "
          f"{'✅ 一致' if ok1 else '❌ 不一致'}")
    if ok2 is not None:
        print(f"  ② 实现（C++ 真实输出）             : "
              f"{'✅ 一致' if ok2 else '❌ 不一致'}")
    print("\n  说明：这两项才回答\"后处理对不对\"。")
    print("  CPU/GPU NMS 的 A/B 只回答\"两个实现是否一致\"，不能替代本脚本。")

    return 0 if (ok1 and (ok2 is None or ok2)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
