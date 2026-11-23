#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_map.py —— P3 W4/W5/W6：mAP 评测（FP32 / FP16 / INT8 三方对比）

为什么用 onnxruntime 而不是 TRT 跑：
  ① 我们已经验证过 ORT 和 TRT 的数值差 1.87e-06（相对），足够算 mAP
  ② 免去 C++ 侧逐图导出的麻烦（128 张 × 3 个引擎 = 384 次进程启动）
  ③ **同一个工具链测三方**，避免"引擎差异"混进"精度差异"
  代价：mAP 不是 TRT 上测的 —— 但"掉点"是相对量，结论不受影响。

数据集：COCO128（128 张带 YOLO 标注，约 7 MB）
  首次运行会自动下载；如果下载失败，手动下：
    wget "https://ghproxy.net/https://ultralytics.com/assets/coco128.zip"

用法：
    # 三方对比（W4）
    python scripts/eval_map.py --models models/yolov8n.onnx \
        models/yolov8n_fp16.onnx models/yolov8n_int8.onnx

    # 只看某个模型
    python scripts/eval_map.py --models models/yolov8n_int8.onnx --limit 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 前处理 / 后处理（和 C++ 侧同逻辑）
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int, pad: int = 114):
    """等比缩放 + 居中填充，返回 (图像, scale, pad_x, pad_y)"""
    from PIL import Image

    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)

    pil = Image.fromarray(img).resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(pil, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0        # [H,W,3]
    chw = np.transpose(arr, (2, 0, 1))[None, ...]             # [1,3,H,W]
    return np.ascontiguousarray(chw), scale, pad_x, pad_y


def decode_v8(y: np.ndarray, nc: int, conf: float):
    """返回 Nx6: [cls, score, x1, y1, x2, y2]（letterbox 空间）"""
    p = y[0]                       # [4+nc, 8400]
    boxes = p[:4]
    scores = p[4:4 + nc]
    cls = scores.argmax(axis=0)
    sc = scores.max(axis=0)
    keep = sc > conf

    b = boxes[:, keep]
    out = np.stack([
        cls[keep].astype(np.float32),
        sc[keep].astype(np.float32),
        b[0] - b[2] * 0.5,
        b[1] - b[3] * 0.5,
        b[0] + b[2] * 0.5,
        b[1] + b[3] * 0.5,
    ], axis=1)
    return out


def undo_letterbox(dets: np.ndarray, scale: float, pad_x: int, pad_y: int,
                   w: int, h: int) -> np.ndarray:
    if len(dets) == 0:
        return dets
    d = dets.copy()
    d[:, 2] = (d[:, 2] - pad_x) / scale
    d[:, 3] = (d[:, 3] - pad_y) / scale
    d[:, 4] = (d[:, 4] - pad_x) / scale
    d[:, 5] = (d[:, 5] - pad_y) / scale
    d[:, 2] = np.clip(d[:, 2], 0, w - 1)
    d[:, 3] = np.clip(d[:, 3], 0, h - 1)
    d[:, 4] = np.clip(d[:, 4], 0, w - 1)
    d[:, 5] = np.clip(d[:, 5], 0, h - 1)
    return d


def nms(dets: np.ndarray, iou_thres: float) -> np.ndarray:
    """逐类贪心 NMS（和 C++ 的 nms() 同逻辑）"""
    if len(dets) == 0:
        return dets
    order = np.argsort(-dets[:, 1])
    dets = dets[order]

    keep = []
    suppressed = np.zeros(len(dets), dtype=bool)

    for i in range(len(dets)):
        if suppressed[i]:
            continue
        keep.append(i)
        a = dets[i]
        for j in range(i + 1, len(dets)):
            if suppressed[j] or dets[j, 0] != a[0]:
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
                suppressed[j] = True
    return dets[keep]


# ---------------------------------------------------------------------------
# mAP 计算
# ---------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [N,4] b: [M,4] → [N,M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101 点插值（COCO 风格）"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(idx, mrec, mpre)))


def evaluate(per_image, gt_by_img, nc: int, iou_thres: float = 0.5):
    """per_image: {img_id: Nx6 dets}；gt_by_img: {img_id: Mx5 (cls,x1,y1,x2,y2)}"""
    aps = []
    for c in range(nc):
        # 收集所有图片里该类的预测和标注
        all_pred = []      # (score, img_id, box)
        n_gt = 0
        gt_count = {}
        for img_id, dets in per_image.items():
            for d in dets:
                if int(d[0]) == c:
                    all_pred.append((float(d[1]), img_id, d[2:6]))
        for img_id, gts in gt_by_img.items():
            n = int((gts[:, 0] == c).sum()) if len(gts) else 0
            gt_count[img_id] = n
            n_gt += n
        if n_gt == 0:
            continue
        if not all_pred:
            aps.append(0.0)
            continue

        all_pred.sort(key=lambda t: -t[0])
        matched = {img_id: np.zeros(gt_count[img_id], dtype=bool) for img_id in gt_count}

        tp = np.zeros(len(all_pred))
        fp = np.zeros(len(all_pred))

        for k, (score, img_id, box) in enumerate(all_pred):
            gts = gt_by_img.get(img_id, np.zeros((0, 5)))
            g = gts[gts[:, 0] == c] if len(gts) else np.zeros((0, 5))
            if len(g) == 0:
                fp[k] = 1
                continue
            ious = iou_matrix(box[None, :], g[:, 1:6])[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thres and not matched[img_id][j]:
                matched[img_id][j] = True
                tp[k] = 1
            else:
                fp[k] = 1

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        recall = tp_c / n_gt
        precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        aps.append(compute_ap(recall, precision))

    return (float(np.mean(aps)) if aps else 0.0), len(aps)


# ---------------------------------------------------------------------------
def load_dataset(root: Path, names: list):
    """读 coco128 的图片和 YOLO 标注"""
    import cv2

    img_dir = root / "images" / "train2017"
    lbl_dir = root / "labels" / "train2017"
    if not img_dir.exists():
        img_dir = root / "images"
        lbl_dir = root / "labels"

    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    items = []
    for p in imgs:
        lbl = lbl_dir / (p.stem + ".txt")
        gts = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                v = line.split()
                if len(v) < 5:
                    continue
                c, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
                img = cv2.imread(str(p))
                if img is None:
                    continue
                H, W = img.shape[:2]
                gts.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                            (cx + w / 2) * W, (cy + h / 2) * H])
        items.append((p, np.array(gts, dtype=np.float32).reshape(-1, 5)))
    return items


def locate_coco128(arg: str | None) -> Path:
    """找到（或下载）coco128"""
    if arg:
        return Path(arg)

    import ultralytics
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset("coco128.yaml")
    p = Path(info["path"])
    print(f"数据集   : {p}（ultralytics 自动定位）")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="多个 ONNX，依次评测（FP32 / FP16 / INT8）")
    ap.add_argument("--data", default=None, help="coco128 根目录（不填则自动下载）")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="mAP 评测用低阈值（标准做法：0.001）")
    ap.add_argument("--iou", type=float, default=0.7,
                    help="NMS 的 IoU 阈值（mAP 评测惯例 0.7）")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 张（W6 的快速实验）")
    ap.add_argument("--out", default="map_results.json")
    args = ap.parse_args()

    import onnxruntime as ort
    ort.set_default_logger_severity(3)

    print("=" * 70)
    print("mAP 评测（P3 W4）")
    print("=" * 70)

    root = locate_coco128(args.data)
    items = load_dataset(root, [])
    if args.limit > 0:
        items = items[:args.limit]
    print(f"图片     : {len(items)} 张")

    # 从第一个模型推断类别数
    s = ort.InferenceSession(args.models[0], providers=["CPUExecutionProvider"])
    out_shape = s.get_outputs()[0].shape
    nc = int(out_shape[1]) - 4
    print(f"类别数   : {nc}（从输出 {out_shape} 推断）")

    results = {}
    for mp in args.models:
        name = Path(mp).name
        print(f"\n--- {name} ---")
        sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name

        per_image = {}
        gt_by_img = {}

        for k, (p, gts) in enumerate(items):
            import cv2
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x, scale, px, py = letterbox(rgb, args.size)
            y = sess.run(None, {in_name: x})[0]

            dets = decode_v8(y, nc, args.conf)
            dets = undo_letterbox(dets, scale, px, py, w, h)
            dets = nms(dets, args.iou)

            per_image[k] = dets
            gt_by_img[k] = gts

            if (k + 1) % 32 == 0:
                print(f"  ... {k + 1}/{len(items)}")

        m, nclasses = evaluate(per_image, gt_by_img, nc, iou_thres=0.5)
        results[name] = {"mAP50": m, "classes": nclasses, "images": len(items)}
        print(f"  mAP@0.5 = {m:.4f}  （{nclasses} 个类别有标注）")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print(f"{'模型':<28s} {'mAP@0.5':>10s} {'掉点(pt)':>10s}")
    print("-" * 70)
    base = None
    for name, r in results.items():
        if base is None:
            base = r["mAP50"]
        drop = (base - r["mAP50"]) * 100      # 正数 = 掉了多少
        # 显示成 -1.76：负数代表"比基准低"，和直觉一致
        print(f"{name:<28s} {r['mAP50']:>10.4f} {-drop:>+9.2f}")
    print("=" * 70)
    print("判据（计划 P3 W4）：INT8 掉点 < 1 个百分点")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果已存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
