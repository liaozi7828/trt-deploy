#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_map_coco.py —— 在 COCO val2017（5000 张）上用 **pycocotools 的标准协议**测 mAP

为什么单独写一个而不是改 eval_map.py：
    eval_map.py 用的是 **coco128（128 张）+ 自写的单 IoU(0.5) 101 点插值**。
    那个实现适合"三个模型互相对比掉点"（相对量），但**不能拿去和论文/官方数值比**：
      · COCO 官方指标是 mAP@0.5:0.95（10 个 IoU 阈值平均），不是单个 0.5
      · 匹配规则、面积范围、maxDets 都有明确规定
    所以这里直接用 pycocotools 的 COCOeval —— **指标实现是官方的，不是我写的**。

和 eval_map.py 的一致性（很重要）：
    前处理 / decode / 反 letterbox / NMS **直接 import 自 eval_map.py**，不复制。
    这样"128 张上的结论"和"5000 张上的结论"用的是同一套 C++ 等价逻辑，
    唯一的差别是**数据集和指标实现** —— 变量分离干净。

类别映射（这里最容易错）：
    YOLO 输出 80 类，索引 0..79；COCO 的 category_id 是 1..90（中间有空洞）。
    必须按**类名**映射，不能假设 index+1 == category_id。

用法：
    # 先小规模验证（建议先跑这个，5 分钟）
    python scripts/eval_map_coco.py --models models/yolov8n.onnx --limit 500

    # 全量三方对比（5000 张 × 3 模型）
    python scripts/eval_map_coco.py \
        --models models/yolov8n.onnx models/yolov8n_fp16.onnx models/yolov8n_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_map import letterbox, decode_v8, undo_letterbox, nms  # noqa: E402

# YOLO 的 80 类顺序（和 COCO 的 category_id 不同！必须按名字映射）
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def locate_coco(arg: str | None) -> tuple[Path, Path]:
    """返回 (图片目录, instances_val2017.json 路径)"""
    if arg:
        root = Path(arg)
        img_dir = root / "images" / "val2017"
        if not img_dir.exists():
            img_dir = root / "val2017"
        ann = root / "annotations" / "instances_val2017.json"
        return img_dir, ann

    # 交给 ultralytics 找（它会自动下载 val2017 并转好 labels）
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset("coco.yaml")
    root = Path(info["path"])
    img_dir = Path(info["val"])
    if img_dir.is_file():                       # 新版会给一个 txt 列表
        img_dir = Path(img_dir.read_text().splitlines()[0]).parent
    ann = root / "annotations" / "instances_val2017.json"
    if not ann.exists():
        # ultralytics 转换后可能把原始 json 放在别处
        cand = list(root.rglob("instances_val2017.json"))
        if cand:
            ann = cand[0]
    return img_dir, ann


def build_id_maps(coco) -> dict[str, int]:
    """类名 → COCO category_id"""
    return {c["name"]: c["id"] for c in coco.dataset["categories"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default=None, help="COCO 根目录（不填则用 ultralytics 自动下载）")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="mAP 评测标准阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU（COCO 评测惯例 0.7）")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 张（验证用）")
    ap.add_argument("--out", default="map_coco_results.json")
    ap.add_argument("--base", default=None,
                    help="ΔmAP 的基准模型文件名（默认取列表第一个；"
                         "可指向之前已存进 --out 的模型）")
    args = ap.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("[FAIL] 需要 pycocotools：  pip install pycocotools")
        return 1

    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    import cv2

    img_dir, ann_path = locate_coco(args.data)
    print("=" * 76)
    print("COCO val2017 mAP（pycocotools 标准协议）")
    print("=" * 76)
    print(f"图片目录 : {img_dir}")
    print(f"标注     : {ann_path}")
    if not ann_path.exists():
        print("[FAIL] 找不到 instances_val2017.json")
        return 1

    coco_gt = COCO(str(ann_path))
    name2id = build_id_maps(coco_gt)

    # 取 GT 里出现的图片（val2017 是 5000 张）
    img_ids = sorted(coco_gt.getImgIds())
    if args.limit > 0:
        img_ids = img_ids[:args.limit]
    print(f"图片     : {len(img_ids)} 张")

    # 建 (image_id → 文件路径) 索引（文件名是 12 位零填充）
    id2path = {}
    for iid in img_ids:
        p = img_dir / f"{iid:012d}.jpg"
        if p.exists():
            id2path[iid] = p
    print(f"找到图片 : {len(id2path)} / {len(img_ids)}")
    if len(id2path) < len(img_ids):
        print(f"  [WARN] 缺 {len(img_ids) - len(id2path)} 张，评测会自动跳过缺失的")

    # 类别数从模型输出推断
    s0 = ort.InferenceSession(args.models[0], providers=["CPUExecutionProvider"])
    out_shape = s0.get_outputs()[0].shape
    nc = int(out_shape[1]) - 4
    print(f"类别数   : {nc}")
    if nc != 80:
        print(f"  [WARN] 模型是 {nc} 类，但 COCO 是 80 类 —— 只评估前 {min(nc,80)} 类")
    cat_ids = [name2id.get(n) for n in COCO80_NAMES[:nc]]
    if any(c is None for c in cat_ids):
        missing = [n for n, c in zip(COCO80_NAMES[:nc], cat_ids) if c is None]
        print(f"  [FAIL] 这些类名在 COCO 标注里找不到: {missing[:5]}")
        return 1
    print(f"类别映射 : YOLO[0]={COCO80_NAMES[0]}→id{cat_ids[0]}  "
          f"YOLO[{nc-1}]={COCO80_NAMES[nc-1]}→id{cat_ids[-1]}")

    # ★ 启动时先读已有的 json —— 否则每次 `--out` 都是**覆盖**写，
    #   分批跑（先 FP32，再 INT8，再 FP16）会把前面的结果一个个冲掉。
    #   这就是"分批跑"最容易踩的坑：每次看 json 都对，但只剩最后一次的。
    summary = {}
    if Path(args.out).exists():
        try:
            _prev = json.loads(Path(args.out).read_text(encoding="utf-8"))
            if isinstance(_prev, dict) and _prev:
                summary.update(_prev)
                print(f"已有结果 : {list(_prev)}（会保留并追加）")
        except Exception as e:
            print(f"[WARN] 读不了已有的 {args.out}（{type(e).__name__}），从空开始")
    for mp in args.models:
        name = Path(mp).name
        print(f"\n--- {name} ---")
        sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name

        dets_out = []
        t0 = time.perf_counter()
        n_done = 0
        for iid in img_ids:
            p = id2path.get(iid)
            if p is None:
                continue
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

            # 转成 COCO 结果格式（bbox 是 [x, y, w, h]，原图像素坐标）
            for d in dets:
                c = int(d[0])
                if c >= len(cat_ids):
                    continue
                x1, y1, x2, y2 = float(d[2]), float(d[3]), float(d[4]), float(d[5])
                if x2 <= x1 or y2 <= y1:
                    continue
                dets_out.append({
                    "image_id": int(iid),
                    "category_id": int(cat_ids[c]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(d[1]),
                })
            n_done += 1
            if n_done % 200 == 0:
                el = time.perf_counter() - t0
                eta = el / n_done * (len(img_ids) - n_done)
                print(f"  ... {n_done}/{len(img_ids)}  "
                      f"{el / n_done * 1000:.0f} ms/张  ETA {eta / 60:.1f} 分钟")

        if not dets_out:
            print("  [FAIL] 没有任何预测（conf 阈值太高？）")
            continue

        # ---- pycocotools 官方评测 ----
        coco_dt = coco_gt.loadRes(dets_out)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        # ★★ 评测集必须是**所有真正跑过推理的图**，而不是"有检出的图"。
        #    之前用 {d["image_id"] for d in dets_out} → 某张图在 conf 阈值下一个框都没出时
        #    会被静默排除，导致它的漏检不计入 FN → **mAP 偏高**。
        #    （FP16 那次就少评了 1 张，images=4999 而不是 5000。）
        eval_ids = sorted(id2path.keys())
        n_no_det = len(set(eval_ids) - {d["image_id"] for d in dets_out})
        ev.params.imgIds = eval_ids
        if n_no_det:
            print(f"  注意: {n_no_det} 张图在 conf={args.conf} 下无检出，"
                  f"仍计入评测（漏检计为 FN）")
        ev.evaluate()
        ev.accumulate()

        # ① 标准 mAP@0.5:0.95
        ev.summarize()
        map_5095 = float(ev.stats[0])
        map_50 = float(ev.stats[1])
        map_75 = float(ev.stats[2])

        # ② 单独算一次 mAP@0.5（把 IoU 阈值钉死在 0.5）
        ev50 = COCOeval(coco_gt, coco_dt, "bbox")
        ev50.params.imgIds = ev.params.imgIds
        ev50.params.iouThrs = np.array([0.5])
        ev50.evaluate()
        ev50.accumulate()
        ev50.summarize()
        map_50_only = float(ev50.stats[0])

        summary[name] = {
            "images": len(ev.params.imgIds),
            "mAP50_95": map_5095,
            "mAP50": map_50_only,
            "mAP75": map_75,
            "dets": len(dets_out),
        }
        # ★ 每评完一个模型就落盘。
        #   之前是全部跑完才写，结果第三个模型加载失败 → 前两个的结果全丢，
        #   只能从日志里手工抄。长任务必须增量保存。
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  （已写入 {args.out}）")
        print(f"  → mAP@0.5:0.95 = {map_5095:.4f}   mAP@0.5 = {map_50_only:.4f}   "
              f"mAP@0.75 = {map_75:.4f}   （{len(dets_out)} 个框）")

    # ---- 汇总 ----
    print("\n" + "=" * 76)
    print(f"{'模型':<26s} {'mAP50-95':>10s} {'mAP50':>9s} {'ΔmAP50(pt)':>12s}")
    print("-" * 76)
    # 基准：--base 指定，否则取列表第一个
    if args.base and args.base in summary:
        base_name = args.base
    else:
        base_name = next(iter(summary))
    base = summary[base_name]["mAP50"]
    print(f"基准: {base_name}  (mAP50={base:.4f})")
    for name, r in summary.items():
        d = (r["mAP50"] - base) * 100
        print(f"{name:<26s} {r['mAP50_95']:>10.4f} {r['mAP50']:>9.4f} {d:>+11.2f}")
    print("=" * 76)
    print("说明：ΔmAP50 以第一个模型为基准，负数 = 比基准低多少个百分点")

    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n结果已存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
