#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_yolo.py —— 导出 YOLOv8 到 ONNX，并打印输出结构（P3 W1 第一步）

用法：
    ~/.venvs/torch/bin/python scripts/export_yolo.py
    ~/.venvs/torch/bin/python scripts/export_yolo.py --model yolov8n.pt --imgsz 640
    ~/.venvs/torch/bin/python scripts/export_yolo.py --random   # 不下载权重，随机初始化

产出：
    models/yolov8n.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--random", action="store_true",
                    help="用 yolov8n.yaml 随机初始化（不下载权重）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        print("[FAIL] 没装 ultralytics，先跑：")
        print("  ~/.venvs/torch/bin/pip install ultralytics -i "
              "https://mirrors.aliyun.com/pypi/simple/")
        return 1

    print("=" * 62)
    print("YOLOv8 → ONNX")
    print("=" * 62)
    print(f"ultralytics: {ultralytics.__version__}")

    spec = "yolov8n.yaml" if args.random else args.model
    print(f"加载: {spec}")
    model = YOLO(spec)          # .yaml → 随机初始化；.pt → 下载/加载权重

    # 导出：nms=False 表示把 decode 留在模型里、把 NMS 留给部署侧做
    path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                        simplify=True, nms=False, dynamic=False)
    print(f"导出: {path}")

    # 如果指定了 --out，挪到目标路径
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(path).replace(args.out)
        path = args.out
    else:
        dest = Path("models") / Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dest.resolve():
            Path(path).replace(dest)
        path = str(dest)

    p = Path(path)
    print(f"文件: {p}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")

    # ---- 检查输出结构 ----
    import onnx
    m = onnx.load(str(p))
    onnx.checker.check_model(m)

    print("\n输入:")
    for t in m.graph.input:
        dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
        print(f"  {t.name}  {dims}")

    print("输出:")
    for t in m.graph.output:
        dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
        print(f"  {t.name}  {dims}")

    # 算一下 8400 从哪来
    anchors = (args.imgsz // 8) ** 2 + (args.imgsz // 16) ** 2 + (args.imgsz // 32) ** 2
    print(f"""
结构说明（这几行是 P3 W1 必须能讲出来的）：

  anchors = {args.imgsz}//8² + {args.imgsz}//16² + {args.imgsz}//32²
          = {args.imgsz//8}² + {args.imgsz//16}² + {args.imgsz//32}²
          = {anchors}

  输出形状 [1, 4+nc, anchors]：
    · 通道在前（这是一级坑）：index(c, i) = c * anchors + i
    · c=0..3   : cx, cy, w, h  ← 已经是 {args.imgsz}×{args.imgsz} 空间的**像素值**
    · c=4..end : 每类分数（导出时已 sigmoid，不需要再做）

  所以 decode 只需要：
    ① 找每列的最大分 + 阈值过滤
    ② cxcywh → xyxy
    ③ 反 letterbox（减 pad、除 scale）
    ④ NMS
""")

    # ---- 顺便做一次参考前向，方便对齐 ----
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox

    lb = LetterBox((args.imgsz, args.imgsz), auto=False, stride=32)
    print("提示：如果你想验证 C++ 的 decode，可以这样生成参考结果——")
    print("""
    from ultralytics import YOLO
    import numpy as np
    model = YOLO('models/yolov8n.pt')
    model.predict('dog.jpg', save=True, conf=0.25, iou=0.45)

    # 拿原始输出
    import torch
    x = torch.from_numpy(prepared_chw[None]).float()   # [1,3,640,640] 0..1
    with torch.no_grad():
        raw = model.model(x)[0].numpy()                # [1, 84, 8400]
    np.save('raw_ref.npy', raw)
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
