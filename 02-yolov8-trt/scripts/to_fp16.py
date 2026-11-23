#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
to_fp16.py —— P3：把 ONNX 转成 FP16（替代 polygraphy --fp-to-fp16）

为什么不用 polygraphy：
    `polygraphy convert --fp-to-fp16` 在 YOLO 的 Resize（上采样）节点上
    会插错 Cast，产出的模型在 onnxruntime 里加载直接报：
      Type Error: Type (tensor(float)) of output arg (/model.10/Resize_output_0)
      of node (/model.10/Resize_output_cast0) does not match expected type (tensor(float16))

    TRT 能容忍（所以 engine 建出来了），但 ORT 不能 —— 我们要用 ORT 算 mAP，
    所以必须产出一个"两个引擎都认"的模型。

做法：
    onnxconverter-common 的 float16 转换器，并且把 Resize/Upsample 放进 block list：
    这些算子在 fp16 下容易出问题，让它们保持 fp32、由转换器自动插 Cast。
    keep_io_types=True → 输入输出仍是 float32（和 C++ 侧喂的数据类型一致）。

用法：
    python scripts/to_fp16.py --model models/yolov8n.onnx --out models/yolov8n_fp16.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n.onnx")
    ap.add_argument("--out", default="models/yolov8n_fp16.onnx")
    ap.add_argument("--keep-op-fp32", default="Resize,Upsample,Range,NonMaxSuppression",
                    help="这些算子保持 fp32（逗号分隔）")
    args = ap.parse_args()

    import onnx
    from onnxconverter_common import float16

    src, dst = Path(args.model), Path(args.out)
    if not src.exists():
        print(f"[FAIL] 找不到 {src}")
        return 1

    print("=" * 60)
    print("ONNX → FP16（onnxconverter-common）")
    print("=" * 60)
    print(f"输入   : {src}  ({src.stat().st_size / 1024 / 1024:.1f} MB)")

    m = onnx.load(str(src))
    block = [s.strip() for s in args.keep_op_fp32.split(",") if s.strip()]
    print(f"保持fp32: {block}")

    m16 = float16.convert_float_to_float16(
        m,
        keep_io_types=True,        # 输入输出保持 float32
        op_block_list=block,       # 这些算子不转
    )
    onnx.save(m16, str(dst))
    print(f"输出   : {dst}  ({dst.stat().st_size / 1024 / 1024:.1f} MB)")

    # ---- 立刻验证 ORT 能不能加载 ----
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    try:
        s = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        o = s.get_outputs()[0]
        print(f"\nORT 加载 ✅")
        print(f"  输入: {i.name} {i.shape} {i.type}")
        print(f"  输出: {o.name} {o.shape} {o.type}")
    except Exception as e:
        print(f"\nORT 加载 ❌: {e}")
        print("  尝试把更多算子加进 --keep-op-fp32")
        return 1

    print("""
下一步：
  # 用 TRT 重建 engine（可选，现在的是 polygraphy 版本）
  ./build/bin/yolov8_demo --onnx models/yolov8n_fp16.onnx \\
      --save-engine models/yolov8n_fp16b.engine --image models/bus.ppm

  # 算 mAP
  python scripts/eval_map.py --data ~/datasets/coco128 \\
      --models models/yolov8n.onnx models/yolov8n_fp16.onnx models/yolov8n_int8.onnx
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
