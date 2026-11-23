#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_resnet50.py —— 把 ResNet50 导成 ONNX（dynamic batch + dynamic H/W）

对应 P1 W4 的交付物，也是项目① 的第一步。

用法：
    ~/.venvs/torch/bin/python export_resnet50.py
    ~/.venvs/torch/bin/python export_resnet50.py --weights DEFAULT   # 用预训练权重

产出：
    models/resnet50.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/resnet50.onnx")
    ap.add_argument("--weights", default="none",
                    help="none = 随机权重（够用）；DEFAULT = 下载预训练权重")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ResNet50 → ONNX")
    print("=" * 60)
    print(f"torch      : {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")

    w = None if args.weights == "none" else args.weights
    try:
        model = torchvision.models.resnet50(weights=w)
    except Exception as e:
        print(f"[WARN] 加载预训练权重失败（{e}），改用随机权重")
        model = torchvision.models.resnet50(weights=None)

    model.eval()   # ★ 必须！否则 BN 会用 batch 统计，导出结果全错
    n_param = sum(p.numel() for p in model.parameters())
    print(f"参数量     : {n_param:,}")

    dummy = torch.randn(1, 3, 224, 224)
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        },
        opset_version=args.opset,
    )

    print("\n导出中...")
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
        how = "legacy (dynamo=False)"
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **kwargs)
        how = "default (dynamo)"
    print(f"导出器     : {how}")
    print(f"文件       : {out}  ({out.stat().st_size/1024/1024:.1f} MB)")

    # ---- 检查 ----
    import onnx
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    print(f"checker    : 通过")
    print(f"节点数     : {len(m.graph.node)}")
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"算子种类   : {ops}")
    for io, label in ((m.graph.input, "input"), (m.graph.output, "output")):
        for t in io:
            dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
            print(f"{label:11s}: {t.name} {dims}")

    # ---- 多形状验证（dynamic 是否真生效）----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    print(f"\n多 shape 验证 (onnxruntime {ort.__version__}):")
    worst = 0.0
    with torch.no_grad():
        for bs, h, w_ in [(1, 224, 224), (4, 224, 224), (1, 320, 320)]:
            x = torch.randn(bs, 3, h, w_)
            y_t = model(x).numpy()
            y_o = sess.run(None, {"input": x.numpy()})[0]
            d = float(abs(y_t - y_o).max())
            worst = max(worst, d)
            print(f"  batch={bs} {h}x{w_}: max|diff| = {d:.3e}")

    print()
    if worst < 1e-4:
        print(f"[OK] 最大误差 {worst:.3e} < 1e-4，动态维度确实生效")
        return 0
    print(f"[FAIL] 最大误差 {worst:.3e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
