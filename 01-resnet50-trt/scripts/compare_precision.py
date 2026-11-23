#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_precision.py —— 精度对齐：PyTorch vs TensorRT

项目① 的验收标准：
    TRT-FP32 与 PyTorch 的 max|abs diff| < 1e-4

用法：
    # 1) 用 FP32 引擎导出输入和输出
    ./build/bin/resnet50_demo --engine models/resnet50.engine \
        --dump-input dump/in.bin --dump-output dump/out_fp32.bin --iters 5

    # 2) 用 FP16 引擎导出输出（输入完全相同，不重复导）
    ./build/bin/resnet50_demo --engine models/resnet50_fp16.engine \
        --dump-output dump/out_fp16.bin --iters 5

    # 3) 比对
    ~/.venvs/torch/bin/python scripts/compare_precision.py \
        --input dump/in.bin --fp32 dump/out_fp32.bin --fp16 dump/out_fp16.bin
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read_tensor(path: str) -> np.ndarray:
    """读 main.cpp 导出的自描述二进制（int32 ndim + dims + float32 数据）"""
    with open(path, "rb") as f:
        (nd,) = struct.unpack("i", f.read(4))
        dims = struct.unpack(f"{nd}i", f.read(4 * nd))
        n = int(np.prod(dims))
        data = np.frombuffer(f.read(4 * n), dtype=np.float32)
    if data.size != n:
        raise RuntimeError(f"{path} 数据长度不符：期望 {n}，实际 {data.size}")
    return data.reshape(dims).copy()   # copy：脱离只读的 frombuffer


def compare(a: np.ndarray, b: np.ndarray, label: str, atol: float) -> bool:
    """打印误差统计，返回是否通过"""
    assert a.shape == b.shape, f"形状不一致: {a.shape} vs {b.shape}"

    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)
    rel = diff / denom

    cos = float(
        np.dot(a.ravel(), b.ravel())
        / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    )

    max_abs = float(diff.max())
    ok = max_abs < atol

    print(f"  {label}")
    print(f"    max|abs|   = {max_abs:.3e}      {'OK' if ok else 'FAIL'}  (阈值 {atol:.0e})")
    print(f"    max|rel|   = {float(rel.max()):.3e}")
    print(f"    mean|abs|  = {float(diff.mean()):.3e}")
    print(f"    cos 相似度 = {cos:.10f}")
    return ok


def topk(x: np.ndarray, k: int = 5):
    idx = np.argsort(x)[::-1][:k]
    return [(int(i), float(x[i])) for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="--dump-input 导出的输入")
    ap.add_argument("--fp32", required=True, help="FP32 引擎导出的输出")
    ap.add_argument("--fp16", default=None, help="FP16 引擎导出的输出（可选）")
    ap.add_argument("--weights", default="DEFAULT", help="torchvision 权重")
    ap.add_argument("--atol", type=float, default=1e-4, help="FP32 的验收阈值")
    ap.add_argument("--atol16", type=float, default=5e-2,
                    help="FP16 的阈值。理由：logits 量级 ~20，FP16 相对误差 ~1e-3，"
                         "所以绝对误差 ~2e-2，取 5e-2 留余量")
    args = ap.parse_args()

    import torch
    import torchvision

    print("=" * 66)
    print("精度对齐：PyTorch vs TensorRT")
    print("=" * 66)
    print(f"torch      : {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")

    # ---- 读输入 ----
    x = read_tensor(args.input)
    print(f"输入       : {args.input}  形状 {x.shape}  范围 [{x.min():.3f}, {x.max():.3f}]")
    print(f"             （已归一化，直接喂 PyTorch 即可）\n")

    # ---- PyTorch 前向 ----
    model = torchvision.models.resnet50(weights=args.weights).eval()
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).numpy()

    # ---- 读 TRT 输出 ----
    y32 = read_tensor(args.fp32)

    print("误差统计：")
    ok32 = compare(y_torch, y32, "PyTorch  vs  TRT-FP32", args.atol)

    ok16 = True
    y16 = None
    if args.fp16:
        y16 = read_tensor(args.fp16)
        # FP16 的合理阈值：相对误差 ~1e-3，logits 量级 1e2 → 绝对 1e-1
        print()
        ok16 = compare(y_torch, y16, "PyTorch  vs  TRT-FP16", args.atol16)
        print()
        compare(y32, y16, "TRT-FP32 vs  TRT-FP16", args.atol16)

    # ---- 分类结果是否一致 ----
    print("\ntop-5 对比（第 0 张图）：")
    t_torch = topk(y_torch[0])
    t_trt32 = topk(y32[0])
    print(f"  PyTorch   : {[(c, round(v, 3)) for c, v in t_torch]}")
    print(f"  TRT-FP32  : {[(c, round(v, 3)) for c, v in t_trt32]}")
    if y16 is not None:
        t_trt16 = topk(y16[0])
        print(f"  TRT-FP16  : {[(c, round(v, 3)) for c, v in t_trt16]}")

    same32 = [c for c, _ in t_torch] == [c for c, _ in t_trt32]
    print(f"\n  FP32 top-5 与 PyTorch 完全一致: {'是' if same32 else '否'}")
    if y16 is not None:
        same16 = [c for c, _ in t_torch] == [c for c, _ in topk(y16[0])]
        print(f"  FP16 top-5 与 PyTorch 完全一致: {'是' if same16 else '否'}")

    # ---- 结论 ----
    print("\n" + "=" * 66)
    if ok32:
        print(f"[PASS] PyTorch vs TRT-FP32 满足 < {args.atol:.0e}")
    else:
        print(f"[FAIL] PyTorch vs TRT-FP32 超出 {args.atol:.0e}")
    if args.fp16:
        print(f"[{'PASS' if ok16 else 'FAIL'}] PyTorch vs TRT-FP16（阈值 {args.atol16:.0e}）")
    print("=" * 66)
    return 0 if (ok32 and ok16) else 1


if __name__ == "__main__":
    raise SystemExit(main())
