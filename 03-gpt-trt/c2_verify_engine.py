#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c2_verify_engine.py —— C2 收尾：**验证 engine 的输出**，而不只是验证 ONNX

为什么必须有这一步：
    trtexec 报的 PASSED 只说明"引擎建出来了"。
    而项目①②里遇到四次"引擎能建、输出错"：plugin 与 Graph 互斥、
    半成品发布 69 格数据错、FP16 在 ORT 与 TRT 上行为不同……
    **所以"能建"和"算得对"是两件事，必须分开验。**

本脚本用**显式保存的 state_dict 和输入**，所以不存在"两次随机初始化 = 两套权重"的风险
（那个坑见 README §8.3）。

检查项：
    ① engine 的输出 vs PyTorch，逐元素相对误差
    ② engine 的延迟 vs PyTorch（部署收益）
    ③ engine 连跑两次是否一致（排除未初始化输出）

用法：
    python c2_verify_engine.py --ckpt c_out/deform_attn_ckpt.pt \
                              --inputs c_out/deform_attn_inputs.pt \
                              --engine c_out/deform_attn.engine
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import tensorrt as trt

from c1_deform_attn import MSDeformAttn

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="c_out/deform_attn_ckpt.pt")
    ap.add_argument("--inputs", default="c_out/deform_attn_inputs.pt")
    ap.add_argument("--engine", default="c_out/deform_attn.engine")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    for p in (args.ckpt, args.inputs, args.engine):
        if not Path(p).exists():
            print(f"[FAIL] 找不到 {p}")
            return 1

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["cfg"]
    levels = ck["levels"]

    print("=" * 76)
    print("C2 收尾：验证 engine 输出（不只是 ONNX）")
    print("=" * 76)

    model = MSDeformAttn(**cfg).eval()
    model.load_state_dict(ck["state_dict"])
    wsum = sum(float(p.abs().sum()) for p in model.parameters())
    print(f"权重校验和 : {wsum:.6f}   ← 应等于 c2_export_deform.py 打印的那个")

    inp = torch.load(args.inputs, weights_only=False)
    query, ref, values = inp["query"], inp["reference"], inp["values"]
    print(f"输入       : query {tuple(query.shape)}  reference {tuple(ref.shape)}  "
          f"levels {[tuple(v.shape) for v in values]}")

    with torch.no_grad():
        y_pt = model(query, ref, values)
    print(f"PyTorch 输出: {tuple(y_pt.shape)}  "
          f"max={y_pt.max().item():+.4f}")

    # ---------------- 加载 engine ----------------
    rt = trt.Runtime(TRT_LOGGER)
    engine = rt.deserialize_cuda_engine(Path(args.engine).read_bytes())
    if engine is None:
        print("[FAIL] 反序列化失败")
        return 1
    ctx = engine.create_execution_context()

    in_names, out_names = [], []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_names.append(n)
        else:
            out_names.append(n)
    print(f"\nengine IO  : {len(in_names)} 输入 / {len(out_names)} 输出")
    print(f"  输入: {in_names}")
    print(f"  输出: {out_names}")

    stream = torch.cuda.Stream()
    gpu_in = {"query": query.cuda(), "reference": ref.cuda()}
    for i, v in enumerate(values):
        gpu_in[f"value_{i}"] = v.cuda()

    # 输出缓冲：形状按 engine 推导
    out_shape = tuple(ctx.get_tensor_shape(out_names[0]))
    out_buf = torch.empty(out_shape, dtype=torch.float32, device="cuda")

    def run_once() -> None:
        for n in in_names:
            ctx.set_tensor_address(n, gpu_in[n].data_ptr())
        ctx.set_tensor_address(out_names[0], out_buf.data_ptr())
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 失败")

    run_once()
    stream.synchronize()
    y_trt = out_buf.clone()

    # ---------------- ① 精度 ----------------
    d = float((y_trt - y_pt.cuda()).abs().max())
    scale = max(float(y_pt.abs().max()), 1e-9)
    rel = d / scale
    print("\n" + "-" * 76)
    print("[①] engine vs PyTorch")
    print(f"    max|diff| = {d:.3e}   相对误差 = {rel:.3e}")
    ok = rel < 1e-2          # TRT 默认 TF32，1e-3 量级属正常
    print(f"    {'✅ 通过（TF32 量级）' if ok else '❌ 不通过 —— 引擎算错了'}")

    # ---------------- ② 可重复性 ----------------
    out_buf.zero_()
    run_once()
    stream.synchronize()
    d2 = float((out_buf - y_trt).abs().max())
    print(f"\n[②] engine 连跑两次是否一致: max|diff| = {d2:.3e}  "
          f"{'✅' if d2 == 0.0 else '⚠️ 不一致'}")

    # ---------------- ③ 延迟 ----------------
    print(f"\n[③] 延迟（{args.iters} 次，块末一次 sync）")
    for _ in range(10):
        run_once()
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_once()
    stream.synchronize()
    t_trt = (time.perf_counter() - t0) * 1000 / args.iters

    with torch.no_grad():
        for _ in range(5):
            model(query, ref, values)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(query, ref, values)
        t_pt = (time.perf_counter() - t0) * 1000 / args.iters

    print(f"    PyTorch : {t_pt:8.4f} ms")
    print(f"    TRT     : {t_trt:8.4f} ms   （{t_pt / max(t_trt, 1e-9):.2f}×）")

    print("\n" + "=" * 76)
    if ok and d2 == 0.0:
        print("结论：TRT 11 内置 GridSample **完全够用** —— Deformable Attention 不需要 plugin。")
        print("      → C 的 plugin 部分必须换目标（TRT 真正缺的是 scatter 类算子，见 C3）。")
        return 0
    print("结论：❌ 有问题，不要往下走")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
