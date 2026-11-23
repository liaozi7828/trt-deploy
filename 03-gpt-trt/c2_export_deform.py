#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c2_export_deform.py —— C2：把 Deformable Attention 导出 ONNX，并用 trtexec 实测能否建引擎

这一步要回答的**唯一一个问题**：
    TRT 11 内置的 IGridSampleLayer，能不能处理 **grid 是中间张量**
    （来自 sampling_offsets 这个 Linear 的输出），而不是网络输入？

为什么要专门分开测：
    之前 trtexec 那个 PASSED 的测试里，grid 是 `torch.rand` 直接灌进去的常量输入。
    而 Deformable Attention 的真实形态是：
        query → Linear → offsets → +reference_points → grid（中间张量）→ grid_sample
    这两种情况在 TRT 里走的路径可能不同（形状推导、grid 范围校验、optimize 阶段的特化）。

所以导出两个变体：
    变体 A：grid 是网络输入        ← 已知能过（对照组）
    变体 B：完整的 MSDeformAttn    ← 真实形态（待验证）

用法：
    python c2_export_deform.py                      # 导出两个变体并调 trtexec
    python c2_export_deform.py --skip-trt           # 只导出，不调 trtexec
    python c2_export_deform.py --trtexec /path/to/trtexec
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from c1_deform_attn import MSDeformAttn


# ---------------------------------------------------------------------------
class GridAsInput(nn.Module):
    """变体 A：grid 直接是网络输入（对照组，已知 TRT 能过）"""

    def __init__(self, align_corners=True):
        super().__init__()
        self.align_corners = align_corners

    def forward(self, feat, grid):
        return F.grid_sample(feat, grid, mode="bilinear",
                             padding_mode="zeros",
                             align_corners=self.align_corners)


class DeformFlat(nn.Module):
    """变体 B 的包装：把 value 列表摊平成多个输入（ONNX 的 IO 必须是扁平的）"""

    def __init__(self, m: MSDeformAttn, n_levels: int):
        super().__init__()
        self.m = m
        self.n_levels = n_levels

    def forward(self, query, reference, *values):
        return self.m(query, reference, list(values))


# ---------------------------------------------------------------------------
def dump_gridsample_attrs(onnx_path: Path) -> int:
    """打印图里每个 GridSample 节点的属性，并返回节点数。

    这一步很重要：ONNX 的 GridSample 属性 → TRT 的 IGridSampleLayer 属性是一一对应的：
        mode='bilinear'  → InterpolationMode::kLINEAR
        padding_mode='zeros' → SampleMode::kFILL
        align_corners=1  → setAlignCorners(true)
    导出后先肉眼确认这三个值，能在调 trtexec 之前就预判能不能建。
    """
    import onnx

    m = onnx.load(str(onnx_path))
    gs = [n for n in m.graph.node if n.op_type == "GridSample"]
    print(f"      总节点数 {len(m.graph.node)}，其中 GridSample {len(gs)} 个")
    for i, n in enumerate(gs):
        attrs = {}
        for a in n.attribute:
            v = a.i if a.type == onnx.AttributeProto.INT else a.s
            if isinstance(v, bytes):
                v = v.decode()
            attrs[a.name] = v
        print(f"        #{i}: mode={attrs.get('mode')!r} "
              f"padding_mode={attrs.get('padding_mode')!r} "
              f"align_corners={attrs.get('align_corners')}")
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"      算子种类: {ops}")
    if "Range" in ops or "NonZero" in ops or "ScatterND" in ops:
        print(f"      [WARN] 出现了 TRT 不擅长的数据相关算子")
    return len(gs)


def run_trtexec(trtexec: str, onnx_path: Path, engine_path: Path) -> tuple[bool, str]:
    """调 trtexec 建引擎。返回 (是否成功, 尾部输出)"""
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return False, f"[SKIP] 找不到 trtexec: {trtexec}"
    except subprocess.TimeoutExpired:
        return False, "[FAIL] trtexec 超时（600s）"
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-12:])
    return ("PASSED" in out and "FAILED" not in out.split("PASSED")[-1][:200]), tail


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="c_out")
    ap.add_argument("--trtexec", default=os.environ.get("TRTEXEC", "trtexec"))
    ap.add_argument("--skip-trt", action="store_true")
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    B, Nq, C = 1, args.n_queries, args.d_model
    H, L, P = 4, 3, 4
    levels = [(64, 64), (32, 32), (16, 16)]

    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("C2：Deformable Attention 导出 ONNX + trtexec 实测")
    print("=" * 80)
    print(f"query {B}×{Nq}×{C}   levels {levels}   heads={H} points={P}")

    torch.manual_seed(0)
    model = MSDeformAttn(d_model=C, n_heads=H, n_levels=L, n_points=P,
                         align_corners=True).eval()
    with torch.no_grad():
        model.sampling_offsets.weight.normal_(0, 0.05)
        model.sampling_offsets.bias.normal_(0, 0.1)
        model.attention_weights.weight.normal_(0, 0.1)
        model.attention_weights.bias.normal_(0, 0.1)

    # ★★ 显式保存权重 + 打印校验和。
    #    为什么要存：验证引擎输出时必须用**和导出时同一份权重**。
    #    如果验证脚本自己再 GPT/随机初始化一次，就是"两套权重"，输出会
    #    "量级对、数值全错"（这个坑在 KV cache 那部分已经踩过一次，见 README §8.3）。
    wsum = sum(float(p.abs().sum()) for p in model.parameters())
    ckpt = outdir / "deform_attn_ckpt.pt"
    torch.save({"state_dict": model.state_dict(),
                "cfg": dict(d_model=C, n_heads=H, n_levels=L, n_points=P,
                            align_corners=True),
                "levels": levels, "n_queries": Nq}, ckpt)
    print(f"权重校验和 : {wsum:.6f}   → 存到 {ckpt.name}")

    query = torch.randn(B, Nq, C)
    ref = torch.rand(B, Nq, L, 2)
    values = [torch.randn(B, C, h, w) for h, w in levels]
    # 同样的输入也存下来，验证脚本直接用，避免"输入不一致"
    torch.save({"query": query, "reference": ref, "values": values},
               outdir / "deform_attn_inputs.pt")

    results = []

    # ---------------- 变体 A：grid 是输入 ----------------
    print("\n[1/2] 变体 A：grid 是网络输入（对照组）")
    a_path = outdir / "gs_grid_input.onnx"
    feat_a = torch.randn(B, C, 64, 64)
    grid_a = torch.rand(B, Nq, P, 2) * 2 - 1          # 已经是 [-1,1]
    with torch.no_grad():
        torch.onnx.export(
            GridAsInput().eval(), (feat_a, grid_a), str(a_path),
            input_names=["feat", "grid"], output_names=["out"],
            opset_version=17, dynamo=False, do_constant_folding=True)
    print(f"      {a_path} ({a_path.stat().st_size / 1024:.0f} KB)")
    dump_gridsample_attrs(a_path)
    results.append(("A: grid 是输入", a_path))

    # ---------------- 变体 B：完整模块（grid 是中间张量）----------------
    print("\n[2/2] 变体 B：完整 MSDeformAttn（grid 由 Linear 产生）")
    b_path = outdir / "deform_attn.onnx"
    flat = DeformFlat(model, L).eval()
    in_names = ["query", "reference"] + [f"value_{i}" for i in range(L)]
    with torch.no_grad():
        torch.onnx.export(
            flat, (query, ref, *values), str(b_path),
            input_names=in_names, output_names=["out"],
            opset_version=17, dynamo=False, do_constant_folding=True)
    print(f"      {b_path} ({b_path.stat().st_size / 1024:.0f} KB)")
    n_gs = dump_gridsample_attrs(b_path)
    results.append(("B: 完整 Deformable Attention", b_path))

    # ---------------- 精度：ORT ----------------
    print("\n[精度] ONNXRuntime vs PyTorch")
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(b_path), providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"query": query.numpy(), "reference": ref.numpy(),
                                **{f"value_{i}": values[i].numpy() for i in range(L)}})[0]
        with torch.no_grad():
            y_pt = model(query, ref, values).numpy()
        d = float(np.abs(y_ort - y_pt).max())
        scale = max(float(np.abs(y_pt).max()), 1e-9)
        print(f"      max|diff| = {d:.3e}  相对 {d / scale:.3e}  "
              f"{'✅' if d / scale < 1e-4 else '❌'}")
    except Exception as e:
        print(f"      [FAIL] {type(e).__name__}: {str(e)[:150]}")

    # ---------------- trtexec ----------------
    if args.skip_trt:
        print("\n[trtexec] 已跳过（--skip-trt）")
    else:
        exe = args.trtexec
        if not (Path(exe).exists() or shutil.which(exe)):
            print(f"\n[trtexec] 找不到 {exe}，跳过")
            print(f"          用 --trtexec /path/to/trtexec 或设环境变量 TRTEXEC")
        else:
            print("\n" + "=" * 80)
            print("trtexec 实测")
            print("=" * 80)
            for label, onnx_path in results:
                eng = outdir / (onnx_path.stem + ".engine")
                ok, tail = run_trtexec(exe, onnx_path, eng)
                print(f"\n  [{label}]")
                print(f"    {'✅ PASSED' if ok else '❌ FAILED/SKIP'}  → {eng.name}")
                for line in tail.splitlines():
                    if any(k in line for k in
                           ("PASSED", "FAILED", "Error", "error", "unsupported",
                            "Could not", "no registered", "Latency: min")):
                        print(f"    {line.strip()[:110]}")

            print("\n" + "=" * 80)
            print("怎么读：")
            print("  · 变体 A 应该 PASSED（对照组）")
            print("  · **变体 B 才是关键**：如果也 PASSED，说明 TRT 能处理"
                  "\"grid 是中间张量\"，")
            print("    Deformable Attention **不需要写 plugin** —— 这和我的原假设相反。")
            print("  · 如果 B FAILED，把完整报错记下来 —— 那就是 C3 撞墙笔记的第一条。")

    print(f"\n产物在 {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
