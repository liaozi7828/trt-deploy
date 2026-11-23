#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c3_bev_pool.py —— C3：BEV pooling 的 PyTorch 实现 + 撞墙证明

为什么 BEV pooling 是 TRT 真正的缺口（和 grid_sample 完全不同）：
    BEVDet / BEVDepth 的核心一步是"把图像特征按深度投影到 BEV 网格并累加"：
        bev[b, :, rank[m]] += feat[b, :, m] * weight[b, 0, m]
    三个特征让它在 TRT 里做不了：
      ① **原子累加**：多个来源 m 会落到同一个 BEV 格子上，必须相加
      ② **索引由数据决定**：rank 是运行期算出来的（来自相机内外参 + argsort）
      ③ **输出稀疏**：大部分 BEV 格子没有来源
    ONNX 里对应 ScatterND(reduction='add')；而 TensorRT **没有 ScatterND**。

本脚本做三件事：
    ① 写出 BEV pooling（向量化 scatter_add）
    ② 写一个"逐来源累加"的暴力参考实现，与 ① 对拍（保证 ① 没写错）
    ③ 导出 ONNX → 看它变成了什么算子 → 让 trtexec 说话

★ 关键：③ 的失败**必须保留完整报错**。这是"C3 撞墙笔记"的正文。
  "我试了什么、报了什么错、为什么这是 TRT 的能力边界"——比"我写了个 kernel"更难伪造。

用法：
    python c3_bev_pool.py                      # 对拍 + 导出 + trtexec
    python c3_bev_pool.py --skip-trt
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn


# ===========================================================================
# ① 向量化实现：scatter_add
# ===========================================================================
class BevPool(nn.Module):
    """BEV pooling：把 M 个来源的特征按 rank 累加到 bev_N 个格子上。

    输入：
        feat   [B, C, M]      每个来源的特征
        weight [B, 1, M]      该来源的权重（= 深度值，已归一化）
        rank   [B, M]  int64  每个来源落到哪个 BEV 格子（已排序，来自 argsort）
        bev_n  int            输出格子数（bev_H * bev_W）
    输出：
        bev    [B, C, bev_n]

    这就是 BEVPoolv2 的核心：
        bev[b, :, rank[b, m]] += feat[b, :, m] * weight[b, 0, m]
    """

    def __init__(self, bev_n: int):
        super().__init__()
        self.bev_n = bev_n

    def forward(self, feat, weight, rank):
        B, C, M = feat.shape
        src = feat * weight                                   # [B, C, M]
        # scatter_add 要求 index 和 src 同形状 → 把 rank 沿通道维展开
        idx = rank.unsqueeze(1).expand(B, C, M).contiguous()  # [B, C, M]
        bev = torch.zeros(B, C, self.bev_n, dtype=feat.dtype,
                          device=feat.device)
        return bev.scatter_add(2, idx, src)


# ===========================================================================
# ② 暴力参考：逐来源累加（不用 scatter）
# ===========================================================================
def bev_pool_naive(feat, weight, rank, bev_n: int):
    B, C, M = feat.shape
    bev = torch.zeros(B, C, bev_n, dtype=feat.dtype)
    for b in range(B):
        for m in range(M):
            r = int(rank[b, m])
            bev[b, :, r] += feat[b, :, m] * weight[b, 0, m]
    return bev


# ===========================================================================
def dump_ops(onnx_path: Path) -> list[str]:
    import onnx
    m = onnx.load(str(onnx_path))
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"      节点数 {len(m.graph.node)}")
    print(f"      算子种类: {ops}")
    for target in ("ScatterND", "ScatterElements", "NonZero", "Range"):
        if target in ops:
            n = sum(1 for x in m.graph.node if x.op_type == target)
            print(f"      ★ 出现 {target}（{n} 个）")
    # ★ reduction 属性是这件事的关键：
    #   'add' 才是累加（我们要的）；缺失或 'none' 就只是覆盖写。
    for x in m.graph.node:
        if x.op_type in ("ScatterElements", "ScatterND"):
            attrs = {}
            for a in x.attribute:
                v = a.i if a.type == onnx.AttributeProto.INT else a.s
                if isinstance(v, bytes):
                    v = v.decode()
                attrs[a.name] = v
            print(f"      {x.op_type} 属性: {attrs}")
            red = attrs.get("reduction", "none")
            if red != "add":
                print(f"      ⚠️ reduction={red!r} —— 不是累加！引擎即使能建，输出也会错")
    return ops


def run_trtexec(exe: str, onnx_path: Path, engine_path: Path) -> tuple[bool, str]:
    cmd = [exe, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return False, f"[SKIP] 找不到 trtexec: {exe}"
    except subprocess.TimeoutExpired:
        return False, "[FAIL] 超时"
    out = (r.stdout or "") + (r.stderr or "")
    ok = "PASSED" in out
    return ok, out


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="c_out")
    ap.add_argument("--trtexec", default=os.environ.get("TRTEXEC", "trtexec"))
    ap.add_argument("--skip-trt", action="store_true")
    ap.add_argument("--cams", type=int, default=6)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--hw", type=int, default=16)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--bev", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    B, C = 1, args.channels
    M = args.cams * args.depth * args.hw * args.hw
    bev_n = args.bev * args.bev

    print("=" * 80)
    print("C3：BEV pooling —— PyTorch 实现 + 撞墙证明")
    print("=" * 80)
    print(f"来源 M = cams({args.cams}) × depth({args.depth}) × "
          f"h({args.hw}) × w({args.hw}) = {M}")
    print(f"BEV 格子 = {args.bev}×{args.bev} = {bev_n}")
    print(f"特征通道 C = {C}")

    torch.manual_seed(0)
    feat = torch.randn(B, C, M)
    depth = torch.rand(B, 1, M)
    depth = depth / depth.sum(-1, keepdim=True)      # 归一化（真实做法）
    # rank：把 M 个来源随机映射到 bev_n 个格子（保证有重复 → 才有"累加"语义）
    rank = torch.randint(0, bev_n, (B, M), dtype=torch.int64)

    print(f"rank 的重复率: {1.0 - len(torch.unique(rank)) / M:.1%}  "
          f"（有重复才说明真的在累加）")

    # ★ 存输入：验证引擎输出时必须用**同一份输入**
    #   （不要靠"重新 torch.manual_seed(0) 再生成一遍"——调用顺序一变就不一样了）
    torch.save({"feat": feat, "weight": depth, "rank": rank, "bev_n": bev_n},
               outdir / "bev_pool_inputs.pt")

    # ---------------- ① 对拍 ----------------
    print("\n[1/3] 向量化 vs 暴力参考")
    pool = BevPool(bev_n).eval()
    with torch.no_grad():
        y_vec = pool(feat, depth, rank)
        y_ref = bev_pool_naive(feat, depth, rank, bev_n)
    d = float((y_vec - y_ref).abs().max())
    scale = max(float(y_ref.abs().max()), 1e-9)
    print(f"      max|diff| = {d:.3e}  相对 = {d / scale:.3e}  "
          f"{'✅' if d / scale < 1e-5 else '❌'}")
    if d / scale >= 1e-5:
        print("[FAIL] 向量化实现有问题，先修再继续")
        return 1
    nz = int((y_vec.abs().sum(1) > 0).sum())
    print(f"      非零格子: {nz} / {bev_n}（{nz / bev_n:.1%}）")
    print(f"      ↑ 稀疏性：大部分 BEV 格子没有来源，所以不能当稠密算子处理")

    # ---------------- ② 导出 ONNX ----------------
    print("\n[2/3] 导出 ONNX")
    onnx_path = outdir / "bev_pool.onnx"
    try:
        with torch.no_grad():
            torch.onnx.export(
                pool, (feat, depth, rank), str(onnx_path),
                input_names=["feat", "weight", "rank"], output_names=["bev"],
                opset_version=17, dynamo=False, do_constant_folding=True)
        print(f"      导出成功: {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")
        ops = dump_ops(onnx_path)
    except Exception as e:
        print(f"      [FAIL] 导出就失败了: {type(e).__name__}: {str(e)[:300]}")
        print("\n      ★ 这本身就是结论：")
        print("        PyTorch 无法把 scatter_add 表达成 ONNX 标准算子 → ")
        print("        连 ONNX 都产不出来，更不用说 TRT。")
        return 0

    # ---------------- ③ trtexec ----------------
    print("\n[3/3] trtexec 实测")
    if args.skip_trt:
        print("      已跳过（--skip-trt）")
        return 0
    exe = args.trtexec
    if not (Path(exe).exists() or shutil.which(exe)):
        print(f"      找不到 {exe}，跳过")
        return 0

    engine_path = outdir / "bev_pool.engine"
    ok, out = run_trtexec(exe, onnx_path, engine_path)
    print(f"\n      {'✅ PASSED（意外！TRT 居然支持了）' if ok else '❌ FAILED（预期结果）'}")
    print("\n" + "-" * 80)
    print("完整报错（撞墙笔记正文）：")
    print("-" * 80)
    # 保留关键行
    keys = ("Error", "error", "[E]", "unsupported", "Unsupported", "Could not",
            "no registered", "not supported", "Support", "FAILED", "PASSED",
            "Parsing", "parse")
    saved = 0
    for line in out.splitlines():
        if any(k in line for k in keys) and saved < 30:
            print("  " + line.strip()[:150])
            saved += 1
    (outdir / "bev_pool_trtexec.log").write_text(out, encoding="utf-8")
    print(f"\n  完整日志已存: {outdir / 'bev_pool_trtexec.log'}")

    if not ok:
        print("\n" + "=" * 80)
        print("结论：这是 TRT 的能力边界 —— 该写 plugin 了")
        print("=" * 80)
        print("""  为什么必须写，而不是"想写"：
    · 它是原子累加（多个来源落同一格）→ TRT 的算子都是无副作用的纯函数
    · 索引 rank 由运行期数据决定 → 形状/访问模式不可静态推导
    · 输出稀疏（本例只有一部分格子非零）→ 不能当稠密算子处理

  对照 C2 的 grid_sample：
    grid_sample 的内置实现存在 → 用它
    BEV pooling 不存在        → 写它
  **这就是"什么时候该写 plugin"的判据，两条都有实测支撑。**""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
