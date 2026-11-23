#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c1_deform_attn.py —— C1：Multi-Scale Deformable Attention 的参考实现与对拍

为什么要自己写一遍而不是直接 import：
    这个项目的目的是**部署**。要部署就必须知道这个算子拆开是什么样：
    哪些部分是"规则的张量运算"（TRT 原生支持）、哪些是"带索引的采样"（需要 grid_sample）。
    只有自己写一遍，才能在 C2 里预判哪些形状/属性会卡住 TRT。

关键结论（本文件末尾会验证）：
    Deformable Attention 的"采样"部分 = **grid_sample(align_corners=True, zeros)**
    而 TRT 11 已内置 IGridSampleLayer（kFILL + kLINEAR + alignCorners）
    → **不需要写 plugin**。这一点在 C2 会用 trtexec 实测确认。

本文件两部分：
    ① MSDeformAttn      —— 向量化实现（用 F.grid_sample），是将来导出 ONNX 的那个
    ② naive_deform_attn —— 按公式暴力循环（for head / for level / for point），
                          慢但一眼能看出对不对

为什么要两个：向量化实现里 reshape/permute 的顺序一旦错，结果不会报错、只会"数值不对"。
            暴力版从公式直接翻译，是唯一可信的对照物。

用法：
    python c1_deform_attn.py                  # 对拍 + 性能对比
    python c1_deform_attn.py --align-corners 0  # 试 align_corners=False
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# ① 向量化实现（将来导出 ONNX 的就是这个）
# ===========================================================================
class MSDeformAttn(nn.Module):
    """多尺度可变形注意力（BEVFormer / Deformable DETR 的核心）

    输入：
        query            [B, Nq, C]
        reference_points [B, Nq, n_levels, 2]  ∈ [0,1]  归一化坐标 (x, y)
        value            list of [B, C, H_l, W_l]       多尺度特征图
    输出：
        [B, Nq, C]

    可学习参数：
        sampling_offsets: Linear(C → n_heads * n_levels * n_points * 2)
        attention_weights: Linear(C → n_heads * n_levels * n_points)
        value_proj:        Linear(C → C)     （对 value 做一次投影）
        output_proj:       Linear(C → C)

    形状约定（这和 mmcv 的实现一致，方便对照）：
        offsets 视图 → [B, Nq, n_heads, n_levels, n_points, 2]
        weights 视图 → [B, Nq, n_heads, n_levels, n_points]，在 (n_levels*n_points) 上 softmax
    """

    def __init__(self, d_model=128, n_heads=4, n_levels=3, n_points=4,
                 align_corners=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.align_corners = align_corners

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # 初始化：偏移从 0 开始（Deformable DETR 的做法）
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, reference_points, value):
        B, Nq, C = query.shape
        H, L, P = self.n_heads, self.n_levels, self.n_points

        # ---- 1. 预测采样偏移与注意力权重 ----
        offsets = self.sampling_offsets(query).view(B, Nq, H, L, P, 2)
        attn = self.attention_weights(query).view(B, Nq, H, L * P)
        attn = F.softmax(attn, dim=-1).view(B, Nq, H, L, P)

        # ---- 2. 把偏移加到参考点上，得到绝对采样坐标（归一化到 [0,1]）----
        #    参考点 [B, Nq, L, 2] → [B, Nq, 1, L, 1, 2]
        ref = reference_points.view(B, Nq, 1, L, 1, 2)
        #    偏移按"特征图尺寸"缩放：不同尺度上同样的偏移量代表不同的像素距离
        #    这里用 mmcv 的约定：偏移量归一化到 [-1,1] 后代表图宽/高的比例
        sample_loc = ref + offsets / max(L, 1)      # [B, Nq, H, L, P, 2]
        #    再归一化到 [-1, 1] 给 grid_sample 用
        grid = sample_loc * 2.0 - 1.0               # [B, Nq, H, L, P, 2]

        # ---- 3. 逐尺度做采样 ----
        #
        # ★ 多头的正确语义：**第 h 个头只看自己那 Cs = C/H 个通道**。
        #   grid_sample 是"整个通道维一起采"的，所以必须把 head 塞进 batch 维：
        #       feat [B, C, Hl, Wl] → [B*H, Cs, Hl, Wl]
        #       grid [B, Nq, H, P, 2] → [B*H, Nq, P, 2]
        #   否则会拿 C 个通道的采样结果去和 Cs 维相加 —— 形状不匹配。
        Cs = C // H
        out = query.new_zeros(B, Nq, H, Cs)
        for l in range(L):
            Hl = value[l].shape[2]
            Wl = value[l].shape[3]
            feat = self.value_proj(value[l].flatten(2).transpose(1, 2))   # [B, HW, C]
            feat = feat.view(B, Hl, Wl, C).permute(0, 3, 1, 2)            # [B, C, Hl, Wl]
            feat_h = feat.view(B, H, Cs, Hl, Wl).reshape(B * H, Cs, Hl, Wl)

            # grid: [B, Nq, H, L, P, 2] → 取第 l 层 → [B, H, Nq, P, 2] → [B*H, Nq, P, 2]
            g = grid[:, :, :, l, :, :].permute(0, 2, 1, 3, 4)
            g = g.reshape(B * H, Nq, P, 2).contiguous()

            sampled = F.grid_sample(feat_h, g, mode="bilinear",
                                    padding_mode="zeros",
                                    align_corners=self.align_corners)
            # [B*H, Cs, Nq, P] → [B, Nq, H, Cs, P]
            sampled = sampled.view(B, H, Cs, Nq, P).permute(0, 3, 1, 2, 4)

            w = attn[:, :, :, l, :].unsqueeze(-2)                         # [B, Nq, H, 1, P]
            out = out + (sampled * w).sum(dim=-1)                          # [B, Nq, H, Cs]

        out = out.reshape(B, Nq, C)
        return self.output_proj(out)


# ===========================================================================
# ② 暴力参考实现：从公式直接翻译，慢但可信
# ===========================================================================
def naive_deform_attn(query, reference_points, value, m: MSDeformAttn):
    """按公式逐项展开：for batch / for query / for head / for level / for point。

    完全不用 grid_sample，自己写双线性插值（zeros padding, align_corners=m.align_corners），
    所以它能独立验证 grid_sample 那条路的 reshape/permute 有没有错。
    """
    B, Nq, C = query.shape
    H, L, P = m.n_heads, m.n_levels, m.n_points
    Cs = C // H

    offsets = m.sampling_offsets(query).view(B, Nq, H, L, P, 2)
    attn = m.attention_weights(query).view(B, Nq, H, L * P)
    attn = F.softmax(attn, dim=-1).view(B, Nq, H, L, P, 1)

    # value 投影后的多尺度特征 [B, C, Hl, Wl]
    feats = []
    for l in range(L):
        f = m.value_proj(value[l].flatten(2).transpose(1, 2))   # [B, HW, C]
        f = f.view(B, value[l].shape[2], value[l].shape[3], C).permute(0, 3, 1, 2)
        feats.append(f)

    out = torch.zeros(B, Nq, H, Cs)
    for b in range(B):
        for q in range(Nq):
            for h in range(H):
                acc = torch.zeros(Cs)
                for l in range(L):
                    f = feats[l][b]                        # [C, Hl, Wl]
                    fh = f[h * Cs:(h + 1) * Cs]            # ★ 只取这个头的通道
                    Hl, Wl = fh.shape[1], fh.shape[2]
                    for p in range(P):
                        # 归一化坐标 [0,1] → grid [-1,1]
                        nx = reference_points[b, q, l, 0] + offsets[b, q, h, l, p, 0] / max(L, 1)
                        ny = reference_points[b, q, l, 1] + offsets[b, q, h, l, p, 1] / max(L, 1)
                        gx = nx * 2.0 - 1.0
                        gy = ny * 2.0 - 1.0
                        # grid → 像素坐标
                        if m.align_corners:
                            px = (gx + 1) * 0.5 * (Wl - 1)
                            py = (gy + 1) * 0.5 * (Hl - 1)
                        else:
                            px = ((gx + 1) * Wl - 1) * 0.5
                            py = ((gy + 1) * Hl - 1) * 0.5
                        # 双线性 + zeros 越界
                        x0, y0 = int(math.floor(px)), int(math.floor(py))
                        tx, ty = px - x0, py - y0
                        val = torch.zeros(Cs)
                        for dx, dy, wgt in ((0, 0, (1 - tx) * (1 - ty)),
                                            (1, 0, tx * (1 - ty)),
                                            (0, 1, (1 - tx) * ty),
                                            (1, 1, tx * ty)):
                            xi, yi = x0 + dx, y0 + dy
                            if 0 <= xi < Wl and 0 <= yi < Hl:
                                val = val + fh[:, yi, xi] * wgt
                        acc = acc + val * attn[b, q, h, l, p, 0]
                out[b, q, h] = acc
    return m.output_proj(out.reshape(B, Nq, C))


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-corners", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    B, Nq, C = args.batch, args.queries, args.d_model
    H, L, P = 4, 3, 4
    levels = [(16, 16), (8, 8), (4, 4)]

    m = MSDeformAttn(d_model=C, n_heads=H, n_levels=L, n_points=P,
                     align_corners=bool(args.align_corners)).eval()
    # 给偏移/权重一个非零的随机初值，否则 softmax 后全是均匀的，测不出顺序错误
    with torch.no_grad():
        m.sampling_offsets.weight.normal_(0, 0.05)
        m.sampling_offsets.bias.normal_(0, 0.1)
        m.attention_weights.weight.normal_(0, 0.1)
        m.attention_weights.bias.normal_(0, 0.1)

    query = torch.randn(B, Nq, C)
    ref = torch.rand(B, Nq, L, 2)
    value = [torch.randn(B, C, h, w) for h, w in levels]

    print("=" * 74)
    print("C1：Multi-Scale Deformable Attention 对拍")
    print("=" * 74)
    print(f"query {tuple(query.shape)}  reference {tuple(ref.shape)}  "
          f"levels {levels}")
    print(f"n_heads={H} n_levels={L} n_points={P} "
          f"align_corners={m.align_corners}")

    with torch.no_grad():
        y_vec = m(query, ref, value)
        t0 = time.perf_counter()
        y_naive = naive_deform_attn(query, ref, value, m)
        t_naive = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        for _ in range(20):
            m(query, ref, value)
        t_vec = (time.perf_counter() - t0) * 1000 / 20

    d = float((y_vec - y_naive).abs().max())
    scale = max(float(y_naive.abs().max()), 1e-9)
    print(f"\n输出形状 : {tuple(y_vec.shape)}")
    print(f"max|diff|: {d:.3e}   相对误差: {d / scale:.3e}")
    print(f"耗时     : 向量化 {t_vec:.3f} ms / 暴力 {t_naive:.1f} ms "
          f"（{t_naive / max(t_vec, 1e-6):.0f}×）")

    ok = d / scale < 1e-5
    print(f"\n判定     : {'✅ 一致（向量化的 reshape/permute 没写错）' if ok else '❌ 不一致'}")

    if ok:
        print("""
下一步（C2）：把向量化版本导出 ONNX
    · 这个模块里**没有 data-dependent 控制流**，所有循环都是常量 L/P 的 Python 循环
      → 导出时会被展开，得到 L 个 grid_sample 节点
    · grid_sample 的属性：mode=bilinear, padding_mode=zeros, align_corners=?
      → 对应 TRT 的 kLINEAR + kFILL + alignCorners
    · ★ 关键待验证：TRT 能否处理 **grid 是动态张量**（来自另一个 Linear 的输出）
      而不是常量 —— 这一点必须实测，不能假设""")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
