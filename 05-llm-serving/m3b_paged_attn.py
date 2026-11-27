#!/usr/bin/env python
"""M3b：分页 attention 的 Triton kernel（decode 阶段，一个 query token 一行）。

M3a 证明了一件事：分页存储省显存，但"gather 成连续张量"会把时间还回去（+21%）。
这个文件要把时间拿回来 —— 让 attention 直接按 block 读，不物化完整 KV。

验收顺序（先正确、再快）：
  1. 与 torch 参考路径（gather + SDPA）逐元素比对
  2. 再比时间：gather 路径 vs 分页 kernel

用法：
    python m3b_paged_attn.py --rows 24 --seq 192
"""
from __future__ import annotations

import argparse
import statistics

import torch
import triton
import triton.language as tl


# =============================================================================
# Triton：decode 阶段的分页 attention
# =============================================================================
@triton.jit
def _paged_attn_decode(
    Q, K, V, BT, SEQ, Out,
    s_qb, s_qh,
    s_k_blk, s_k_t, s_k_h,
    s_v_blk, s_v_t, s_v_h,
    s_bt_b,
    s_ob, s_oh,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // HQ
    h = pid % HQ
    kvh = h * HKV // HQ                      # GQA：多个 query 头共享一个 KV 头

    seq = tl.load(SEQ + b)
    nblk = (seq + BS - 1) // BS

    d_off = tl.arange(0, D)
    q = tl.load(Q + b * s_qb + h * s_qh + d_off)          # (D,)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    t_off = tl.arange(0, BS)
    for i in range(0, nblk):
        blk = tl.load(BT + b * s_bt_b + i)
        k = tl.load(K + blk * s_k_blk + t_off[:, None] * s_k_t + kvh * s_k_h + d_off[None, :])
        v = tl.load(V + blk * s_v_blk + t_off[:, None] * s_v_t + kvh * s_v_h + d_off[None, :])

        s = tl.sum(k.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * SM_SCALE
        pos = i * BS + t_off
        s = tl.where(pos < seq, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)

        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    tl.store(Out + b * s_ob + h * s_oh + d_off, (acc / l_i).to(tl.float16))


def paged_attn_triton(q, pool_k, pool_v, bt, seq, block_size: int, scale: float):
    """q: (B, Hq, D) fp16；pool_k/v: (NB, BS, Hkv, D)；bt: (B, maxb) int32；seq: (B,) int32。"""
    B, Hq, D = q.shape
    Hkv = pool_k.shape[2]
    out = torch.empty_like(q)
    grid = (B * Hq,)
    _paged_attn_decode[grid](
        q, pool_k, pool_v, bt, seq, out,
        q.stride(0), q.stride(1),
        pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
        pool_v.stride(0), pool_v.stride(1), pool_v.stride(2),
        bt.stride(0),
        out.stride(0), out.stride(1),
        HQ=Hq, HKV=Hkv, D=D, BS=block_size, SM_SCALE=scale,
        num_warps=1,
    )
    return out


# =============================================================================
# torch 参考：gather 成连续张量再 SDPA（这就是 M3a 走的路）
# =============================================================================
def paged_attn_gather(q, pool_k, pool_v, bt, seq, block_size: int, scale: float, mask=None):
    """gather 成连续张量再 SDPA（M3a 走的路）。不展开 GQA 头，用 enable_gqa。"""
    B, Hq, D = q.shape
    k = pool_k[bt]                                  # (B, maxb, BS, Hkv, D)
    v = pool_v[bt]
    b, maxb, bs, h, d = k.shape
    k = k.reshape(b, maxb * bs, h, d).transpose(1, 2)      # (B, Hkv, M, D)
    v = v.reshape(b, maxb * bs, h, d).transpose(1, 2)
    L = k.shape[2]
    if mask is None:
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(2), k, v, scale=scale, enable_gqa=True)
        return out.squeeze(2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(2), k, v,
        attn_mask=mask[:, None, None, :],
        scale=scale, enable_gqa=True,
    )
    return out.squeeze(2)


def paged_attn_dense(q, k_dense, v_dense, mask, scale: float):
    """对照组：KV 本来就是连续的一整块（m2b / 现状走的路），不 gather、不展开头。"""
    if mask is None:                       # decode：query 在最后一个位置，无需 mask
        return torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(2), k_dense, v_dense, scale=scale, enable_gqa=True).squeeze(2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(2), k_dense, v_dense,
        attn_mask=mask[:, None, None, :],
        scale=scale, enable_gqa=True,
    )
    return out.squeeze(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=14, help="query 头数")
    ap.add_argument("--kv-heads", type=int, default=2, help="KV 头数（GQA）")
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    B, Hq, Hkv, D, BS = args.rows, args.heads, args.kv_heads, args.head_dim, args.block
    seq_len = args.seq
    maxb = (seq_len + BS - 1) // BS

    # 物理池：每行恰好 maxb 块（本测只关心 attention 本身）
    nb = B * maxb
    pool_k = torch.randn(nb, BS, Hkv, D, dtype=torch.float16, device=dev) * 0.1
    pool_v = torch.randn(nb, BS, Hkv, D, dtype=torch.float16, device=dev) * 0.1
    bt = torch.arange(nb, device=dev, dtype=torch.int32).view(B, maxb)
    seq = torch.full((B,), seq_len, dtype=torch.int32, device=dev)
    q = torch.randn(B, Hq, D, dtype=torch.float16, device=dev) * 0.1
    scale = 1.0 / (D ** 0.5)

    # 对照组：把 KV gather 成连续张量（一次性，不计入时间）
    mask = torch.arange(maxb * BS, device=dev)[None, :] < seq[:, None]
    ref = paged_attn_gather(q, pool_k, pool_v, bt, seq, BS, scale, mask)
    k_dense = pool_k[bt].reshape(B, maxb * BS, Hkv, D).transpose(1, 2)
    v_dense = pool_v[bt].reshape(B, maxb * BS, Hkv, D).transpose(1, 2)
    mine = paged_attn_triton(q, pool_k, pool_v, bt, seq, BS, scale)
    diff = (ref.float() - mine.float()).abs()
    print(f"随机输入：max|diff| = {diff.max().item():.3e}   "
          f"相对 = {(diff.max() / ref.float().abs().max()).item():.3e}")
    dref = (paged_attn_dense(q, k_dense, v_dense, mask, scale).float() - mine.float()).abs()
    print(f"vs 连续 dense SDPA：max|diff| = {dref.max().item():.3e}")

    # 边界：故意制造"部分块被 mask"和"长度不一"
    seq2 = torch.tensor([1] * (B // 2) + [seq_len] * (B - B // 2),
                        dtype=torch.int32, device=dev)
    mask2 = torch.arange(maxb * BS, device=dev)[None, :] < seq2[:, None]
    ref2 = paged_attn_gather(q, pool_k, pool_v, bt, seq2, BS, scale, mask2)
    mine2 = paged_attn_triton(q, pool_k, pool_v, bt, seq2, BS, scale)
    d2 = (ref2.float() - mine2.float()).abs()
    print(f"长度不一（1 / {seq_len}）：max|diff| = {d2.max().item():.3e}")

    # ---- 时间：交错测量，每块一次 sync ----
    def block(fn, n):
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        for _ in range(n):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / n

    # 计时用无 mask 版本：所有行都满长、query 在最后一个位置，此时 mask 是 no-op。
    # （带 mask 的 SDPA 会退回 math 后端，把"gather 的代价"这一项整个淹没掉。）
    fns = {"dense SDPA     ": lambda: paged_attn_dense(q, k_dense, v_dense, None, scale),
           "gather + SDPA  ": lambda: paged_attn_gather(q, pool_k, pool_v, bt, seq, BS, scale, None),
           "paged kernel   ": lambda: paged_attn_triton(q, pool_k, pool_v, bt, seq, BS, scale)}
    for f in fns.values():
        for _ in range(20):
            f()
    torch.cuda.synchronize()

    times = {k: [] for k in fns}
    for r in range(6):
        order = list(fns)[r % len(fns):] + list(fns)[:r % len(fns)]      # 轮转起点
        for k in order:
            times[k].append(block(fns[k], args.iters))

    print(f"\nrows={B} heads={Hq} kv_heads={Hkv} D={D} block={BS} seq={seq_len}")
    base = statistics.median(times["gather + SDPA  "])
    tdens = statistics.median(times["dense SDPA     "])
    for k, v in times.items():
        m = statistics.median(v)
        print(f"  {k}  {m * 1000:8.1f} us   vs gather {base / m:5.2f}x   vs dense {tdens / m:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
