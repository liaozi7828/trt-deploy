"""M4b：分页池上的 chunked prefill —— q_len > 1 的分页 attention kernel

为什么需要新 kernel
------------------
M3 的 kernel 只支持 `q_len = 1`（decode）。而 chunked prefill 的本质是
**一次喂 T 个 token 进同一个池子**，所以必须能：

- 处理一段 query（T 个位置，每行的绝对起点不同）；
- 沿 block table 走完这行**已写过的全部块**，不是只走一块；
- 块内做**因果 + 行内长度**双重 mask：第 t 个 query 只能看 `<= start+t`，
  且不能越过这一行自己的有效长度。

做法是 flash 风格的 online softmax：`m_i / l_i / acc` 三个累加器，
每处理一个块就更新一次，所以显存只需 `BM x D` 的寄存器，与序列长度无关。

与 decode 的关系
----------------
同一个池子、同一张 block table：prefill 往里写整段 KV，decode 往里写一个。
两者共用 `seq`/`bt`，只是 `q_len` 不同 —— 这正是"prefill 与 decode 用同一套缓存"
的含义（M3 的限制 #1 就是它没做到这件事）。
"""
from __future__ import annotations

import argparse
import math
import statistics

import torch
import triton
import triton.language as tl

BS_DEFAULT = 16
BM_DEFAULT = 16


@triton.jit
def _paged_prefill(Q, K, V, BT, POS, LEN, Out,
                   sq_b, sq_h, sq_t, sq_d,
                   sk_blk, sk_t, sk_h, sk_d,
                   sv_blk, sv_t, sv_h, sv_d,
                   sb_b, sb_t,
                   so_b, so_h, so_t, so_d,
                   HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                   BS: tl.constexpr, BM: tl.constexpr, T: tl.constexpr,
                   SM_SCALE: tl.constexpr):
    pid = tl.program_id(0)                       # rows * HQ
    pid_m = tl.program_id(1)                     # 每 BM 个 query 一个 program
    b = pid // HQ
    h = pid % HQ
    kvh = h * HKV // HQ

    start = tl.load(POS + b)
    length = tl.load(LEN + b)
    seq = start + T                               # 这一段已经全部写进池子

    offs_m = pid_m * BM + tl.arange(0, BM)        # chunk 内的相对位置
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_t
                + offs_d[None, :] * sq_d, mask=offs_m[:, None] < T, other=0.0)

    # 每个 query 的绝对位置，以及它能看到的最大绝对位置
    abs_m = start + offs_m
    limit = tl.minimum(abs_m, length - 1)

    m_i = tl.full([BM], -1.0e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    nblk = tl.cdiv(seq, BS)
    for i in range(0, nblk):
        blk = tl.load(BT + b * sb_b + i * sb_t)
        offs_n = tl.arange(0, BS)
        kp = i * BS + offs_n                                  # 该块内每个 key 的绝对位置
        good = kp[None, :] <= limit[:, None]                  # 因果 + 行内长度

        k = tl.load(K + blk * sk_blk + offs_n[:, None] * sk_t
                    + kvh * sk_h + offs_d[None, :] * sk_d)    # (BS, D)
        s = tl.dot(q, tl.trans(k)) * SM_SCALE                 # (BM, BS) fp32
        s = tl.where(good, s, -1.0e30)

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(V + blk * sv_blk + offs_n[:, None] * sv_t
                    + kvh * sv_h + offs_d[None, :] * sv_d)    # (BS, D)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(Out + b * so_b + h * so_h + offs_m[:, None] * so_t
             + offs_d[None, :] * so_d, out.to(tl.float16),
             mask=offs_m[:, None] < T)


def paged_prefill_triton(q, pool_k, pool_v, bt, pos, length, block_size,
                         scale, bm: int = BM_DEFAULT):
    """q: (rows, HQ, T, D) fp16；pos/length: (rows,) int64。

    池子里必须**已经写好**这 T 个 token（本函数只做 attention）。
    """
    rows, hq, T, d = q.shape
    hkv = pool_k.shape[2]
    out = torch.empty_like(q)
    grid = (rows * hq, triton.cdiv(T, bm))
    _paged_prefill[grid](
        q, pool_k, pool_v, bt, pos.to(torch.int32), length.to(torch.int32), out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        pool_k.stride(0), pool_k.stride(1), pool_k.stride(2), pool_k.stride(3),
        pool_v.stride(0), pool_v.stride(1), pool_v.stride(2), pool_v.stride(3),
        bt.stride(0), bt.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        HQ=hq, HKV=hkv, D=d, BS=block_size, BM=bm, T=T,
        SM_SCALE=scale, num_warps=2, num_stages=2,
    )
    return out


# ----------------------------------------------------------------------------
def reference_attn(q, k_dense, v_dense, lengths, scale, start=None):
    """dense 参考：k_dense/v_dense 形状 (rows, hkv, L, D)，带因果与长度 mask。

    注意：这里踩过一次坑 —— `lim` 必须是**一维**再 `[:, None]`，
    写成 `pos[None, :][:, None]` 会变成 (1,1,T) 并把 mask 广播成错误的形状。
    """
    rows, hq, T, d = q.shape
    hkv = k_dense.shape[1]
    g = hq // hkv
    kk = k_dense.repeat_interleave(g, dim=1).float()      # (rows, hq, L, D)
    vv = v_dense.repeat_interleave(g, dim=1).float()
    L = kk.shape[2]
    start = torch.zeros(rows, dtype=torch.long, device=q.device) if start is None else start
    abs_pos = start[:, None] + torch.arange(T, device=q.device)[None, :]   # (rows, T)
    keypos = torch.arange(L, device=q.device)[None, :]                      # (1, L)
    mask = (keypos[None, :, :] <= abs_pos[:, :, None]) & \
           (keypos[None, :, :] < lengths[:, None, None].to(q.device))
    s = torch.einsum("rhtd,rhld->rhtl", q.float(), kk) * scale
    s = s.masked_fill(~mask[:, None, :, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("rhtl,rhld->rhtd", p, vv).to(torch.float16)


def _harness_selftest() -> None:
    """先证明参考实现自己是对的：小规模上 dense 向量化 vs 手写循环。"""
    torch.manual_seed(1)
    rows, hq, hkv, T, d = 2, 2, 1, 12, 16
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(rows, hq, T, d, dtype=torch.float16, device="cuda")
    k = torch.randn(rows, hkv, T, d, dtype=torch.float16, device="cuda")
    v = torch.randn(rows, hkv, T, d, dtype=torch.float16, device="cuda")
    lengths = torch.tensor([12, 7], device="cuda")
    vec = reference_attn(q, k, v, lengths, scale).float()
    # 手写循环（rows/hq/T 都很小，慢但绝对直白）
    loop = torch.zeros_like(vec)
    kk = k.repeat_interleave(hq // hkv, dim=1).float()
    vv = v.repeat_interleave(hq // hkv, dim=1).float()
    for r in range(rows):
        for hh in range(hq):
            for t in range(T):
                hi = min(t, int(lengths[r]) - 1)
                if hi < 0:
                    continue
                s = (q[r, hh, t].float() @ kk[r, hh, :hi + 1].T) * scale
                loop[r, hh, t] = torch.softmax(s, -1) @ vv[r, hh, :hi + 1]
    rel = (vec - loop).abs().max() / loop.abs().max()
    print(f"[自检] 向量化参考 vs 手写循环 rel = {rel:.3e} "
          f"{'OK' if rel < 5e-3 else 'FAIL —— 参考实现本身有错'}")
    assert rel < 5e-3, "参考实现错了，先修参考再谈 kernel"


def build_pool(rows, length, hkv, d, block_size, layers=1):
    per_row = triton.cdiv(length, block_size)
    nb = rows * per_row + 4
    pk = torch.zeros(nb, block_size, hkv, d, dtype=torch.float16, device="cuda")
    pv = torch.zeros(nb, block_size, hkv, d, dtype=torch.float16, device="cuda")
    bt = torch.zeros(rows, per_row, dtype=torch.int32, device="cuda")
    for r in range(rows):
        bt[r] = torch.arange(r * per_row, (r + 1) * per_row,
                             dtype=torch.int32, device="cuda")
    return pk, pv, bt


def write_chunk(pk, pv, bt, k, v, start, block_size):
    """k/v: (rows, hkv, T, D)；start: (rows,) 起点的绝对位置。"""
    rows, hkv, T, d = k.shape
    for r in range(rows):
        s = int(start[r])
        p = torch.arange(s, s + T, device=k.device)
        blk = bt[r, p // block_size].long()
        off = p % block_size
        pk[blk, off] = k[r].permute(1, 0, 2)
        pv[blk, off] = v[r].permute(1, 0, 2)


# ----------------------------------------------------------------------------
def test_random() -> bool:
    """随机张量：分页 prefill vs dense 参考。"""
    torch.manual_seed(0)
    rows, hq, hkv, T, d, bs = 3, 4, 2, 48, 64, 16
    scale = 1.0 / math.sqrt(d)
    lengths = torch.tensor([48, 37, 20], device="cuda")
    q = torch.randn(rows, hq, T, d, dtype=torch.float16, device="cuda")
    k = torch.randn(rows, hkv, T, d, dtype=torch.float16, device="cuda")
    v = torch.randn(rows, hkv, T, d, dtype=torch.float16, device="cuda")
    pk, pv, bt = build_pool(rows, T, hkv, d, bs)
    start = torch.zeros(rows, dtype=torch.long, device="cuda")
    write_chunk(pk, pv, bt, k, v, start, bs)
    got = paged_prefill_triton(q, pk, pv, bt, start, lengths, bs, scale)
    ref = reference_attn(q, k, v, lengths, scale)
    # 只比有效位置（每条请求自己的长度以内）
    valid = torch.arange(T, device="cuda")[None, :] < lengths[:, None]
    mask = valid[:, None, :, None].expand_as(got)
    diff = (got.float() - ref.float()).abs()[mask]
    rel = diff.max() / ref.float().abs()[mask].max()
    print(f"[随机] max|diff| = {diff.max():.3e}  相对 {rel:.3e}  "
          f"{'OK' if rel < 5e-3 else 'FAIL'}")
    return bool(rel < 5e-3)


def bench_chunks() -> None:
    """同一个 prompt 切成 1/2/4/8 块，看 prefill 的固定开销。"""
    torch.manual_seed(0)
    rows, hq, hkv, T, d, bs = 1, 14, 2, 1024, 64, 16
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(rows, hq, T, d, dtype=torch.float16, device="cuda")
    k = torch.randn(rows, hkv, T, d, dtype=torch.float16, device="cuda")
    v = torch.randn(rows, hkv, T, d, dtype=torch.float16, device="cuda")
    print(f"\n[分块] T={T}，每块内核耗时（ms）")
    for nchunk in (1, 2, 4, 8):
        c = T // nchunk
        pk, pv, bt = build_pool(rows, T, hkv, d, bs)
        write_chunk(pk, pv, bt, k, v, torch.zeros(rows, dtype=torch.long,
                                                 device="cuda"), bs)
        times = []
        for _ in range(5):
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            for i in range(nchunk):
                st = torch.full((rows,), i * c, dtype=torch.long, device="cuda")
                ln = torch.full((rows,), (i + 1) * c, dtype=torch.long, device="cuda")
                _ = paged_prefill_triton(q[:, :, i * c:(i + 1) * c], pk, pv, bt,
                                         st, ln, bs, scale)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        med = statistics.median(times)
        print(f"  {nchunk} 块 x {c:4d} token: {med:7.3f} ms  "
              f"({T / med / 1000:6.1f} K tok/s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    print("=== M4b: 分页 prefill kernel（q_len > 1）===")
    _harness_selftest()
    ok = test_random()
    if not args.skip_bench:
        bench_chunks()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
