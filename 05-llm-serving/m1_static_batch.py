#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m1_static_batch.py —— 静态 batching：把 N 个请求一起 decode

为什么先做这一步（而不是先写 paged attention）：
  m0 的基线里 ITL = 17.4 ms，而 0.5B 的权重只有 942 MB。
  942 MB / 17.4 ms = 54 GB/s —— 只有峰值的百分之十几。
  也就是说每一步的 17 ms 里，绝大部分**不是算，也不是读权重**，而是固定开销
  （500 次左右的 kernel launch + Python dispatch）。

  固定开销的解法只有两个：
    ① 摊薄 —— 一个 batch 里塞更多请求（本脚本）
    ② 消除 —— CUDA Graph（M2 之后）

  所以 M1 要回答的问题是：**ITL 随 batch 涨多快？**
    涨得慢（< 20%）→ 确认是固定开销主导 → batching 的收益接近线性
    涨得快（≈ 线性）→ 说明是带宽/算力主导 → batching 没用，得换思路

口径纪律（沿用 p1/w2_matrix.py）：
  · 所有 batch 值在同一次运行里测，不跨运行比
  · 块内不同步，一次 forward 只 sync 一次
  · 报 median / p95，不报 mean

已知简化（写进 README，别藏）：
  · 所有请求的 prompt 长度相同 → 不需要 left padding，避开了 position_ids 的坑
  · 不提前终止（EOS 不停），所有请求都生成满 max_new
  · 没有 HTTP，没有调度 —— 这是 M1，不是 M3

用法：
  ~/.venvs/torch/bin/python -u m1_static_batch.py
  ~/.venvs/llm/bin/python   -u m1_static_batch.py --model ~/models/qwen0.5b
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

FILLER = "The quick brown fox jumps over the lazy dog. " * 400


def pctl(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def make_cache():
    """transformers 5 用 DynamicCache 对象；返回 None 让模型自己建。"""
    return None


def fwd(model, ids: torch.Tensor, cache, v5: bool):
    """一次前向。同时兼容 transformers 5 的 cache= 和旧版的 past_key_values=。"""
    if v5:
        out = model(input_ids=ids, cache=cache)
    else:
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return out


@torch.no_grad()
def run_batch(model, prompts: torch.Tensor, max_new: int, v5: bool):
    """prompts: [B, L]（所有行等长）。返回 (ttft_ms, itl_list_ms, [B, max_new] tokens)"""
    cache = make_cache()
    cur = prompts
    ttft = float("nan")
    itls: list[float] = []
    out_tokens: list[torch.Tensor] = []

    for step in range(max_new + 1):
        t0 = time.perf_counter()
        out = fwd(model, cur, cache, v5)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        cache = out.past_key_values

        if step == 0:
            ttft = dt                    # 第 0 步是 prefill：B 个 prompt 一起
        else:
            itls.append(dt)              # 之后每步 1 个 token

        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur = nxt
        out_tokens.append(nxt)

        if step == max_new:
            break

    return ttft, itls, out_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--peak-bw", type=float, default=384.0,
                    help="显存带宽 GB/s（手册值，5060 Laptop 8GB GDDR7 128-bit 约 384）")
    ap.add_argument("--out", default="p5/m1_static_batch.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用")
        return 1

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v5 = int(transformers.__version__.split(".")[0]) >= 5
    print(f"transformers: {transformers.__version__}  ({'cache=' if v5 else 'past_key_values='} API)")

    dtype = torch.float16
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.cuda().eval()

    n_param = sum(p.numel() for p in model.parameters())
    w_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"模型: {args.model}  {n_param/1e6:.1f} M  权重 {w_mb:.0f} MB")
    print(f"GPU : {torch.cuda.get_device_name(0)}")

    base = tok(FILLER, return_tensors="pt").input_ids[0]
    L = args.prompt_len
    batches = [int(x) for x in args.batches.split(",")]
    max_b = max(batches)
    # 每请求错开 7 个 token 的起点，让 prompt 内容不同（但等长 → 免 padding）
    need = L + 7 * (max_b - 1)
    if need > base.numel():
        reps = (need + base.numel() - 1) // base.numel()
        orig = base.numel()
        base = base.repeat(reps)
        print(f"[warn] FILLER 只有 {orig} token，需要 {need} → 平铺 {reps} 次到 "
              f"{base.numel()}（计时用，语义重复无影响）")

    def prompts_for(B: int) -> torch.Tensor:
        # 各请求内容不同（错开起点），但长度相同 → 不需要 padding
        rows = [base[i * 7 : i * 7 + L] for i in range(B)]
        return torch.stack(rows).cuda()

    # ---- warmup ----
    print(f"\nwarmup: {args.warmup} 次")
    for _ in range(args.warmup):
        run_batch(model, prompts_for(1), 4, v5)
    torch.cuda.synchronize()

    rows = []
    for B in batches:
        p = prompts_for(B)
        ttfts, itls = [], []
        t_start = time.perf_counter()
        for _ in range(args.iters):
            torch.cuda.reset_peak_memory_stats()
            ttft, it, _ = run_batch(model, p, args.max_new, v5)
            ttfts.append(ttft)
            itls.extend(it)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_start

        total_tok = B * args.max_new * args.iters
        thr = total_tok / wall
        med_itl = statistics.median(itls) if itls else float("nan")
        # 每请求吞吐 = 1/ITL（每步每个请求各出一个 token）
        #   旧版写成 B/ITL —— 那是"总吞吐"，列标题却叫"每请求"，是标签错
        per_req = 1000.0 / med_itl if itls else float("nan")
        eff_bw = w_mb / 1024.0 / (med_itl / 1000.0) if itls else float("nan")

        rows.append({
            "batch": B,
            "ttft_ms_median": round(statistics.median(ttfts), 3),
            "itl_ms_median": round(statistics.median(itls), 3),
            "itl_ms_p95": round(pctl(itls, 0.95), 3),
            "itl_ms_max": round(max(itls), 3),
            "total_tokens_per_s": round(thr, 1),
            "per_request_tok_s": round(per_req, 1),
            "effective_bw_gb_s": round(eff_bw, 1),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        })

    # ---- 报告 ----
    print(f"\nprompt_len={L}  max_new={args.max_new}  iters={args.iters}")
    print(f"{'batch':>5} {'TTFT':>8} {'ITL med':>9} {'ITL p95':>9} "
          f"{'tok/s':>9} {'每请求':>9} {'有效带宽':>10} {'显存':>8}")
    print("-" * 78)
    base_itl = rows[0]["itl_ms_median"] if rows else 1.0
    for r in rows:
        print(f"{r['batch']:>5} {r['ttft_ms_median']:>8.3f} {r['itl_ms_median']:>9.3f} "
              f"{r['itl_ms_p95']:>9.3f} {r['total_tokens_per_s']:>9.1f} "
              f"{r['per_request_tok_s']:>9.1f} {r['effective_bw_gb_s']:>8.1f}GB/s "
              f"{r['peak_mem_mb']:>7.0f}MB")

    if len(rows) >= 2:
        b0, bn = rows[0], rows[-1]
        itl_ratio = bn["itl_ms_median"] / b0["itl_ms_median"]
        thr_ratio = bn["total_tokens_per_s"] / b0["total_tokens_per_s"]
        print(f"\n判定：batch {b0['batch']}→{bn['batch']}")
        print(f"  ITL 涨了 {itl_ratio:.2f}×，吞吐涨了 {thr_ratio:.2f}×"
              f"（理想线性 = {bn['batch']/b0['batch']:.2f}×）")
        if itl_ratio < 1.3:
            print("  → ITL 几乎不涨：**固定开销主导**，batching 摊薄有效 ✅")
        elif itl_ratio < 1.8:
            print("  → ITL 涨得比 batch 慢：混合主导（固定开销 + 带宽）")
        else:
            print("  → ITL 接近线性上涨：带宽/算力主导，batching 摊不动")
        mb = w_mb / 1024.0
        print(f"  理论下界（权重全读一遍）: {mb/args.peak_bw*1000:.2f} ms"
              f"   ← 按峰值 {args.peak_bw} GB/s")
        print(f"                          {mb/334.0*1000:.2f} ms"
              f"   ← 按 m0b 实测 334 GB/s（推荐引用这个）")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model, "n_param": n_param, "weight_mb": round(w_mb, 1),
        "prompt_len": L, "max_new": args.max_new, "iters": args.iters,
        "peak_bw_assumed": args.peak_bw, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
