#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m0_baseline.py —— 基线：HF transformers 的朴素 decode（batch = 1）

这是后面所有优化的参照系。没有它，"我把吞吐提高了 5 倍"这句话不成立。

测三个指标（infra 的核心口径，不是"总耗时"）：
  TTFT   time to first token   —— prefill 决定，用户感知"开始响应"
  ITL    inter-token latency   —— decode 决定，用户感知"打字速度"
  吞吐   output tokens / s     —— 系统容量

测量纪律（沿用 p1/w2_matrix.py 那一套）：
  · 每个配置单独 warmup
  · 块内不同步：一次 forward 只 sync 一次，测到的才是这一步的真实耗时
  · 只报 median / p95，不报 mean（tail 才是 infra 关心的）
  · 不跨运行比 —— 所有数字必须来自同一次运行

用法：
  export HF_ENDPOINT=https://hf-mirror.com          # 国内加速（可选）
  ~/.venvs/torch/bin/python m0_baseline.py
  ~/.venvs/torch/bin/python m0_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


FILLER = "The quick brown fox jumps over the lazy dog. " * 80


def pctl(xs: list[float], q: float) -> float:
    """分位数（不依赖 numpy，且不假设 xs 已排序）。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


@torch.no_grad()
def run_one(model, ids: torch.Tensor, max_new: int, eos_id: int):
    """手写 decode 循环（不用 generate），这样才能逐步计时。

    返回 (ttft_ms, itl_list_ms, n_out)
      · 第 0 步是 prefill（输入整个 prompt）→ 这一次的耗时就是 TTFT
      · 之后每步输入 1 个 token → 这些耗时就是 ITL
    """
    ttft = float("nan")
    itls: list[float] = []
    past = None
    cur = ids
    n_out = 0

    for step in range(max_new):
        t0 = time.perf_counter()
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()          # 只 sync 一次，测这一步
        dt = (time.perf_counter() - t0) * 1000.0

        past = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        if step == 0:
            ttft = dt
        else:
            itls.append(dt)

        cur = nxt
        n_out += 1
        if int(nxt) == eos_id:
            break

    return ttft, itls, n_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=128, help="每请求生成多少 token")
    ap.add_argument("--lens", default="16,128,512", help="prompt 长度列表（逗号分隔）")
    ap.add_argument("--iters", type=int, default=5, help="每个配置重复几次")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="p5/m0_baseline.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用")
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16
    print(f"模型      : {args.model}")
    print(f"设备      : {torch.cuda.get_device_name(0)}")
    print(f"dtype     : {dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        # transformers >= 4.56 用 dtype=（torch_dtype= 已弃用）
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.cuda().eval()

    n_param = sum(p.numel() for p in model.parameters())
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"参数量    : {n_param/1e6:.1f} M   权重 {w_bytes/1024**2:.0f} MB")
    print(f"block_size: {getattr(model.config, 'max_position_embeddings', '?')}")

    eos_id = tok.eos_token_id
    lengths = [int(x) for x in args.lens.split(",")]
    full = tok(FILLER, return_tensors="pt").input_ids.cuda()

    # ---- warmup：把 kernel / cudnn / allocator 全部预热，否则第一个数是假的 ----
    print(f"\nwarmup    : {args.warmup} 次（结果丢弃）")
    for _ in range(args.warmup):
        run_one(model, full[:, :32], 8, eos_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for L in lengths:
        if L > full.shape[1]:
            print(f"[SKIP] len={L} 超过 FILLER 能提供的 {full.shape[1]} 个 token")
            continue
        ids = full[:, :L]

        ttfts, all_itl, outs = [], [], []
        t_start = time.perf_counter()
        for _ in range(args.iters):
            ttft, itls, n = run_one(model, ids, args.max_new, eos_id)
            ttfts.append(ttft)
            all_itl.extend(itls)
            outs.append(n)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_start

        total_out = sum(outs)
        rows.append(
            {
                "prompt_len": L,
                "ttft_ms_median": round(statistics.median(ttfts), 3),
                "itl_ms_median": round(statistics.median(all_itl), 3),
                "itl_ms_p95": round(pctl(all_itl, 0.95), 3),
                "out_tokens": total_out,
                "tokens_per_s": round(total_out / wall, 2),
                "iters": args.iters,
            }
        )

        r = rows[-1]
        print(
            f"\nprompt_len = {L}"
            f"\n  TTFT      : {r['ttft_ms_median']:.3f} ms"
            f"\n  ITL       : median {r['itl_ms_median']:.3f} ms   p95 {r['itl_ms_p95']:.3f} ms"
            f"\n  吞吐      : {r['tokens_per_s']:.2f} tok/s（batch=1，{total_out} tok / {wall:.2f} s）"
        )

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n峰值显存  : {peak:.0f} MB")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "n_param": n_param,
        "weight_mb": round(w_bytes / 1024**2, 1),
        "peak_mem_mb": round(peak, 1),
        "batch": 1,
        "max_new": args.max_new,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入    : {out_path}")

    print("""
口径提醒（写进 README 之前先想清楚）：
  · 这是 batch=1、无调度、无分页的**地板**，不是 vLLM 的对手。
  · 它的价值是"后面每一点提升都能对着它算倍数"。
  · 下一步 M1：静态 batch（多请求一起 decode），看吞吐怎么随 batch 涨。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
