#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m0f_graph_batch.py —— CUDA Graph × batching：两个杠杆能不能相乘

m0e 已经把「单请求 decode」从 19.5 ms 压到 **5.27 ms（3.71×）**，
但离 GPU 地板 2.76 ms 还差 1.9 倍。分解是：

    2.76 ms  真读一遍权重（942 MB ÷ 334 GB/s）
  + 2.13 ms  1851 个 kernel 在 GPU 上依次排队的缝隙（1851 × 1.15 µs，m0b 实测单价）
  ≈ 4.9 ms   vs 实测 5.27 ms ✓

也就是说：**CUDA Graph 消掉的是 CPU 侧的 launch 开销，不是 GPU 侧的 kernel 条数。**
那 1851 个 kernel 仍然一个一个在 GPU 上跑。

这带来一个直接的问题：graph 和 batching 能不能相乘？

    eager :  batch 1 = 57 tok/s，batch 8 = 433 tok/s（7.6×，m1 实测）
    graph :  batch 1 = 190 tok/s（m0e 实测）
    ?
    相乘      :  batch 8 ≈ 1500 tok/s？

如果成立，那 M1 的正确做法不是「再优化 kernel」，而是
**为每个 batch size 各捕获一张 graph**（vLLM 就是这么干的：

    "capture CUDA graphs for batch sizes [1, 2, 4, 8, 16, 32, ...]"

——不是捕获一张万能图，而是捕获一叠小图，按当前 batch 挑一张）。

⚠️ 本脚本的已知简化（必须写进 writeup）：
   所有请求用**同一个 prompt**。真实服务里 prompt 不同 → 长度不同 → 需要
   padding 或 continuous batching 才能凑成固定 batch。这里只测「图形状固定时
   batching 的上限」，不代表真实服务能白拿这个数。

用法：
  ~/.venvs/torch/bin/python -u p5/m0f_graph_batch.py
  ~/.venvs/torch/bin/python -u p5/m0f_graph_batch.py --batches 1,4,8
"""
from __future__ import annotations

import argparse
import inspect
import statistics
import time

import torch

FILLER = "The quick brown fox jumps over the lazy dog. " * 400


def out_cache(out):
    c = getattr(out, "cache", None)
    if c is None:
        c = getattr(out, "past_key_values", None)
    return c


def fwd(model, ids, cache, pos, v5: bool):
    kw = {"input_ids": ids, "cache_position": pos}
    if v5:
        kw["cache"] = cache
    else:
        kw["past_key_values"] = cache
    return model(**kw)


def new_cache(model, max_len: int):
    from transformers import StaticCache

    sig = inspect.signature(StaticCache.__init__).parameters
    want = {"config": model.config, "max_cache_len": max_len}
    return StaticCache(**{k: v for k, v in want.items() if k in sig})


def eager_decode(model, ids, max_new, v5):
    """返回 (tokens[B, max_new] cpu, itl_ms, ttft_ms)"""
    B = ids.shape[0]
    L = ids.shape[1]
    dev = ids.device
    cache = new_cache(model, L + max_new + 8)

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fwd(model, ids, cache, torch.arange(L, device=dev), v5)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000

        cache = out_cache(out)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks = [nxt]
        itls = []
        for i in range(max_new - 1):
            pos = torch.full((B,), L + i, dtype=torch.long, device=dev)
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fwd(model, nxt, cache, pos, v5)
            torch.cuda.synchronize()
            itls.append((time.perf_counter() - t) * 1000)
            cache = out_cache(out)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(nxt)
    return torch.cat(toks, dim=1).cpu(), statistics.median(itls), ttft


def graph_decode(model, ids, max_new, v5):
    """捕获一步 decode，replay。返回 (tokens, itl_ms, ttft_ms)"""
    B = ids.shape[0]
    L = ids.shape[1]
    dev = ids.device
    max_len = L + max_new + 8

    # ---- 1) 热身：必须用一次性 cache（热身会真的写 KV）----
    warm = new_cache(model, max_len)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    w_ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
    w_pos = torch.zeros(B, dtype=torch.long, device=dev)
    with torch.cuda.stream(side):
        with torch.no_grad():
            o = fwd(model, ids, warm, torch.arange(L, device=dev), v5)
            w_ids.copy_(o.logits[:, -1, :].argmax(-1, keepdim=True))
            for i in range(3):
                w_pos.fill_(L + i)
                fwd(model, w_ids, warm, w_pos, v5)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    del warm
    torch.cuda.empty_cache()

    # ---- 2) 正式 cache + prefill（graph 外）----
    cache = new_cache(model, max_len)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fwd(model, ids, cache, torch.arange(L, device=dev), v5)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000
        cache = out_cache(out)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)

    # ---- 3) 捕获（capture 期间 kernel 不执行，cache 不会被推进）----
    st_ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
    st_pos = torch.zeros(B, dtype=torch.long, device=dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            st_out = fwd(model, st_ids, cache, st_pos, v5)

    # ---- 4) replay ----
    toks = [nxt]
    itls = []
    with torch.no_grad():
        for i in range(max_new - 1):
            st_ids.copy_(nxt)
            st_pos.fill_(L + i)
            torch.cuda.synchronize()
            t = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            itls.append((time.perf_counter() - t) * 1000)
            nxt = st_out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(nxt)
    return torch.cat(toks, dim=1).cpu(), statistics.median(itls), ttft


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--batches", default="1,2,4,8,16")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v5 = int(transformers.__version__.split(".")[0]) >= 5
    print(f"torch {torch.__version__}  transformers {transformers.__version__}")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.cuda().eval()

    w_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    floor = w_mb / 1024 / 334 * 1000
    print(f"权重 {w_mb:.0f} MB   权重地板 {floor:.2f} ms（batch 1）")

    base = tok(FILLER, return_tensors="pt").input_ids[0]
    one = base[:args.prompt_len].reshape(1, -1).cuda()
    L = one.shape[1]

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows = []
    for B in batches:
        ids = one.repeat(B, 1)
        print(f"\n{'='*72}\nbatch={B}  prompt_len={L}  max_new={args.max_new}")
        try:
            t_a, itl_a, ttft_a = eager_decode(model, ids, args.max_new, v5)
            print(f"  [A eager]  TTFT {ttft_a:7.2f}  ITL {itl_a:7.3f} ms  "
                  f"{B*1000/itl_a:7.1f} tok/s")
        except Exception as e:
            print(f"  [A eager] 失败 {type(e).__name__}: {e}")
            continue
        try:
            t_g, itl_g, ttft_g = graph_decode(model, ids, args.max_new, v5)
            same = torch.equal(t_a, t_g)
            print(f"  [E graph]  TTFT {ttft_g:7.2f}  ITL {itl_g:7.3f} ms  "
                  f"{B*1000/itl_g:7.1f} tok/s   加速 {itl_a/itl_g:5.2f}×")
            print(f"             token {'完全一致 ✅' if same else '不一致 ⚠️'}")
            rows.append((B, itl_a, itl_g, B*1000/itl_a, B*1000/itl_g, same))
        except Exception as e:
            print(f"  [E graph] 失败 {type(e).__name__}: {e}")
            rows.append((B, itl_a, None, B*1000/itl_a, None, None))
        del ids
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"{'batch':>6} {'eager ITL':>10} {'graph ITL':>10} {'eager tok/s':>12} "
          f"{'graph tok/s':>12} {'graph 加速':>10}  token")
    print("-" * 78)
    for B, a, g, ta, tg, same in rows:
        if g is None:
            print(f"{B:>6} {a:>10.3f} {'—':>10} {ta:>12.1f} {'—':>12} {'—':>10}  —")
        else:
            print(f"{B:>6} {a:>10.3f} {g:>10.3f} {ta:>12.1f} {tg:>12.1f} "
                  f"{a/g:>9.2f}×  {'✅' if same else '⚠️'}")
    print("=" * 78)
    if rows and rows[-1][4]:
        print(f"batch {rows[-1][0]}：graph {rows[-1][4]:.0f} tok/s")
    print("对照：m1 的 eager 最好成绩 = 3268 tok/s @ batch 128（显存 74%）")
    print("      本脚本 batch 上限受显存限制，先看曲线形状，不要追绝对峰值")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
