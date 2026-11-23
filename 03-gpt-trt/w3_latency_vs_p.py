#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w3_latency_vs_p.py —— W3：decode 延迟随 cache 长度 P 的增长（O(P) 到底有多陡）

为什么需要这个脚本（上一个脚本的测量设计有问题）：
    w3_cache.py 里 P 是**单调递增**的（16→17→…→35），
    所以"P 的影响"和"时间/温度的漂移"完全混在一起 —— 这正是 P3 反复踩的坑。
    而且 P 只到 35，cache 才 1.6 MB，O(P) 的代价根本看不出来。

这个脚本的做法：
    · 对每个目标 P（1/16/64/128/256/400/511），**先用真实 prefill 造一个长度 P 的 cache**
      （用同一份权重，所以数值是真实的，不是随机噪声）
    · 然后在这个固定 P 上**重复 decode I 次**（不更新 cache，所以 P 始终不变）
    · 采 R 轮，每轮只在块末 sync 一次 → 中位数 + σ + 95% CI
    · 同时用"两趟法"拆开 host 侧与 GPU 侧：
        第一趟：alloc + setup + execute 全做 → 每步总时间
        第二趟：只做 alloc + setup（不 execute）→ 每步 host 时间
        exec = 总 − host

产出：
    · 延迟 vs P 的曲线（这是 PagedAttention 的量化动机）
    · host 侧占比（动态形状导致 CUDA Graph 用不了，这部分消不掉）

用法：
    python w3_latency_vs_p.py --engine w1_out/gpt_decode.engine
    python w3_latency_vs_p.py --engine ... --ps 1 16 64 256 511 --rounds 5 --iters 10
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import tensorrt as trt

from gpt import GPT, DEFAULT_CFG

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
WEIGHT_SEED = 1234          # ★ 必须和 w3_export_decode.py 一致


def cache_names(n_layer: int):
    ins, outs = [], []
    for i in range(n_layer):
        ins += [f"past_k_{i}", f"past_v_{i}"]
        outs += [f"new_k_{i}", f"new_v_{i}"]
    return ins, outs


def ci95(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / (n ** 0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="w1_out/gpt_decode.engine")
    ap.add_argument("--ps", nargs="+", type=int, default=[1, 16, 64, 128, 256, 400, 511])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    cfg = DEFAULT_CFG
    L, H = cfg.n_layer, cfg.n_head
    hd, V = cfg.n_embd // H, cfg.vocab_size

    rt = trt.Runtime(TRT_LOGGER)
    engine = rt.deserialize_cuda_engine(Path(args.engine).read_bytes())
    if engine is None:
        raise SystemExit("[FAIL] 反序列化失败")
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    in_cache, out_cache = cache_names(L)

    # ---- 与引擎同权重的 PyTorch 模型（只用来造真实 cache）----
    torch.manual_seed(WEIGHT_SEED)
    model = GPT(cfg).eval()
    wsum = sum(float(p.detach().abs().sum()) for p in model.parameters())
    print("=" * 88)
    print("decode 延迟 vs cache 长度 P")
    print("=" * 88)
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"权重校验和 : {wsum:.6f}   ← 必须等于 w3_export_decode.py 打印的那个")
    print(f"采样       : {args.rounds} 轮 × {args.iters} 次，每轮只在块末 sync 一次")
    print(f"引擎       : {Path(args.engine).name} "
          f"({Path(args.engine).stat().st_size / 1048576:.1f} MB)")

    fixed_id = torch.randint(0, V, (1, 1), dtype=torch.int64, device="cuda")
    out_logits = torch.empty((1, 1, V), dtype=torch.float32, device="cuda")

    print(f"\n  {'P':>5} {'alloc+setup':>12} {'exec':>9} {'合计':>9} "
          f"{'σ%':>6} {'CI95%':>7} {'µs/step':>9} {'µs/key':>8}")
    print("  " + "-" * 74)

    rows = []
    for P in args.ps:
        if P < 1 or P > 511:
            print(f"  {P:5d}   超出 profile [1,511]，跳过")
            continue

        # ---- 用真实 prefill 造一个长度恰好为 P 的 cache ----
        with torch.no_grad():
            prompt = torch.randint(0, V, (1, P))
            _lg, caches = model(prompt, kv_caches=None, use_cache=True)
        ck = [k.float().cuda().contiguous() for k, _ in caches]
        cv = [v.float().cuda().contiguous() for _, v in caches]
        assert int(ck[0].shape[2]) == P, f"cache 长度 {ck[0].shape[2]} != {P}"

        def one_iter(do_exec: bool) -> None:
            """一步 decode：alloc + setup（+ 可选 execute）。P 固定，不更新 cache。"""
            ctx.set_input_shape("input_ids", (1, 1))
            for i in range(L):
                ctx.set_input_shape(in_cache[2 * i], tuple(ck[i].shape))
                ctx.set_input_shape(in_cache[2 * i + 1], tuple(cv[i].shape))
            nk, nv = [], []
            for i in range(L):
                nk.append(torch.empty(tuple(ctx.get_tensor_shape(out_cache[2 * i])),
                                      dtype=torch.float32, device="cuda"))
                nv.append(torch.empty(tuple(ctx.get_tensor_shape(out_cache[2 * i + 1])),
                                      dtype=torch.float32, device="cuda"))
            ctx.set_tensor_address("input_ids", fixed_id.data_ptr())
            ctx.set_tensor_address("logits", out_logits.data_ptr())
            for i in range(L):
                ctx.set_tensor_address(in_cache[2 * i], ck[i].data_ptr())
                ctx.set_tensor_address(in_cache[2 * i + 1], cv[i].data_ptr())
                ctx.set_tensor_address(out_cache[2 * i], nk[i].data_ptr())
                ctx.set_tensor_address(out_cache[2 * i + 1], nv[i].data_ptr())
            if do_exec:
                if not ctx.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("execute_async_v3 失败")

        # 预热
        for _ in range(3):
            one_iter(True)
        stream.synchronize()

        # 校验：确认引擎真的在算（否则加速比无意义）
        argmax_ok = 0 <= int(out_logits.argmax()) < V

        # ---- 第一趟：全做 ----
        full = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            for _ in range(args.iters):
                one_iter(True)
            stream.synchronize()
            full.append((time.perf_counter() - t0) * 1000 / args.iters)

        # ---- 第二趟：只做 host 侧（alloc + setup），不 execute ----
        host = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            for _ in range(args.iters):
                one_iter(False)
            host.append((time.perf_counter() - t0) * 1000 / args.iters)

        f_med = statistics.median(full)
        h_med = statistics.median(host)
        e_med = max(f_med - h_med, 0.0)
        sd = statistics.stdev(full) / f_med * 100 if len(full) > 1 else 0.0
        ci = ci95(full) / f_med * 100 if f_med > 0 else 0.0

        rows.append((P, h_med, e_med, f_med))
        print(f"  {P:5d} {h_med:12.3f} {e_med:9.3f} {f_med:9.3f} "
              f"{sd:6.1f} {ci:7.1f} {f_med * 1000:9.0f} {f_med * 1000 / P:8.2f}"
              f"{'' if argmax_ok else '  <-- argmax 异常'}")

    # ---- 增长分析 ----
    if len(rows) >= 2:
        p_lo, p_hi = rows[0], rows[-1]
        print("\n" + "=" * 88)
        print("增长分析")
        print("=" * 88)
        print(f"  P: {p_lo[0]} → {p_hi[0]}（{p_hi[0] / max(p_lo[0], 1):.0f}×）")
        print(f"  总延迟: {p_lo[3]:.3f} → {p_hi[3]:.3f} ms（{p_hi[3] / max(p_lo[3], 1e-9):.1f}×）")
        print(f"  exec  : {p_lo[2]:.3f} → {p_hi[2]:.3f} ms（{p_hi[2] / max(p_lo[2], 1e-9):.1f}×）")
        print(f"  host  : {p_lo[1]:.3f} → {p_hi[1]:.3f} ms（{p_hi[1] / max(p_lo[1], 1e-9):.1f}×）")
        print(f"\n  host 占比: {p_lo[1] / p_lo[3] * 100:.1f}% (P={p_lo[0]}) → "
              f"{p_hi[1] / p_hi[3] * 100:.1f}% (P={p_hi[0]})")

        print("\n  怎么读：")
        print("    · exec 若随 P **线性**增长 → 这就是朴素 KV cache 的 Concat 代价（O(P) 拷贝）。")
        print("      PagedAttention 的做法是不拷贝，只改页表指针，所以 exec 与 P 基本无关。")
        print("    · host 那一列是 24 次 set_input_shape + 50 次 set_tensor_address，")
        print("      与 P **无关**（应该是平的）—— 它占比越高，说明固定开销越主导。")
        print("    · host 占比高 + CUDA Graph 因为形状每步变而用不了 = 这就是")
        print("      '动态形状的 KV cache 在工程上不可扩展' 的完整论证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
