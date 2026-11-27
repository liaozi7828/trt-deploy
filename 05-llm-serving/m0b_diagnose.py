#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m0b_diagnose.py —— 在跑 M1 之前，先弄清楚 M0 那 17.4 ms 是什么

要回答四个问题：
  [1] 一次 kernel launch 在这台机器上多少钱？（WSL2 的 launch 延迟是原生 Linux 的 2-3 倍）
  [2] 权重流式读一遍到底要多久？（决定"地板"在哪，别拿手册带宽猜）
  [3] 一次 decode forward 到底发了多少个 kernel？（把 17.4 ms 拆成 个数 × 单价）
  [4] 用 CUDA Graph 把同样多的 kernel 打包，单价能降到多少？（这就是 M3/M4 的方向）

每一项都独立 try/except —— 任何一项失败不影响其他项，也不会让脚本崩。
profiler 在 WSL2 里可能拿不到 CUPTI 数据（nsys 已经证明这条路不通），失败就跳过。

用法：
  ~/.venvs/torch/bin/python -u p5/m0b_diagnose.py
  ~/.venvs/torch/bin/python -u p5/m0b_diagnose.py --skip-model   # 只测硬件，不加载模型
"""
from __future__ import annotations

import argparse
import time

import torch

N_LAUNCH = 2000      # 纯 launch 的迭代次数
N_GRAPH = 500        # 塞进 CUDA Graph 的 op 数


def bench(fn, iters: int, warmup: int = 20) -> float:
    """返回每次调用的毫秒数。块内不同步，只在首尾各 sync 一次。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def sec(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--compile", action="store_true", help="额外测 torch.compile 的 ITL（慢，要 1-3 分钟）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用")
        return 1

    print(f"torch {torch.__version__}  python {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    print(f"GPU   {torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    print(f"      {p.multi_processor_count} SM  {p.total_memory/1024**3:.1f} GB")

    # ---------------------------------------------------------------- [1] launch
    sec("[1] 一次 kernel launch 的单价")
    a = torch.zeros(1, device="cuda")
    a1m = torch.zeros(1024 * 1024, device="cuda", dtype=torch.float32)   # 4 MiB
    nbytes_1m = a1m.numel() * a1m.element_size()

    try:
        ms_tiny = bench(lambda: a.add_(1.0), N_LAUNCH)
        print(f"  1 个元素 add_ ×{N_LAUNCH:<6} : {ms_tiny*1e3:8.2f} µs/次")
    except Exception as e:
        ms_tiny = float("nan")
        print(f"  [跳过] 小 kernel: {e}")

    try:
        ms_4mb = bench(lambda: a1m.add_(1.0), 500)
        bw_1m = 2 * nbytes_1m / 1024**3 / (ms_4mb / 1000)    # add_ 读一遍写一遍
        print(f"  {nbytes_1m/1024**2:.0f} MiB add_ ×500  : {ms_4mb*1e3:8.2f} µs/次  "
              f"（{bw_1m:.0f} GB/s，读写各一遍）")
    except Exception as e:
        ms_4mb = float("nan")
        print(f"  [跳过] 4 MB kernel: {e}")

    # ---- 同一个 op 塞进 CUDA Graph 之后 ----
    ms_graph_per_op = float("nan")
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):          # 必须在 side stream 上 warmup，否则捕获失败
            for _ in range(3):
                a.add_(1.0)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(N_GRAPH):
                a.add_(1.0)

        g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        ms_replay = (time.perf_counter() - t0) * 1000.0 / 20
        ms_graph_per_op = ms_replay / N_GRAPH
        print(f"  {N_GRAPH} 个 add 捕获成 1 张 Graph : 重放 {ms_replay:.3f} ms "
              f"→ {ms_graph_per_op*1e3:.2f} µs/op")
        if ms_tiny == ms_tiny:
            print(f"  → 打包后单价降到 1/{ms_tiny/ms_graph_per_op:.0f}，"
                  f"这就是 CUDA Graph 对 decode 的意义")
    except Exception as e:
        print(f"  [跳过] CUDA Graph 捕获失败: {type(e).__name__}: {e}")

    # ---------------------------------------------------- [2] 权重流式下界
    sec("[2] 一次大 GEMM 的等效带宽（决定 decode 的地板）")
    bw_gbs = float("nan")
    try:
        C, N = 896, 524288                      # 896 * 524288 * 2 B ≈ 939.5 MB
        W = torch.randn(C, N, device="cuda", dtype=torch.float16)
        x = torch.randn(1, C, device="cuda", dtype=torch.float16)
        nbytes = W.numel() * W.element_size()
        ms = bench(lambda: torch.mm(x, W), 20)
        bw_gbs = nbytes / 1024**3 / (ms / 1000)
        print(f"  [1,896] @ [896,{N}] fp16 = {nbytes/1024**2:.0f} MB")
        print(f"  单次 {ms:.3f} ms → 等效带宽 {bw_gbs:.0f} GB/s")
        print(f"  → 942 MB 权重读一遍的下界 = {942/1024/bw_gbs*1000:.2f} ms")
        del W
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [跳过] {e}")

    # ------------------------------------------------- [3][4] 模型：kernel 数 + compile
    if args.skip_model:
        print("\n(--skip-model：跳过模型部分)")
        return 0

    sec("[3] 一次 decode forward 发了多少个 kernel")
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v5 = int(transformers.__version__.split(".")[0]) >= 5
    print(f"  transformers {transformers.__version__}  ({'cache=' if v5 else 'past_key_values='})")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.cuda().eval()
    print(f"  attn_implementation: {getattr(model.config, '_attn_implementation', '?')}")

    ids = tok("The quick brown fox", return_tensors="pt").input_ids.cuda()

    def step(cur, cache):
        if v5:
            out = model(input_ids=cur, cache=cache)
        else:
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
        return out

    # 先跑 20 步把 cache 灌到 ~25 token，再测"稳定态"的一步
    with torch.no_grad():
        out = step(ids, None)
        cache = out.past_key_values
        for _ in range(20):
            out = step(torch.argmax(out.logits[:, -1, :], -1, keepdim=True), cache)
            cache = out.past_key_values

        # 一遍稳定的 ITL（20 步）
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out = step(nxt, cache)
            cache = out.past_key_values
            nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        torch.cuda.synchronize()
        itl = (time.perf_counter() - t0) * 1000.0 / 20
    print(f"  稳定态 ITL: {itl:.3f} ms/step")

    n_kernel = None
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = step(nxt, cache)
            torch.cuda.synchronize()
        total = 0
        for e in prof.key_averages():
            # 只数真正跑在 GPU 上的
            if getattr(e, "self_device_time_total", 0) > 0 and e.count > 0:
                total += e.count
        n_kernel = total
        print(f"  这一 step 的 GPU kernel 数: {n_kernel}")
        if n_kernel and itl == itl:
            print(f"  → {itl*1e3:.0f} µs / {n_kernel} 个 kernel = "
                  f"{itl*1e3/n_kernel:.1f} µs/kernel   "
                  f"(上面测的单 op 成本含 Python dispatch，这里是纯 launch 摊薄)")
    except Exception as e:
        print(f"  [跳过] profiler 不可用: {type(e).__name__}: {e}")

    # ------------------------------------------------------------- 拆账
    sec("[*] 把 ITL 拆开")
    if bw_gbs == bw_gbs:
        floor_ms = 942 / 1024 / bw_gbs * 1000
        print(f"  权重下界      : {floor_ms:.2f} ms   （942 MB / {bw_gbs:.0f} GB/s）")
        print(f"  实测 ITL      : {itl:.2f} ms")
        print(f"  → 倍数        : {itl/floor_ms:.1f}×   这部分全是「没在算也没在读」的时间")
    if n_kernel and ms_tiny == ms_tiny:
        print(f"  kernel 数 × 单价: {n_kernel} × {ms_tiny*1e3:.1f} µs = "
              f"{n_kernel*ms_tiny:.2f} ms   （单价含 Python dispatch，是上限）")
    print("\n  如果上面「kernel 数 × 单价」≈ 实测 ITL，那 17.4 ms 的性质就确定了：")
    print("  **它是 Python/launch 开销，不是 GPU 在算。** → M1 的 batching 与 M3 的 CUDA Graph"
          " 都能吃这块。")

    # ------------------------------------------------------- [4] torch.compile
    if args.compile:
        sec("[4] torch.compile 对照（第二个 baseline）")
        try:
            cm = torch.compile(model, mode="default", dynamic=True)
            with torch.no_grad():
                out2 = cm(input_ids=ids, **({"cache": None} if v5 else {"past_key_values": None, "use_cache": True}))
                cache2 = out2.past_key_values
                nxt2 = torch.argmax(out2.logits[:, -1, :], -1, keepdim=True)
                for _ in range(20):          # 让它把各种 shape 都编完
                    out2 = cm(input_ids=nxt2, **({"cache": cache2} if v5 else {"past_key_values": cache2, "use_cache": True}))
                    cache2 = out2.past_key_values
                    nxt2 = torch.argmax(out2.logits[:, -1, :], -1, keepdim=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    out2 = cm(input_ids=nxt2, **({"cache": cache2} if v5 else {"past_key_values": cache2, "use_cache": True}))
                    cache2 = out2.past_key_values
                    nxt2 = torch.argmax(out2.logits[:, -1, :], -1, keepdim=True)
                torch.cuda.synchronize()
                itl_c = (time.perf_counter() - t0) * 1000.0 / 20
            print(f"  compile 后 ITL: {itl_c:.3f} ms  （eager {itl:.3f} → {itl/itl_c:.2f}×）")
            print("  ⚠️ decode 时 KV cache 每步变长 → 形状在变 → inclusive cudagraph 会反复重录，"
                  "所以这里只用 mode='default'")
        except Exception as e:
            print(f"  [跳过] torch.compile 失败: {type(e).__name__}: {e}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
