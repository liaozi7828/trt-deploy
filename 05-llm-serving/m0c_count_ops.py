#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m0c_count_ops.py —— 数一次 forward 到底调了多少个算子

为什么需要它：m0b 里 torch.profiler 在 WSL2 上报
    CUPTI_ERROR_INVALID_DEVICE → 一个 kernel 都数不到
（这是 WSL2 已知限制，compute-sanitizer 和 nsys 也一样死）

但我们要数的是**算子调用次数**，不是 GPU kernel —— 那件事不需要 CUPTI，
用 TorchDispatchMode 在 ATen 分发层拦就行。

⚠️ 一条必须记住的限制：
    TorchDispatchMode 会让 torch.compile 失效（inductor 的融合调用被重新拉回
    Python 层）。所以「数算子」和「测 compile」**不能同时做**：
    本脚本用 mode 数 eager 的算子数，compile 只测时间、用时间比反推等效算子数。

用法：
  ~/.venvs/torch/bin/python -u p5/m0c_count_ops.py --compile
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class OpCounter(TorchDispatchMode):
    """在 ATen 分发层数调用次数。不碰 GPU，不需要 CUPTI。"""

    def __init__(self) -> None:
        self.n = 0
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def time_steps(step, nxt, cache, n: int = 20):
    """返回 (itl_ms, 最后的 token, cache)。块内不同步，首尾各一次。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = step(nxt, cache)
        cache = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n, nxt, cache


def group_of(name: str) -> str:
    """把 1851 个算子按"它在干什么"归类。"""
    if name.endswith((".view.default", ".view_as.default", "_unsafe_view.default",
                      ".transpose.int", ".t.default", ".slice.Tensor",
                      ".unsqueeze.default", ".squeeze.default", ".expand.default",
                      ".permute.default", ".reshape.default")):
        return "形状操作（零计算）"
    if name.endswith((".mm.default", ".addmm.default", ".bmm.default",
                      ".matmul.default", ".linear.default")):
        return "矩阵乘"
    if name.endswith((".mul.Tensor", ".add.Tensor", ".sub.Tensor", ".div.Tensor",
                      ".mul.Scalar", ".add.Scalar", ".rsqrt.default", ".pow.Tensor_Scalar")):
        return "逐元素算术"
    if name.endswith(".cat.default"):
        return "拼接（KV cache / rope）"
    if name.endswith(("_to_copy.default", ".copy_.default", ".to.dtype")):
        return "类型/设备转换"
    if name.endswith(("_softmax.default", ".softmax.int", ".silu.default", ".gelu.default",
                      ".sigmoid.default", ".tanh.default")):
        return "逐元素非线性"
    return "其它"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--compile", action="store_true", help="额外测 compile 的 ITL（慢，1-3 分钟）")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v5 = int(transformers.__version__.split(".")[0]) >= 5
    print(f"torch {torch.__version__}  transformers {transformers.__version__}  "
          f"({'cache=' if v5 else 'past_key_values='})")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.cuda().eval()

    def make_step(m):
        def step(cur, cache):
            if v5:
                return m(input_ids=cur, cache=cache)
            return m(input_ids=cur, past_key_values=cache, use_cache=True)
        return step

    step = make_step(model)
    ids = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        # ---- 预热到稳定态（cache ~25 token）----
        out = step(ids, None)
        cache = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        for _ in range(20):
            out = step(nxt, cache)
            cache = out.past_key_values
            nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)

        # ---- [1] eager 算子数（这里用 mode 是安全的，因为不涉及 compile）----
        with OpCounter() as ce:
            step(nxt, cache)
        n_e = ce.n
        itl_e, nxt2, cache2 = time_steps(step, nxt, cache)

        print(f"\n[1] eager · 一次 decode forward: {n_e} 个算子")
        print(f"    eager ITL: {itl_e:.3f} ms  →  {itl_e*1000/n_e:.2f} µs / 算子")
        print(f"    m0b 里裸循环的 add_ 单价 8.99 µs —— 两者接近说明成本几乎全在 Python 层")

        # ---- 按用途归类 ----
        print("\n    按用途归类:")
        buckets: Counter[str] = Counter()
        for name, c in ce.counts.items():
            buckets[group_of(name)] += c
        for g, c in buckets.most_common():
            print(f"      {c:>5}  ({100*c/n_e:>4.1f}%)  {g}")

        print("\n    出现最多的 10 个:")
        for name, c in ce.counts.most_common(10):
            print(f"      {c:>4} × {name}")

        # ---- [2] prefill 对比 ----
        with OpCounter() as cp:
            step(ids, None)
        print(f"\n[2] eager · 一次 prefill（{ids.shape[1]} token）: {cp.n} 个算子")
        print(f"    prefill / decode 算子数 = {cp.n/n_e:.2f}"
              f"  → decode 的成本是「个数」，不是「每个多大」")

        # ---- [3] compile：只测时间，不数算子 ----
        if args.compile:
            print("\n[3] torch.compile：只测时间，用时间比反推等效算子数")
            print("    ⚠️ 不能套 dispatch mode 数算子 —— 那会把 inductor 的融合打回 Python，")
            print("       实测会看到「算子数 1851→1851、速度 0.98×」，等于没 compile。")
            try:
                cm = torch.compile(model, mode="default", dynamic=True)
                cstep = make_step(cm)
                out_c = cstep(ids, None)
                cache_c = out_c.past_key_values
                nxt_c = torch.argmax(out_c.logits[:, -1, :], -1, keepdim=True)
                for _ in range(20):
                    out_c = cstep(nxt_c, cache_c)
                    cache_c = out_c.past_key_values
                    nxt_c = torch.argmax(out_c.logits[:, -1, :], -1, keepdim=True)

                itl_c, _, _ = time_steps(cstep, nxt_c, cache_c)
                n_eff = n_e * itl_c / itl_e          # 假设每个算子的成本不变
                print(f"    compile ITL: {itl_c:.3f} ms  "
                      f"（eager {itl_e:.3f} → {itl_e/itl_c:.2f}×）")
                print(f"    反推等效算子数: {n_e} × {itl_c:.2f}/{itl_e:.2f} = {n_eff:.0f}")
                print(f"    → 相当于融合掉 {100*(1-n_eff/n_e):.0f}% 的算子")
            except Exception as e:
                print(f"    [跳过] {type(e).__name__}: {e}")

    print("\n结论怎么看：")
    print("  · eager ITL ÷ 算子数 ≈ 裸循环单价（9 µs）→ 时间几乎全是 Python 开销，不是 GPU 在算")
    print("  · prefill / decode 算子数几乎相同 → decode 的成本是「个数」，不是「每个多大」")
    print("  · 所以只有两条出路：减算子数（融合）或绕过 Python（CUDA Graph）")
    print("  · 地板是 2.76 ms（m0b 实测的权重流式时间），剩下的都是可以抢的")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
