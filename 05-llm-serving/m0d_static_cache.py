#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m0d_static_cache.py —— StaticCache + CUDA Graph，测「绕过 Python」到底能省多少

M0/M1 已经证明：一次 decode 的 1851 个算子 × 9.2 µs = 17 ms，
而 GPU 真正读一遍权重只要 2.8 ms。也就是说 **14 ms 是 CPU 在发号施令**。

那么唯一能拿回这 14 ms 的手段就是「让 CPU 不要每一步都发号施令」——
把整个模型捕获成一张 CUDA Graph，之后每步只 replay 一次。

CUDA Graph 的前提是**形状固定**，而带 KV cache 的 decode 每步 P 都在变。
HF 的 `StaticCache` 就是为此而生的：预分配 [batch, max_len] 的 KV，
用 mask 和 cache_position 表达「哪些位置有效」，形状因此恒定为 max_len。

于是本脚本做一个 2×2 的对照（每一格都在同一个进程里交替测）：

                 eager        compile(default)   compile(reduce-overhead)
    Dynamic      A            —                  —
    Static       B            C                  D

  A = M0/M1 的现状（动态 cache，每步 P+1）
  B = StaticCache 但仍是 eager        → 只改形状，不改执行方式（预测：不变）
  C = StaticCache + compile(default)  → 只拿到「算子融合」，还没有 graph
  D = StaticCache + compile(reduce-overhead) → 融合 + CUDA Graph（这就是天花板）

⚠️ 为什么不做「Dynamic + compile」：形状每步都变，inductor 会每步重新编译，
   那不是测量，那是把编译时间当延迟。

⚠️ 正确性也要比：四个变体都是贪心解码，理论上应该吐出同样的 token。
   前 N 个 token 一致才算「同一条计算路径」，否则比的是两件不同的事。

用法：
  ~/.venvs/torch/bin/python -u p5/m0d_static_cache.py
  ~/.venvs/torch/bin/python -u p5/m0d_static_cache.py --skip-compile
"""
from __future__ import annotations

import argparse
import inspect
import statistics
import time

import torch

FILLER = "The quick brown fox jumps over the lazy dog. " * 400


# ---------------------------------------------------------------------------
# 兼容层：transformers 4.x / 5.x 的 API 名字换过
# ---------------------------------------------------------------------------
def set_cache_v5(model) -> bool:
    """返回 True 表示用 v5 的 cache= 关键字。"""
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


def forward(model, ids, cache, cache_pos, v5: bool):
    """统一入口：把 KV cache 和 cache_position 传进去。"""
    kw = {"input_ids": ids, "cache_position": cache_pos}
    if v5:
        kw["cache"] = cache
    else:
        kw["past_key_values"] = cache
    return model(**kw)


def make_static_cache(model, batch: int, max_len: int):
    """按当前版本的签名构造 StaticCache，并回读出实际容量以便核对。"""
    from transformers import StaticCache

    sig = inspect.signature(StaticCache.__init__).parameters
    print(f"    StaticCache 签名: {list(sig)[:8]}{' …' if len(sig) > 8 else ''}")

    want = {"config": model.config, "max_batch_size": batch, "max_cache_len": max_len}
    kw = {k: v for k, v in want.items() if k in sig}
    if len(kw) < len(want):
        # 不同版本的参数名不一样，尽力而为，并把没用上的显式说出来
        alt = {"batch_size": batch, "max_length": max_len, "max_batch": batch}
        for k, v in alt.items():
            if k in sig and k not in kw:
                kw[k] = v
        missed = [k for k in want if k not in kw]
        if missed:
            print(f"    [warn] 这些参数本版本不支持，忽略: {missed}")

    cache = StaticCache(**kw)

    # 回读容量，确认真的按我们要的尺寸分配了（否则后面测的是「默认可能更小」的图）
    for getter in ("get_max_cache_shape", "get_seq_length", "get_max_length"):
        if hasattr(cache, getter):
            try:
                print(f"    {getter}() = {getattr(cache, getter)()}")
            except Exception:
                pass
            break
    return cache


def out_cache(out):
    """v5 输出可能叫 out.cache，v4 叫 out.past_key_values —— 两个都试。"""
    c = getattr(out, "cache", None)
    if c is None:
        c = getattr(out, "past_key_values", None)
    return c


# ---------------------------------------------------------------------------
# 一次「prefill + N 步贪心 decode」，全程不 .item()（否则每步都强制同步）
# ---------------------------------------------------------------------------
def run_decode(fw, prompt_ids, max_new: int, cache=None):
    """返回 (ttft_ms, itl_ms_list, tokens_cpu)。

    ⚠️ 口径：这里**每一步都 synchronize**。原因是 D（CUDA Graph）的 CPU 侧
       只有 1 次 launch（~10 µs），不同步就测不到 GPU 的执行时间。
       代价是失去了步与步之间的流水重叠（真实服务会重叠），所以三个变体
       都比「不同步」略高一点 —— 但它们之间是可比的。
    """
    dev = prompt_ids.device
    L = prompt_ids.shape[1]
    itls: list[float] = []
    toks: list[torch.Tensor] = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fw(prompt_ids, cache, torch.arange(L, device=dev))
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000.0

    cache = out_cache(out)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    toks.append(nxt)

    for i in range(max_new - 1):
        p = L + i
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fw(nxt, cache, torch.tensor([p], device=dev))
        torch.cuda.synchronize()
        itls.append((time.perf_counter() - t) * 1000.0)
        cache = out_cache(out)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(nxt)

    return ttft, itls, torch.cat([t.reshape(-1) for t in toks]).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-cache-len", type=int, default=0,
                    help="0 = prompt_len + max_new + 8")
    ap.add_argument("--skip-compile", action="store_true", help="只测 A/B（快，~30s）")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v5 = set_cache_v5(None)
    print(f"torch {torch.__version__}  transformers {transformers.__version__}  "
          f"({'cache=' if v5 else 'past_key_values='})")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.cuda().eval()

    w_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"模型: {args.model}  权重 {w_mb:.0f} MB  "
          f"显存 {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    base = tok(FILLER, return_tensors="pt").input_ids[0]
    ids = base[:args.prompt_len].reshape(1, -1).cuda()
    L = ids.shape[1]
    max_len = args.max_cache_len or (L + args.max_new + 8)
    print(f"prompt_len={L}  max_new={args.max_new}  max_cache_len={max_len}")

    results: dict[str, dict] = {}
    ref_tokens: torch.Tensor | None = None

    def record(tag: str, fw, note: str) -> None:
        nonlocal ref_tokens
        print(f"\n[{tag}] {note}")
        try:
            cache = None
            if tag in ("B", "C", "D"):
                cache = make_static_cache(model, 1, max_len)
            ttft, itls, toks = run_decode(fw, ids, args.max_new, cache)
            itl = statistics.median(itls) if itls else float("nan")
            thr = 1000.0 / itl if itls else float("nan")
            mem = torch.cuda.memory_allocated() / 1024**2

            same = ""
            if ref_tokens is None:
                ref_tokens = toks
                same = "（参考）"
            else:
                n = min(len(toks), len(ref_tokens))
                diff = (toks[:n] != ref_tokens[:n]).nonzero()
                same = ("token 完全一致 ✅" if len(diff) == 0
                        else f"第 {int(diff[0])+1} 步起分叉 ⚠️")

            print(f"    TTFT {ttft:8.3f} ms   ITL med {itl:7.3f} ms   "
                  f"≈ {thr:7.1f} tok/s")
            print(f"    {same}")
            results[tag] = {"note": note, "ttft_ms": ttft, "itl_ms": itl,
                            "tok_s": thr, "peak_mb": mem}
            del cache
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    [跳过] {type(e).__name__}: {e}")
            results[tag] = {"note": note, "error": f"{type(e).__name__}: {e}"}

    def fw_eager(m, cache):
        return lambda ids_, c_, pos: forward(m, ids_, c_, pos, v5)

    # ---- 热身 ----
    with torch.no_grad():
        run_decode(fw_eager(model, None), ids, 4)
        torch.cuda.synchronize()

        # ---- A：动态 cache（现状）----
        record("A", fw_eager(model, None), "eager + DynamicCache（M0/M1 的现状）")

        # ---- B：StaticCache + eager ----
        record("B", fw_eager(model, None), "eager + StaticCache（只改形状，不改执行）")

        if not args.skip_compile:
            # ---- C：StaticCache + compile(default)：只拿算子融合 ----
            try:
                cm_d = torch.compile(model, mode="default", dynamic=False)
                record("C", fw_eager(cm_d, None), "compile(default) + StaticCache（融合）")
            except Exception as e:
                print(f"\n[C] compile(default) 失败: {type(e).__name__}: {e}")

            # ---- D：StaticCache + compile(reduce-overhead)：融合 + CUDA Graph ----
            try:
                cm_g = torch.compile(model, mode="reduce-overhead", dynamic=False)
                record("D", fw_eager(cm_g, None),
                       "compile(reduce-overhead) + StaticCache（融合 + CUDA Graph）")
            except Exception as e:
                print(f"\n[D] compile(reduce-overhead) 失败: {type(e).__name__}: {e}")

    # ---- 汇总 ----
    print("\n" + "=" * 76)
    print(f"{'':4} {'TTFT':>10} {'ITL':>9} {'tok/s':>9} {'相对A':>8} {'显存':>8}  说明")
    print("-" * 76)
    a_itl = results.get("A", {}).get("itl_ms")
    for tag in ("A", "B", "C", "D"):
        r = results.get(tag)
        if not r:
            continue
        if "error" in r:
            print(f"{tag:4} {'—':>10} {'—':>9} {'—':>9} {'—':>8} {'—':>8}  {r['error'][:40]}")
            continue
        sp = (a_itl / r["itl_ms"]) if a_itl else float("nan")
        print(f"{tag:4} {r['ttft_ms']:>10.3f} {r['itl_ms']:>9.3f} {r['tok_s']:>9.1f} "
              f"{sp:>7.2f}× {r['peak_mb']:>7.0f}M  {r['note']}")
    print("=" * 76)
    print(f"GPU 地板（权重 {w_mb:.0f} MB / 334 GB/s 实测带宽）= {w_mb/1024/334*1000:.2f} ms")
    print("  → 如果 D 的 ITL 接近这个数，说明 CUDA Graph 把 Python 开销吃干净了")
    print("  → 如果 D 只比 C 快一点，说明瓶颈不在 launch 而在 kernel 之间的依赖")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
