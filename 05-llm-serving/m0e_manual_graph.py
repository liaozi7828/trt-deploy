#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m0e_manual_graph.py —— 自己抓 CUDA Graph（绕开 inductor 的保守检查）

m0d 的结论：`mode="reduce-overhead"` 被 inductor 主动放弃了：

    [__cudagraphs] skipping cudagraphs due to mutated inputs (72 instances)
      ... cache_utils.py:483, in update
          self.cumulative_length.add_(kv_length)

也就是说 inductor 看到「输入张量被原地修改」就不敢捕获。
但这只是**保守检查**，不是 CUDA Graph 的硬限制：
KV cache 的原地写正是 graph 要复用的东西，
只要每次 replay 的地址完全一样，原地写完全合法。

所以本脚本用 `torch.cuda.CUDAGraph` **手动捕获 decode 一步**——
capture 里只有 kernel，replay 时 CPU 只发 1 条命令（外加 2 条小 op 喂输入）。

预测（如果「launch 数量就是全部瓶颈」这个模型成立）：
  A  eager + Static     1851 op × ~9.5 µs ≈ 20 ms   （m0d 实测 20.181）
  C  compile(default)    840 op × ~8.8 µs ≈  7.3 ms （m0d 实测 7.325）
  E  手动 graph            1 launch        ≈  3 ms   ← GPU 地板 2.76 ms

如果 E 真在 3 ms 附近，那就证明：**从 17.6 ms 到 3 ms 的全部差距都是 launch 次数**，
并且——这是重点——**不需要 paged attention 也能拿到这 6 倍**；
paged attention 的价值要重新算（它省的是显存，不是 launch）。

用法：
  ~/.venvs/torch/bin/python -u p5/m0e_manual_graph.py
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


def forward(model, ids, cache, cache_pos, v5: bool):
    kw = {"input_ids": ids, "cache_position": cache_pos}
    if v5:
        kw["cache"] = cache
    else:
        kw["past_key_values"] = cache
    return model(**kw)


def make_static_cache(model, max_len: int):
    from transformers import StaticCache

    sig = inspect.signature(StaticCache.__init__).parameters
    want = {"config": model.config, "max_cache_len": max_len, "max_batch_size": 1}
    kw = {k: v for k, v in want.items() if k in sig}
    cache = StaticCache(**kw)
    # 让 inductor / cudagraph 知道这些 buffer 的地址是稳定的（尽力而为）
    marked = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    try:
                        torch._dynamo.mark_static_address(t)
                        marked += 1
                    except Exception:
                        pass
    print(f"    StaticCache(max_cache_len={max_len})  mark_static_address: {marked} 个张量")
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=32)
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

    w_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    floor = w_mb / 1024 / 334 * 1000
    print(f"模型 {args.model}  权重 {w_mb:.0f} MB  GPU 地板 {floor:.2f} ms")

    base = tok(FILLER, return_tensors="pt").input_ids[0]
    ids = base[:args.prompt_len].reshape(1, -1).cuda()
    L = ids.shape[1]
    max_len = L + args.max_new + 8
    dev = ids.device

    # =====================================================================
    # [A] 参考：eager + StaticCache（同 m0d 的 B 组）
    # =====================================================================
    print(f"\n[A] eager + StaticCache（参考）")
    cache_a = make_static_cache(model, max_len)
    with torch.no_grad():
        out = forward(model, ids, cache_a, torch.arange(L, device=dev), v5)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ref = [nxt.reshape(-1)]
        cache_a = out_cache(out)
        itls_a = []
        for i in range(args.max_new - 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = forward(model, nxt, cache_a, torch.tensor([L + i], device=dev), v5)
            torch.cuda.synchronize()
            itls_a.append((time.perf_counter() - t) * 1000)
            cache_a = out_cache(out)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            ref.append(nxt.reshape(-1))
    ref_tokens = torch.cat(ref).cpu()
    itl_a = statistics.median(itls_a)
    print(f"    ITL med {itl_a:7.3f} ms   ≈ {1000/itl_a:6.1f} tok/s")

    # =====================================================================
    # [E] 手动 CUDA Graph：捕获「一步 decode」
    # =====================================================================
    print(f"\n[E] 手动 torch.cuda.CUDAGraph（捕获一步 decode）")
    ok = False
    try:
        # 1) 热身必须用**另一个** cache：热身会真的写 KV，
        #    用同一个会把 cache 推到 L+3，后面位置就错了。
        print("    热身（用一次性 cache，避免污染真正要捕获的那个）")
        warm = make_static_cache(model, max_len)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        w_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        w_pos = torch.zeros(1, dtype=torch.long, device=dev)
        with torch.cuda.stream(side):
            with torch.no_grad():
                o = forward(model, ids, warm, torch.arange(L, device=dev), v5)
                w_ids.fill_(int(o.logits[:, -1, :].argmax(-1)))
                for i in range(3):
                    w_pos.fill_(L + i)
                    forward(model, w_ids, warm, w_pos, v5)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        del warm
        torch.cuda.empty_cache()

        # 2) 正式的那个 cache：prefill 在 graph 外面做（它只跑一次，捕获没意义）
        cache_e = make_static_cache(model, max_len)
        with torch.no_grad():
            out = forward(model, ids, cache_e, torch.arange(L, device=dev), v5)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cache_e = out_cache(out)

        # 3) 固定的输入 buffer —— 每次 replay 前把新 token 拷进去
        st_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        st_pos = torch.zeros(1, dtype=torch.long, device=dev)

        # 4) 捕获（capture 期间 kernel 不执行，只被录下来，所以 cache 不会被推进）
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                st_out = forward(model, st_ids, cache_e, st_pos, v5)

        # 5) replay 循环
        got: list[torch.Tensor] = []
        itls_e: list[float] = []
        with torch.no_grad():
            for i in range(args.max_new - 1):
                st_ids.copy_(nxt.reshape(1, 1))
                st_pos.fill_(L + i)
                torch.cuda.synchronize()          # 把喂输入的 2 个 launch 排空，不计入
                t = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                itls_e.append((time.perf_counter() - t) * 1000)
                nxt = st_out.logits[:, -1, :].argmax(-1, keepdim=True)
                got.append(nxt.reshape(-1))

        itl_e = statistics.median(itls_e)
        got_tokens = torch.cat(got).cpu()

        # 6) 正确性：和 A 比（同一个 cache 类型、同一份权重，应该逐位一致）
        n = min(len(got_tokens), len(ref_tokens) - 1)
        # A 的 ref_tokens[0] 是 prefill 后的第一个 token，E 的 got_tokens[0] 是第 2 个
        diff = (got_tokens[:n] != ref_tokens[1:1 + n]).nonzero()
        verdict = ("token 完全一致 ✅" if len(diff) == 0
                   else f"第 {int(diff[0])+1} 步起分叉 ⚠️")

        print(f"    ITL med {itl_e:7.3f} ms   ≈ {1000/itl_e:6.1f} tok/s")
        print(f"    {verdict}   加速 {itl_a/itl_e:5.2f}×   距地板 {itl_e/floor:4.1f}×")
        ok = True
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # =====================================================================
    # 汇总
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"  A  eager + StaticCache      {itl_a:7.3f} ms    {1000/itl_a:6.1f} tok/s")
    print(f"  C  compile(default)         7.325 ms（m0d 实测）  136.5 tok/s")
    if ok:
        print(f"  E  手动 CUDA Graph           {itl_e:7.3f} ms    {1000/itl_e:6.1f} tok/s"
              f"   {itl_a/itl_e:.2f}× vs A")
    print(f"  GPU 地板                    {floor:7.3f} ms    {1000/floor:6.1f} tok/s")
    print("=" * 70)
    if ok:
        print("判读：")
        print(f"  E 接近地板（{floor:.1f} ms 的 1.5 倍以内）→ 差距全部来自 launch 次数，")
        print("    且**不需要 paged attention 也能拿到**；paged 的价值要改口成「省显存」。")
        print(f"  E 明显高于地板但有数倍加速 → 还剩 GPU 侧瓶颈（attention 的 P 维、fp16 GEMM 效率）。")
        print(f"  E 失败 → 记下错误类型：D2H/同步类(operation not permitted) 或")
        print(f"    捕获内分配类(out of memory)，两者对应不同的修法。")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
