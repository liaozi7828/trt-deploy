"""M4a：chunked prefill 到底解决什么问题 —— 阻塞 vs 切块

要回答的问题
----------
M1/M2 已经知道 prefill 很贵（m2b 里它占墙钟 56%）。但"贵"有两种：

1. 贵在**总量**（算力 + 每 forward 的固定开销）—— 这部分 chunked prefill 一分钱省不了，
   反而因为切的块多了，固定开销会变多；
2. 贵在**它是一次不可打断的长任务** —— 长 prompt 一来，所有正在 decode 的请求都得等它。

chunked prefill 只针对第二种：把一次长 prefill 切成 C token 一块，每个解码步吃一块，
于是"一次性卡住 60 ms"变成"每步多 8 ms"。

本脚本测的就是这个交换：

- `--rows` 条短请求先跑起来并一直 decode；
- 第 `--arrive` 步时来了 1 条长请求（默认 2048 token）；
- `blocking`：整条 prefill 一次做完（多数框架的默认行为）；
- `chunked` ：切成 `--chunk` token 一块，每个解码步吃一块。

指标：**短请求每步延迟的时序**（中位数 / p95 / 最大 / 尖峰落在第几步）、
长请求的 TTFT、总墙钟 —— 最后一项是 chunked 的代价。

正确性
------
两条路径把长请求生成的前 `--newlong` 个 token 与"整条 prefill 后 decode"的参考
逐位对比。chunked prefill 在数学上必须与整条 prefill 完全等价。
"""
from __future__ import annotations

import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FILLER = "The quick brown fox jumps over the lazy dog. " * 2000


@torch.no_grad()
def prefill(model, ids):
    out = model(input_ids=ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


@torch.no_grad()
def extend(model, cache, ids, base: int):
    """把 ids 追加进已有 cache；base = 这段之前已有多少 token。"""
    am = torch.ones(1, base + ids.shape[1], dtype=torch.long, device="cuda")
    out = model(input_ids=ids, past_key_values=cache, attention_mask=am, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


@torch.no_grad()
def one_step(model, cache, nxt):
    out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].argmax(-1, keepdim=True)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2, help="先跑起来的短请求数")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--arrive", type=int, default=4, help="长请求在第几步到达")
    ap.add_argument("--plen", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--newlong", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda", attn_implementation="eager",
        local_files_only=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    full = tok(FILLER, return_tensors="pt").input_ids[0].tolist()

    short_prompts = [full[r * 17: r * 17 + 64] for r in range(args.rows)]
    long_ids = torch.tensor([full[: args.plen]], device="cuda")
    chunks = [long_ids[:, i:i + args.chunk]
              for i in range(0, args.plen, args.chunk)]

    # ---- 参考：整条 prefill + decode ----
    rc, rl = prefill(model, long_ids)
    ref_tokens, cur = [], rl.argmax(-1, keepdim=True)
    for _ in range(args.newlong):
        ref_tokens.append(cur[0, 0].item())
        rc, cur = one_step(model, rc, cur)
    print(f"长请求参考 token（{args.newlong} 个）: {ref_tokens}")

    ev = lambda: torch.cuda.Event(True)
    summary = {}

    for mode in ("blocking", "chunked"):
        # 短请求先起来
        caches, nxts, short_toks = [], [], []
        for p in short_prompts:
            c, lg = prefill(model, torch.tensor([p], device="cuda"))
            caches.append(c)
            nxts.append(lg.argmax(-1, keepdim=True))
            short_toks.append([])

        long_cache, long_nxt, long_toks = None, None, []
        cpos, ttft_step = 0, None
        lat = []

        t_all0 = ev()
        t_all0.record()
        for step in range(args.steps):
            t0, t1 = ev(), ev()
            t0.record()

            # (1) 长请求的 prefill 进度
            if step >= args.arrive and ttft_step is None:
                if mode == "blocking":
                    long_cache, lg = prefill(model, long_ids)
                    cpos = len(chunks)
                else:
                    if long_cache is None:
                        long_cache, lg = prefill(model, chunks[0])
                    else:
                        long_cache, lg = extend(model, long_cache, chunks[cpos],
                                                cpos * args.chunk)
                    cpos += 1
                if cpos == len(chunks):
                    long_nxt = lg.argmax(-1, keepdim=True)
                    ttft_step = step

            # (2) 三条请求各解码一步（长请求 prefill 完成后也进入 decode）
            for i in range(args.rows):
                caches[i], nxts[i] = one_step(model, caches[i], nxts[i])
                short_toks[i].append(nxts[i][0, 0].item())
            if long_nxt is not None and len(long_toks) < args.newlong:
                long_toks.append(long_nxt[0, 0].item())
                long_cache, long_nxt = one_step(model, long_cache, long_nxt)

            t1.record()
            torch.cuda.synchronize()
            lat.append(t0.elapsed_time(t1))

        t_all1 = ev()
        t_all1.record()
        torch.cuda.synchronize()
        total = t_all0.elapsed_time(t_all1)

        body = lat[2:]
        med = statistics.median(body)
        p95 = sorted(body)[max(0, int(len(body) * 0.95) - 1)]
        spike = body.index(max(body)) + 2
        summary[mode] = dict(med=med, p95=p95, mx=max(body), spike=spike,
                             ttft=ttft_step - args.arrive, total=total,
                             toks=long_toks, short=short_toks)
        print(f"\n[{mode}] 每步延迟（ms，从第 {args.arrive - 1} 步起前 8 步）: "
              + " ".join(f"{v:.0f}" for v in lat[args.arrive - 1: args.arrive + 7]))
        print(f"  中位数 {med:6.1f}   p95 {p95:6.1f}   最大 {max(body):6.1f}   "
              f"尖峰在第 {spike} 步   总墙钟 {total:.0f} ms")
        print(f"  长请求 TTFT 相对到达延迟 = {ttft_step - args.arrive} 步")
        print(f"  长请求 token {long_toks}  {'✅ 与参考一致' if long_toks == ref_tokens else '❌ 不一致'}")
        print(f"  短请求 token {short_toks}")

    b, c = summary["blocking"], summary["chunked"]
    print("\n=== 对比 ===")
    print(f"  尖峰延迟   blocking {b['mx']:6.1f} ms   chunked {c['mx']:6.1f} ms   "
          f"→ 压低 {b['mx'] / c['mx']:.2f}×")
    print(f"  p95        blocking {b['p95']:6.1f} ms   chunked {c['p95']:6.1f} ms")
    print(f"  中位数     blocking {b['med']:6.1f} ms   chunked {c['med']:6.1f} ms   "
          f"→ chunked 多付 {c['med'] / b['med'] - 1:+.0%}")
    print(f"  总墙钟     blocking {b['total']:6.0f} ms   chunked {c['total']:6.0f} ms   "
          f"→ chunked 多付 {c['total'] / b['total'] - 1:+.0%}")
    print(f"  短请求 token 两模式一致: {b['short'] == c['short']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
