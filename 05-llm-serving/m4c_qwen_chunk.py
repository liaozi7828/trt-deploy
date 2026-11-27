"""M4c：分页池上的 chunked prefill，接进 Qwen2.5，端到端对拍 + 计时。

与 M4a 的区别
-------------
M4a 比的是"整条 prefill 一次做完"vs"切成块"，但两条都走 HF 的 dense
DynamicCache —— 也就是说它测的是**调度策略**，不是**分页**。

M4c 把块真正喂进 M3 的分页池：每来一块就 `write_chunk` 写进池子，
再用 `paged_prefill_triton`（q_len > 1）做 attention。于是：

- prefill 和 decode 共用同一块池子、同一张 block table（M3 的限制 #1 被补上）；
- 可以直接验证"分块 prefill 出来的 token 与整条 prefill 完全一致"；
- 可以量出"把 prefill 切成 c 块"真实的端到端代价（不是只看 kernel）。

验收线：先 token 逐位一致，再看时间。
"""
from __future__ import annotations

import argparse
import statistics
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from m3b_paged_attn import paged_attn_triton
from m3c_qwen_paged import (FILLER, MODEL, build_pool, decode, head_dim_of,
                            prefill_and_fill, unpatch, write_tokens)
from m4b_paged_prefill import paged_prefill_triton, write_chunk


def make_fwd(li: int, st: dict):
    """同时处理 q_len == 1（decode）与 q_len > 1（prefill 的一块）。"""

    def fwd(self, hidden_states, position_embeddings=None, attention_mask=None,
            cache=None, **kw):
        input_shape = hidden_states.shape[:-1]
        b, qlen = input_shape
        hs = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        pk, pv, bt, seq, bs = st["pk"], st["pv"], st["bt"], st["seq"], st["bs"]
        if qlen == 1:
            kk = k[:, :, 0, :].contiguous()
            vv = v[:, :, 0, :].contiguous()
            write_tokens(pk, pv, bt, seq, li, kk, vv, bs)
            o = st["attn"](q[:, :, 0, :].contiguous(), pk[li], pv[li], bt, seq,
                           bs, self.scaling)
            o = o.reshape(*input_shape, -1).contiguous()
            return self.o_proj(o), None

        # --- prefill 的一块：先整段写进池子，再做 (rows, HQ, T, D) 的 attention
        start = (seq.long() - qlen)
        write_chunk(pk[li], pv[li], bt, k, v, start, bs)
        o = paged_prefill_triton(q, pk[li], pv[li], bt, start, seq.long(), bs,
                                 self.scaling)
        o = o.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(o), None

    return fwd


def patch(model, st: dict):
    saved = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        saved.append((attn, attn.forward))
        attn.forward = types.MethodType(make_fwd(i, st), attn)
    return saved


@torch.no_grad()
def prefill_chunked(model, st: dict, ids, chunk: int, start_at: int = 0):
    """把 ids 的 [start_at:] 切成 chunk 块喂进去。返回最后位置的 logits。

    `start_at > 0` 用于 prefix caching：前 `start_at` 个 token 的 KV 已经在池子里
    （可能是共享块），所以直接从那个位置开始写和算。
    """
    L = ids.shape[1]
    st["seq"].fill_(start_at)
    logits = None
    for s in range(start_at, L, chunk):
        c = min(chunk, L - s)
        st["seq"] += c
        pos = torch.arange(s, s + c, device="cuda").view(1, -1)
        out = model(input_ids=ids[:, s:s + c], position_ids=pos,
                    use_cache=False, attention_mask=None)
        logits = out.logits[:, -1, :].float()
    return logits


@torch.no_grad()
def hf_reference(model, ids, steps: int):
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    toks = [nxt[:, 0]]
    for _ in range(steps - 1):
        out = model(input_ids=nxt, cache=cache, use_cache=True)
        cache = out.cache if hasattr(out, "cache") else out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(nxt[:, 0])
    return torch.stack(toks, dim=1)


def bench_chunks(model, cfg, prompt_len: int, chunks, steps: int, tok) -> None:
    print(f"\n[分块 prefill + 分页 decode] prompt={prompt_len} token（真实文本），"
          f"decode {steps} 步")
    print(f"{'块大小':>8} {'块数':>5} {'prefill ms':>11} {'prefill tok/s':>14} "
          f"{'decode ms/步':>13} {'token 一致':>11}")
    text = (FILLER * 20)[: prompt_len * 4]
    ids = tok(text, return_tensors="pt").input_ids[:, :prompt_len].cuda()
    prompt_len = ids.shape[1]
    ref = hf_reference(model, ids, steps)
    for chunk in chunks:
        pool = build_pool(cfg, 1, prompt_len + steps + 8, 16)
        st = {"pk": pool[0], "pv": pool[1], "bt": pool[2], "seq": pool[3],
              "bs": 16, "pos": torch.zeros(1, dtype=torch.long, device="cuda"),
              "attn": paged_attn_triton}
        saved = patch(model, st)
        try:
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(True)
            t1 = torch.cuda.Event(True)
            t0.record()
            logits = prefill_chunked(model, st, ids, chunk)
            t1.record()
            torch.cuda.synchronize()
            pre_ms = t0.elapsed_time(t1)
            nxt = logits.argmax(-1)
            ids2 = nxt.view(1, 1)
            st["pos"].fill_(prompt_len)
            toks, _, times = decode(model, st, ids2, steps)
            # 对齐：hf_reference 的第 0 个 token 就是 prefill 的 argmax（已作为
            # ids2 喂进了 decode），所以 paged 的 toks[:,0] 等于 ref[:,1]。
            k = min(toks.shape[1], ref.shape[1] - 1)
            eq = (toks[:, :k] == ref[:, 1:k + 1])
            if bool(eq.all()):
                verdict = "一致 ✅"
            else:
                first = int((~eq).nonzero()[0][1])
                gap = (logits[0].topk(2).values[0] - logits[0].topk(2).values[1])
                verdict = f"第{first}步起不一致（top1-top2={gap:.4f}）"
            print(f"{chunk:>8} {(prompt_len + chunk - 1) // chunk:>5} "
                  f"{pre_ms:>11.1f} {prompt_len / pre_ms * 1000:>14.0f} "
                  f"{statistics.median(times):>13.2f} {verdict:>11}")
        finally:
            unpatch(saved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--verify-prompt", type=int, default=192)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda", local_files_only=True)
    model.eval()
    cfg = model.config
    print(f"模型 {MODEL}  layers={cfg.num_hidden_layers} "
          f"hq={cfg.num_attention_heads} hkv={cfg.num_key_value_heads} "
          f"head_dim={head_dim_of(cfg)}")

    # ---- 1) 正确性：同一 prompt，整条 prefill vs 切 2/3/5 块，token 必须一致
    texts = [FILLER[: 7 * args.verify_prompt]]
    ids = tok(texts, return_tensors="pt").input_ids.cuda()
    L = ids.shape[1]
    print(f"\n[正确性] prompt {L} token，decode {args.steps} 步")
    pool = build_pool(cfg, 1, L + 64, 16)
    base_toks = None
    for chunk in (L, max(1, L // 2), max(1, L // 3), max(1, L // 5)):
        st = {"pk": pool[0], "pv": pool[1], "bt": pool[2],
              "seq": torch.zeros(1, dtype=torch.int32, device="cuda"),
              "bs": 16, "pos": torch.zeros(1, dtype=torch.long, device="cuda"),
              "attn": paged_attn_triton}
        saved = patch(model, st)
        try:
            logits = prefill_chunked(model, st, ids, chunk)
            nxt = logits.argmax(-1).view(1, 1)
            st["pos"].fill_(L)
            toks, _, _ = decode(model, st, nxt, args.steps)
            if base_toks is None:
                base_toks = toks[:, 0].clone()
                print(f"  块={chunk:4d}（=整条）token: {toks[:, 0].tolist()}")
            else:
                same = torch.equal(toks[:, 0], base_toks)
                print(f"  块={chunk:4d} token: {toks[:, 0].tolist()}  "
                      f"{'一致 ✅' if same else '不一致 ❌'}")
        finally:
            unpatch(saved)

    # ---- 2) 时间：块大小扫描
    bench_chunks(model, cfg, args.prompt, [args.prompt, 512, 256, 128, 64],
                 args.steps, tok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
