"""M3c：把 Triton 分页 attention 接进 Qwen2.5，端到端对拍 + 计时。

验收线先定死：**token 必须逐位一致**，再看时间。

做法
----
1. prefill 仍走 HF 原路径（DynamicCache），逐条跑；把每层 KV 拷进分页池。
2. decode 时把每层 `self_attn.forward` 换成自己的实现：
   Q/K/V 投影 + RoPE（复用 HF 的 apply_rotary_pos_emb）→ 写一个 token 进池子
   → 分页 attention（Triton kernel 或 gather+SDPA）→ o_proj。
3. 三条路对拍：HF 原路径 / 分页+gather / 分页+Triton kernel。
"""
from __future__ import annotations

import argparse
import statistics
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from m3b_paged_attn import paged_attn_gather, paged_attn_triton

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FILLER = "The quick brown fox jumps over the lazy dog. " * 400


# ----------------------------------------------------------------------------
def head_dim_of(cfg) -> int:
    return getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)


def build_pool(cfg, rows: int, cap: int, bs: int):
    nl = cfg.num_hidden_layers
    hkv = cfg.num_key_value_heads
    d = head_dim_of(cfg)
    per_row = (cap + bs - 1) // bs
    nb = rows * per_row + 8
    pk = [torch.zeros(nb, bs, hkv, d, dtype=torch.float16, device="cuda") for _ in range(nl)]
    pv = [torch.zeros(nb, bs, hkv, d, dtype=torch.float16, device="cuda") for _ in range(nl)]
    bt = torch.zeros(rows, per_row, dtype=torch.int32, device="cuda")
    for r in range(rows):                       # 每行先占住自己的一段块
        bt[r] = torch.arange(r * per_row, (r + 1) * per_row, dtype=torch.int32, device="cuda")
    seq = torch.zeros(rows, dtype=torch.int32, device="cuda")
    return pk, pv, bt, seq, per_row


def write_tokens(pk, pv, bt, seq, li: int, k, v, bs: int):
    """k/v: (rows, hkv, d) fp16。

    约定：`seq` 是**当前序列长度（已含正在生成的这个 token）**，
    所以新 token 写到下标 `seq-1`，attention 读 `0..seq-1`（长度 seq）。
    """
    rows = seq.numel()
    st = seq.long() - 1
    blk = bt[torch.arange(rows, device=seq.device), st // bs].long()
    off = st % bs
    pk[li][blk, off] = k
    pv[li][blk, off] = v


def make_fwd(li: int, st: dict):
    def fwd(self, hidden_states, position_embeddings=None, attention_mask=None,
            cache=None, **kw):
        # 注意：transformers 5.x 的签名是 (hidden_states, position_embeddings,
        # attention_mask, cache=None)，且 attention 模块上只有 head_dim /
        # num_key_value_groups / scaling，头数要从 self.config 读。
        input_shape = hidden_states.shape[:-1]
        b, qlen = input_shape
        if qlen != 1:
            raise RuntimeError(f"分页路径只处理 decode（qlen=1），收到 {qlen}")
        hs = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        kk = k[:, :, 0, :].contiguous()
        vv = v[:, :, 0, :].contiguous()
        write_tokens(st["pk"], st["pv"], st["bt"], st["seq"], li, kk, vv, st["bs"])
        o = st["attn"](q[:, :, 0, :].contiguous(), st["pk"][li], st["pv"][li],
                       st["bt"], st["seq"], st["bs"], self.scaling)
        # kernel 返回的是 (B, Hq, D)：q_len=1，所以直接展平成 (B, 1, Hq*D)。
        # 千万别 transpose(1,2) —— 那会把 (Hq,D) 转成 (D,Hq)，头内维度全错位。
        o = o.reshape(*input_shape, -1).contiguous()
        return self.o_proj(o), None
    return fwd


def patch(model, st: dict):
    saved = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        saved.append((attn, attn.forward))
        attn.forward = types.MethodType(make_fwd(i, st), attn)
    return saved


def unpatch(saved):
    for attn, fn in saved:
        attn.forward = fn


# ----------------------------------------------------------------------------
@torch.no_grad()
def prefill_and_fill(model, pool, prompts, rows: int, bs: int):
    """逐条 prefill（HF 原路径），把 KV 拷进池子。返回每行最后一个 token 的 id。"""
    pk, pv, bt, seq, _ = pool
    nl = len(pk)
    first = []
    for r in range(rows):
        enc = torch.tensor([prompts[r]], dtype=torch.long, device="cuda")
        out = model(input_ids=enc, use_cache=True)
        layer0 = out.past_key_values.layers[0]
        L = layer0.keys.shape[2]
        for li in range(nl):
            ks = out.past_key_values.layers[li].keys[0].transpose(0, 1)   # (L, hkv, d)
            vs = out.past_key_values.layers[li].values[0].transpose(0, 1)
            per_row = bt.shape[1]
            blk = bt[r, torch.arange(0, L, device="cuda") // bs].long()
            off = torch.arange(0, L, device="cuda") % bs
            pk[li][blk, off] = ks
            pv[li][blk, off] = vs
        seq[r] = L
        first.append(out.logits[0, -1, :].argmax())
    return torch.stack(first)


@torch.no_grad()
def decode(model, st: dict, ids, steps: int):
    """ids: (rows, 1)。返回 (rows, steps) 的 token、(rows, steps, V) 的 logits、每步墙钟。"""
    pos = st["pos"].clone().view(-1, 1)          # 必须是 2-D (B, q_len)
    toks, lgs, times = [], [], []
    for _ in range(steps):
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        st["seq"] += 1                       # 先加长度：新 token 落在 seq-1
        t0.record()
        out = model(input_ids=ids, position_ids=pos, use_cache=False, attention_mask=None)
        logits = out.logits[:, -1, :].float()
        nxt = logits.argmax(-1, keepdim=True)
        ids.copy_(nxt)
        pos += 1
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
        toks.append(nxt[:, 0].clone())
        lgs.append(logits.clone())
    return torch.stack(toks, dim=1), torch.stack(lgs, dim=1), times


@torch.no_grad()
def reference(model, prompts, steps: int):
    toks, lgs = [], []
    for p in prompts:
        enc = torch.tensor([p], dtype=torch.long, device="cuda")
        out = model(input_ids=enc, use_cache=True)
        cache = out.past_key_values
        row_t, row_l = [], []
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        row_t.append(nxt[0, 0].item())
        row_l.append(out.logits[0, -1, :].float())
        for _ in range(steps - 1):
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            row_t.append(nxt[0, 0].item())
            row_l.append(out.logits[0, -1, :].float())
        toks.append(row_t)
        lgs.append(torch.stack(row_l))
    return torch.tensor(toks, device="cuda"), torch.stack(lgs)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--pmin", type=int, default=64)
    ap.add_argument("--pmax", type=int, default=192)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda", attn_implementation="eager",
        local_files_only=True)
    model.eval()
    cfg = model.config
    print(f"{MODEL}  layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"kv_heads={cfg.num_key_value_heads} d={head_dim_of(cfg)}")

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    full = tok(FILLER, return_tensors="pt").input_ids[0].tolist()
    prompts = []
    for r in range(args.rows):
        L = args.pmin + (args.pmax - args.pmin) * (r % 4) // 3
        off = r * 13                       # 每行错开一点，避免完全相同的 prompt
        prompts.append(full[off: off + L])
    lens = [len(p) for p in prompts]
    print(f"prompt 长度 {lens}")

    cap = max(lens) + args.steps + 4
    pool = build_pool(cfg, args.rows, cap, args.block)

    # 参考答案（HF 原路径，逐条）
    ref, ref_lg = reference(model, prompts, args.steps)
    print(f"参考 token 前 3 步: {ref[:, :3].tolist()}")

    ids = prefill_and_fill(model, pool, prompts, args.rows, args.block).view(-1, 1)
    firsts = ids[:, 0].tolist()

    results = {}
    for mode in ("gather", "kernel"):
        pool2 = build_pool(cfg, args.rows, cap, args.block)
        st = {"pk": pool2[0], "pv": pool2[1], "bt": pool2[2], "seq": pool2[3],
              "bs": args.block, "pos": torch.tensor(lens, dtype=torch.long, device="cuda"),
              "attn": paged_attn_triton if mode == "kernel" else paged_attn_gather}
        prefill_and_fill(model, pool2, prompts, args.rows, args.block)
        saved = patch(model, st)
        cur = ids.clone()
        torch.cuda.synchronize()
        toks, lgs, times = decode(model, st, cur, args.steps)
        unpatch(saved)
        ok = bool(torch.equal(toks[:, :-1], ref[:, 1:]))
        med = statistics.median(times[2:]) if len(times) > 2 else statistics.median(times)
        results[mode] = (ok, med, toks)
        # 对齐：我的第 k 步 logits 预测的是第 k+1 个 token，对应参考的第 k+1 步
        mine_lg, mine_tok = lgs[:, :-1], toks[:, :-1]
        ref_lg2, ref_tok2 = ref_lg[:, 1:], ref[:, 1:]
        step_ok = (mine_tok == ref_tok2).all(dim=0).sum().item()
        dl = (mine_lg - ref_lg2).abs()
        rel = (dl.max(dim=-1).values / ref_lg2.abs().max(dim=-1).values.clamp_min(1e-9))
        print(f"[{mode:6s}] token 逐位={ok}  步时中位数={med:.2f} ms  "
              f"全程一致的步={step_ok}/{mine_tok.shape[1]}")
        print(f"          logits max|Δ| 首步={dl[:, 0].max():.3e} 末步={dl[:, -1].max():.3e}  "
              f"相对 max={rel.max():.3e}")
        bad = (mine_tok != ref_tok2).any(dim=1).nonzero().flatten().tolist()
        print(f"          不一致的行={bad}")

    # 参考路径的步时（batch 化的 HF 原路径：逐条跑，取中位数）
    with torch.no_grad():
        encs = [torch.tensor([p], dtype=torch.long, device="cuda") for p in prompts]
        outs = [model(input_ids=e, use_cache=True) for e in encs]
        caches = [o.past_key_values for o in outs]
        nxt = [o.logits[:, -1, :].argmax(-1, keepdim=True) for o in outs]
        times = []
        for _ in range(args.steps - 1):
            t0 = torch.cuda.Event(True)
            t1 = torch.cuda.Event(True)
            t0.record()
            for i in range(args.rows):
                o = model(input_ids=nxt[i], past_key_values=caches[i], use_cache=True)
                caches[i] = o.past_key_values
                nxt[i] = o.logits[:, -1, :].argmax(-1, keepdim=True)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
    base = statistics.median(times[2:]) if len(times) > 2 else statistics.median(times)
    print(f"[HF 原路径] 逐条跑 {args.rows} 条一轮 = {base:.2f} ms "
          f"（每条约 {base / args.rows:.2f} ms）")

    print("\n首 token 一致性：", firsts == ref[:, 0].tolist())
    for mode, (ok, med, _) in results.items():
        print(f"  {mode:6s}  {med:7.2f} ms   vs HF 逐条 {base / med:.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
