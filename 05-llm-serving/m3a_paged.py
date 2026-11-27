#!/usr/bin/env python
"""M3a：分页 KV cache（按需分配 block + block table 寻址）。

这一步只做两件事，刻意先不接 CUDA Graph：

  1. 正确性：分页路径算出的 token，必须与 dense 静态 cache 逐位一致。
  2. 显存：把"槽位 x 水位"的预留式分配，换成"按行按需分配 block"，
     量出预分配 vs 实际用掉多少。

为什么先不接 graph：gather 出来的宽张量必须形状固定，接了 graph 反而会把
"按需"的自由度锁死，看不出分配器本身的价值。速度留给 M3b（paged attention kernel）。

关键实现：
  * keys/values 存在 pool 里，形状 (num_blocks, block_size, kv_heads, head_dim)
  * 每行一张 block_table (slots, max_blocks)，只记物理块号
  * 写：一次 index_copy_ 写到 (block*block_size + offset)
  * 读：整个 pool 按 block_table gather 成 (B, H, M, D)

本步所有行同步推进（pos 相同），所以 KV 是 0..pos-1 的连续前缀，
不需要逐行 mask —— 逐行不同长度是 M3b 才有意义的东西。
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 16          # 每块 16 个 token（vLLM 默认也是 16）
DENSE_CAP = 256     # dense 对照组的固定水位


class PagedLayer:
    """只实现 decode 需要的两个动作：写若干行、gather 出可见的 KV。"""

    def __init__(self, pool_k: torch.Tensor, block_size: int):
        self.keys = pool_k                 # (num_blocks, block_size, H, D)
        self.block_size = block_size

    def write_rows(self, blocks: torch.Tensor, offsets: torch.Tensor, k: torch.Tensor) -> None:
        """blocks/offsets: (n,) 长整型；k: (n, H, D)。一次写 n 行。"""
        flat = self.keys.view(-1, self.keys.shape[2], self.keys.shape[3])
        idx = blocks * self.block_size + offsets
        flat.index_copy_(0, idx, k)

    def gather(self, bt: torch.Tensor, width: int):
        """bt: (B, nblk) 物理块号 -> (B, H, nblk*block_size, D) 再切到 width。"""
        k = self.keys[bt]                              # (B, nblk, bs, H, D)
        b, nblk, bs, h, d = k.shape
        return k.reshape(b, nblk * bs, h, d).transpose(1, 2)[:, :, :width, :]


class PagedCache:
    """对外只需要 pos / bt / gather() / ensure_blocks()。"""

    def __init__(self, config, slots: int, max_blocks: int, block_size: int = BLOCK,
                 dtype=torch.float16, device="cuda"):
        self.slots = slots
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or \
            config.hidden_size // config.num_attention_heads
        self.device = device

        shape = (slots * max_blocks, block_size, self.n_heads, self.head_dim)
        self.pool_k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(self.n_layers)]
        self.pool_v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(self.n_layers)]
        self.layers = [PagedLayer(self.pool_k[i], block_size) for i in range(self.n_layers)]

        self.pos = torch.zeros(slots, dtype=torch.long, device=device)     # 本步写到哪
        self.bt = torch.zeros(slots, max_blocks, dtype=torch.long, device=device)
        self.nblk = torch.zeros(slots, dtype=torch.long, device=device)
        self.next_blk = 0

    # -- 分配 -------------------------------------------------------------
    def ensure_blocks(self, row: int, need_tokens: int) -> None:
        need_blk = (need_tokens + self.block_size - 1) // self.block_size
        while int(self.nblk[row]) < need_blk:
            if self.next_blk >= self.slots * self.max_blocks:
                raise RuntimeError(f"物理块用尽：{self.next_blk} 块")
            self.bt[row, int(self.nblk[row])] = self.next_blk
            self.next_blk += 1
            self.nblk[row] += 1

    # -- 写一个 token（所有层一起）----------------------------------------
    def append_token(self, k_rows, v_rows) -> None:
        """k_rows/v_rows: list[n_layers] of (B, H, D)。写到每行 pos 处，然后 pos += 1。"""
        pos = self.pos
        blocks = self.bt.gather(1, (pos // self.block_size)[:, None])[:, 0]
        offs = pos % self.block_size
        idx = blocks * self.block_size + offs
        for l in range(self.n_layers):
            self.layers[l].write_rows(blocks, offs, k_rows[l])
            flat_v = self.pool_v[l].view(-1, self.n_heads, self.head_dim)
            flat_v.index_copy_(0, idx, v_rows[l])
        self.pos += 1

    def gather(self, rows: int):
        width = int(self.pos[:rows].max())
        bt = self.bt[:rows]
        ks = [l.gather(bt, width) for l in self.layers]
        vs = []
        for l in range(self.n_layers):
            v = self.pool_v[l][bt]
            b, nblk, bs, h, d = v.shape
            vs.append(v.reshape(b, nblk * bs, h, d).transpose(1, 2)[:, :, :width, :])
        return ks, vs

    # -- 账 ---------------------------------------------------------------
    def reserved_bytes(self) -> int:
        n = self.pool_k[0].numel() * self.pool_k[0].element_size()
        return 2 * self.n_layers * n

    def used_bytes(self) -> int:
        per_block = self.block_size * self.n_heads * self.head_dim * 2 * 2
        return self.next_blk * per_block * self.n_layers


def kv_bytes_per_token(cfg) -> int:
    h = cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * h * d * 2


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--reqs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--block", type=int, default=BLOCK)
    args = ap.parse_args()

    bs = args.block
    dev = "cuda"
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    print(f"{args.model}  layers={cfg.num_hidden_layers} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={hd} block={bs}  KV={kv_bytes_per_token(cfg) / 1024:.0f} KB/token")

    text = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids[0]
    ids = text[4:4 + args.prompt].unsqueeze(0).repeat(args.reqs, 1).to(dev)

    # ---- 1. dense 路径 ----
    from transformers import StaticCache, DynamicCache
    dcache = StaticCache(cfg, max_cache_len=DENSE_CAP)
    out = model(input_ids=ids, cache=dcache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    first = nxt.clone()                      # prefill 给出的第一个 token（两条路径的公共起点）
    dense_tokens = nxt.clone()
    dense_ms = []
    for _ in range(args.steps - 1):
        t0 = time.perf_counter()
        out = model(input_ids=nxt, cache=dcache, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        dense_tokens = torch.cat([dense_tokens, nxt], dim=1)
        dense_ms.append((time.perf_counter() - t0) * 1000)

    # ---- 2. 分页路径：把 dense 的 KV 搬进物理池，再逐 token decode ----
    max_blocks = (args.prompt + args.steps + bs - 1) // bs + 2
    pc = PagedCache(cfg, args.reqs, max_blocks, bs, torch.float16, dev)
    for r in range(args.reqs):
        pc.ensure_blocks(r, args.prompt)
        for l in range(pc.n_layers):
            for bi in range(int(pc.nblk[r])):
                s = bi * bs
                e = min(s + bs, args.prompt)
                blk = int(pc.bt[r, bi])
                pc.pool_k[l][blk, :e - s] = dcache.layers[l].keys[r, :, s:e, :].transpose(0, 1)
                pc.pool_v[l][blk, :e - s] = dcache.layers[l].values[r, :, s:e, :].transpose(0, 1)
        pc.pos[r] = args.prompt
        pc.ensure_blocks(r, args.prompt + 1)

    paged_tokens = first.clone()             # 从同一个起点开始，否则会整体错开一格
    paged_ms = []
    for _ in range(args.steps - 1):
        t0 = time.perf_counter()
        ks, vs = pc.gather(args.reqs)
        cache = DynamicCache()
        for l in range(pc.n_layers):
            cache.update(ks[l].contiguous(), vs[l].contiguous(), l)
        pos_ids = torch.full((args.reqs, 1), int(pc.pos.max()), dtype=torch.long, device=dev)
        out = model(input_ids=paged_tokens[:, -1:], cache=cache,
                    position_ids=pos_ids, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        paged_tokens = torch.cat([paged_tokens, nxt], dim=1)

        # DynamicCache 里最后一个位置的 KV 就是刚算出的新 token，取出来写进物理池
        k_rows = [cache.layers[l].keys[:, :, -1, :].clone() for l in range(pc.n_layers)]
        v_rows = [cache.layers[l].values[:, :, -1, :].clone() for l in range(pc.n_layers)]
        pc.append_token(k_rows, v_rows)
        for r in range(args.reqs):
            pc.ensure_blocks(r, int(pc.pos[r]) + 1)
        paged_ms.append((time.perf_counter() - t0) * 1000)

    print(f"dense  {dense_tokens[0, :8].tolist()}")
    print(f"paged  {paged_tokens[0, :8].tolist()}")
    same = torch.equal(dense_tokens, paged_tokens)
    print(f"对拍：{'逐位一致 OK' if same else '不一致 FAIL'}  ({dense_tokens.numel()} 个 token)")
    if not same:
        bad = (dense_tokens != paged_tokens).nonzero()
        print(f"  首个分叉: {bad[0].tolist()}")

    print(f"\n步时中位数  dense {statistics.median(dense_ms):.2f} ms   "
          f"paged {statistics.median(paged_ms):.2f} ms")
    print(f"物理块：已分配 {pc.next_blk}（每行上限 {max_blocks}，共 {pc.slots * max_blocks}）")
    print(f"KV 显存：按需 {pc.used_bytes() / 2**20:.1f} MiB / 预留 {pc.reserved_bytes() / 2**20:.1f} MiB / "
          f"dense 对照组 {args.reqs * DENSE_CAP * kv_bytes_per_token(cfg) / 2**20:.1f} MiB")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
