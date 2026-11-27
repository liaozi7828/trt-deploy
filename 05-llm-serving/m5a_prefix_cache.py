"""M5a：prefix caching —— 相同前缀的 KV 块直接复用，只 prefill 不同的尾巴。

判据（先正确性后速度）
----------------------
同一批请求，开缓存与关缓存，**吐出的 token 必须逐位一致**。
然后再看省了多少 prefill token、TTFT 降多少。

分级设计（工程上真正要写的东西）
--------------------------------
- `BlockBank`：`hash -> (blk, refcount)`，按**块内容的链式哈希**索引；
  只有**填满的**块才会被登记（半空的尾巴块是私有的，否则 decode 会写脏共享块）。
- 请求进来时先算"最长命中前缀"（块对齐），命中的块直接挂进它的 block table，
  prefill 只跑 `ids[cached:]`，起点位置 `cached`。
- 请求结束后把它 prompt 里**填满的块**登记进 bank（refcount +1）。

已知简化
--------
- 只复用 prompt 前缀，不复用"上一次生成的 token"（那需要 session 级别的复用）。
- LRU 淘汰、内存压力、跨请求的 refcount 释放策略都没实现（本机显存足够）。
"""
from __future__ import annotations

import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3b_paged_attn import paged_attn_triton
from m3c_qwen_paged import MODEL, build_pool, decode, head_dim_of, unpatch
from m4c_qwen_chunk import patch, prefill_chunked


class BlockBank:
    """块级 prefix cache：链式哈希 -> 共享块号。"""

    def __init__(self, nblocks: int, block_size: int):
        self.bs = block_size
        self.free = list(range(nblocks))
        self.table: dict[int, list[int]] = {}     # hash -> [blk, refcount]
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0

    # ---- 分配 ----
    def alloc(self) -> int:
        if not self.free:
            raise RuntimeError("块池耗尽")
        return self.free.pop()

    def release(self, blk: int) -> None:
        self.free.append(blk)

    # ---- 哈希 ----
    def block_hashes(self, ids) -> list[int]:
        """(L,) 或 (1, L) -> 满块数 个链式哈希（最后半块不登记）。"""
        ids = ids.reshape(-1)                  # 传 (1, L) 时 shape[0] 是 1，会算成 0 块
        n = ids.shape[0] // self.bs
        out, h = [], 0
        toks = ids.tolist()
        for i in range(n):
            h = hash((h, tuple(toks[i * self.bs:(i + 1) * self.bs])))
            out.append(h)
        return out

    def match(self, ids) -> int:
        """返回最长命中的前缀 token 数（块对齐）。"""
        n = 0
        for h in self.block_hashes(ids):
            if h not in self.table:
                break
            n += 1
        return n * self.bs

    # ---- 登记 ----
    def commit(self, ids, row_blocks: list[int]) -> None:
        for h, blk in zip(self.block_hashes(ids), row_blocks):
            if h in self.table:
                self.table[h][1] += 1
            else:
                self.table[h] = [blk, 1]


def build_tables(cfg, rows: int, cap: int, bs: int, nblocks: int):
    """建池子，但 block table 由外部（bank）决定，不再按行连续切。"""
    pk, pv, bt, seq, per_row = build_pool(cfg, rows, cap, bs)
    pk, pv = pk, pv
    return pk, pv, bt, seq, per_row


@torch.no_grad()
def run_request(model, st, bank, ids, chunk: int, use_cache: bool,
                debug: bool = False):
    """一条请求：算命中前缀 → 只 prefill 未命中部分 → 返回 logits 与新块列表。"""
    L = ids.shape[1]
    bs = st["bs"]
    row = 0
    hashes = bank.block_hashes(ids)
    cached = bank.match(ids) if use_cache else 0
    cached = min(cached, (L // bs) * bs)          # 只能命中满块

    # 该行需要的块数（含尾巴块）
    need = (L + bs - 1) // bs
    row_blocks = []
    # 1) 命中的共享块（只挂表，不写）
    n_cached_blk = cached // bs
    for i in range(n_cached_blk):
        h = hashes[i]
        blk, ref = bank.table[h]
        bank.table[h][1] = ref + 1
        row_blocks.append(blk)
        bank.reused_tokens += bs
    # 2) 剩下的私有块
    for _ in range(need - n_cached_blk):
        row_blocks.append(bank.alloc())

    st["bt"][row, :len(row_blocks)] = torch.tensor(row_blocks, dtype=torch.int32,
                                                   device="cuda")
    logits = prefill_chunked(model, st, ids, chunk, start_at=cached)
    if debug:
        print(f"    [dbg] L={L} cached={cached} 新块={need - n_cached_blk} "
              f"空闲={len(bank.free)} 表={len(bank.table)}")

    if use_cache:
        # 只登记"已填满"的块：尾巴块留私有，否则 decode 会写脏共享块
        n_full = L // bs
        bank.commit(ids[:, : n_full * bs], row_blocks[:n_full])
    return logits, row_blocks


def bench(model, cfg, bank, prefix_text, suffixes, chunk, use_cache, steps,
          nblocks, bs):
    out = []
    for suf in suffixes:
        ids = tokenizer_ids(prefix_text + suf)
        # 不开缓存时每条请求用新 bank（否则块会被耗光）
        use_bank = bank if use_cache else BlockBank(nblocks, bs)
        st = {"pk": pool_pk, "pv": pool_pv, "bt": pool_bt,
              "seq": torch.zeros(1, dtype=torch.int32, device="cuda"),
              "bs": bs, "pos": torch.zeros(1, dtype=torch.long, device="cuda"),
              "attn": paged_attn_triton}
        saved = patch(model, st)
        try:
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            logits, _ = run_request(model, st, use_bank, ids, chunk, use_cache,
                                    debug=use_cache)
            t1.record()
            torch.cuda.synchronize()
            ttft = t0.elapsed_time(t1)
            nxt = logits.argmax(-1).view(1, 1)
            st["pos"].fill_(ids.shape[1])
            toks, _, times = decode(model, st, nxt, steps)
            out.append({"ttft_ms": ttft, "itl_ms": statistics.median(times),
                        "tokens": toks[:, 0].tolist()})
        finally:
            unpatch(saved)
    return out


tokenizer = None
pool_pk = pool_pv = pool_bt = None


def tokenizer_ids(text: str):
    return tokenizer(text, return_tensors="pt").input_ids.cuda()


def main() -> int:
    global tokenizer, pool_pk, pool_pv, pool_bt
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--reqs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda", local_files_only=True)
    model.eval()
    cfg = model.config
    bs = 16
    cap = args.prefix + 256
    pool_pk, pool_pv, pool_bt, _, per_row = build_tables(cfg, 1, cap, bs, 4096)
    nblocks = pool_pk[0].shape[0]
    print(f"模型 {MODEL}  块大小 {bs}  池子 {nblocks} 块 "
          f"({nblocks * bs / 1000:.0f} K token 容量)")

    # 系统提示（所有请求共享）+ 各不相同的尾巴
    sys_text = ("You are a helpful assistant.\n\n" +
                "Reference manual:\n" +
                ("The quick brown fox jumps over the lazy dog. " * 200))
    sys_ids = tokenizer(sys_text, return_tensors="pt").input_ids
    prefix_text = tokenizer.decode(sys_ids[0, : args.prefix])
    suffixes = [f"\n\nQuestion {i}: summarize the manual in one line.\nAnswer:"
                for i in range(args.reqs)]

    print(f"\n系统提示 {args.prefix} token（共享），每条请求尾巴不同，"
          f"decode {args.steps} 步，块 {bs}，prefill 分块 {args.chunk}")

    # ---- 预热：第一次调用含 Triton JIT 编译，绝不能算进计时 ----
    warm_bank = BlockBank(nblocks, bs)
    bench(model, cfg, warm_bank, prefix_text, ["\n\nwarmup?"], args.chunk,
          use_cache=False, steps=1, nblocks=nblocks, bs=bs)

    # ---- 不开缓存 ----
    bank_off = BlockBank(nblocks, bs)
    off = bench(model, cfg, bank_off, prefix_text, suffixes, args.chunk,
                use_cache=False, steps=args.steps, nblocks=nblocks, bs=bs)
    off_cost = sum(r["ttft_ms"] for r in off)

    # ---- 开缓存（同一个 bank 顺序服务这批请求）----
    bank_on = BlockBank(nblocks, bs)
    on = bench(model, cfg, bank_on, prefix_text, suffixes, args.chunk,
               use_cache=True, steps=args.steps, nblocks=nblocks, bs=bs)
    on_cost = sum(r["ttft_ms"] for r in on)

    print(f"\n{'#':>3} {'无缓存 TTFT':>12} {'有缓存 TTFT':>12} {'降幅':>8} "
          f"{'token 一致':>11}")
    for i, (a, b) in enumerate(zip(off, on)):
        same = "✅" if a["tokens"] == b["tokens"] else "❌"
        print(f"{i:>3} {a['ttft_ms']:>12.1f} {b['ttft_ms']:>12.1f} "
              f"{(1 - b['ttft_ms'] / a['ttft_ms']) * 100:>7.0f}% {same:>11}")
    print(f"\n总 prefill 时间 {off_cost:.1f} -> {on_cost:.1f} ms "
          f"({off_cost / on_cost:.2f}x)")
    print(f"复用的 KV token 数 {bank_on.reused_tokens}，"
          f"缓存表里 {len(bank_on.table)} 个块")
    allsame = all(a["tokens"] == b["tokens"] for a, b in zip(off, on))
    print(f"\n全部 token 一致: {'✅' if allsame else '❌'}")
    return 0 if allsame else 1


if __name__ == "__main__":
    raise SystemExit(main())
