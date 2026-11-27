"""M2b：连续批处理（continuous batching）—— 固定槽位 + 静态 KV + 单张 decode 图

和 m2a 那三种"一次性排好"的模式不同，这里做**在线调度**：
槽位满了就等，谁先做完，立刻把等待队列里的下一条补进**同一个槽位**。
这是 vLLM/TRT-LLM 里 continuous batching 的核心动作 —— 只是 KV 还是静态槽位、
不是 paged block，所以 m2b 同时也是 M3（paged KV）的对照组。

两条关键设计（都是被 m0d/m0e/m2a 的实测逼出来的）：

1. **所有行共享同一个"填充水位"（level）**。
   HF 的 StaticCache 用**标量** cumulative_length 决定写入位置，一行一个位置它表达不了
   （见 _probe_cache* 的结论：StaticLayer.update 用共享的 cache_position）。
   所以这里不追求 per-row 位置，而是要求**任意时刻所有行的 KV 都填到同一层**：
   新请求进来时用"填充 token + 真 prompt"左填到当前 level，prefill 完再按行拷进槽位。
   代价 = 每次补位都要 prefill level 个 token（其中只有 plen 个是真的）；
   好处 = decode 全程**一张图**、causal mask 只依赖标量 level，可整体 CUDA Graph 捕获。

2. **level 在连续模式下只涨不跌**，所以 CAP 要按**整轮**的步数算：
   `CAP = pmax + ceil(reqs/slots) × max(max_new) + 4`。
   只有"所有槽位都空了"那一刻才允许把 level 调小（新一批会重写 0..level-1）。
   代价：连续模式下越往后每行都要多读前面几个 token 的 KV —— 这是静态槽位的固有开销，
   也是 M3（paged KV）要消掉的东西。

对照组 wave：每批 N 条一起跑到**最慢的那条**结束，再放下一批。
两组的 prefill 总量、填充总量完全一样，唯一差别是**短请求做完后槽位是否空转**
—— 这样比较才干净。因此请求的 max_new 必须长短不同（统一 32 的话两组结果完全一样）。

用法：
    ~/.venvs/torch/bin/python m2b_continuous.py
    ~/.venvs/torch/bin/python m2b_continuous.py --slots 24 --reqs 96
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
W_MB = 942.0        # 权重体积（实测）
PEAK_BW = 334.0     # m0b 实测等效带宽 GB/s
FILLER = (
    "The quick brown fox jumps over the lazy dog near the river bank while the "
    "morning light slowly climbs over the quiet hills and the old bridge. "
) * 40


@dataclasses.dataclass
class Req:
    rid: int
    plen: int
    max_new: int
    ids: list
    slot: int = -1
    admit_ms: float = 0.0
    done_ms: float = 0.0
    gen: int = 0
    tokens: list = dataclasses.field(default_factory=list)   # 只在 --verify 时填

    @property
    def e2e_ms(self) -> float:
        return self.done_ms - self.admit_ms if self.done_ms else 0.0


@dataclasses.dataclass
class Stat:
    name: str = ""
    wall_ms: float = 0.0
    capture_ms: float = 0.0
    prefill_ms: float = 0.0
    pad_tokens: int = 0
    steps: int = 0
    gen_tokens: int = 0
    slot_steps: int = 0      # Σ 每步仍在跑的槽位数 → 槽位利用率
    lats: list = dataclasses.field(default_factory=list)


def pct(xs: list, q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Engine:
    """模型 + 大 cache（slots 个槽位）+ 捕获好的 decode 图。"""

    def __init__(self, model, tok, slots: int, cap: int):
        self.model = model
        self.tok = tok
        self.slots = slots
        self.cap = cap
        self.dev = next(model.parameters()).device

        self.cache = StaticCache(model.config, max_cache_len=cap)
        self._allocate()

        self.ids = torch.zeros(slots, 1, dtype=torch.long, device=self.dev)
        # 固定长度 mask（0 = 屏蔽）与逐行 position_ids：两者都是静态缓冲，
        # 形状不变才能在 CUDA Graph 里原地改。左填充靠它们被彻底屏蔽
        # （_dbg_mask3.py 已验证：换填充文本结果 max|diff| = 0）。
        self.mask = torch.zeros(slots, cap, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(slots, 1, dtype=torch.long, device=self.dev)
        self.g = None
        self.capture_ms = 0.0

    # -- 分配：跑一次 dummy forward 把 lazy_initialization 挤出来，然后清零 ------
    def _allocate(self):
        with torch.no_grad():
            dummy = torch.zeros(self.slots, 1, dtype=torch.long, device=self.dev)
            self.model(input_ids=dummy, cache=self.cache, use_cache=True)
        self.cache.reset()
        self.n_layers = len(self.cache.layers)
        self.heads = self.cache.layers[0].keys.shape[1]
        self.hd = self.cache.layers[0].keys.shape[-1]

    def _set_level(self, level: int):
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(level)

    def capture(self):
        st_ids = self.ids
        warm = StaticCache(self.model.config, max_cache_len=self.cap)
        with torch.no_grad():
            for _ in range(3):
                self.model(input_ids=st_ids, cache=warm, use_cache=True,
                           attention_mask=torch.ones(self.slots, self.cap,
                                                     dtype=torch.long, device=self.dev))
        torch.cuda.synchronize()
        del warm
        g = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(g):
            with torch.no_grad():
                out = self.model(input_ids=st_ids, cache=self.cache, use_cache=True,
                                 attention_mask=self.mask, position_ids=self.pos)
                st_ids.copy_(out.logits[:, -1, :].argmax(dim=-1, keepdim=True))
        torch.cuda.synchronize()
        self.capture_ms = (time.perf_counter() - t0) * 1000
        self.g = g

    # -- 一次 prefill 放若干条请求进各自槽位（批量化，省掉每条的固定开销）------
    def admit_many(self, reqs: list[Req], level: int):
        """把一批请求一起 prefill，再把每行的 KV 拷进大 cache 的对应槽位。

        **不需要左填充**：mask 是逐行的，一行只 attend 自己 mask=1 的位置，
        所以 prompt 就放在 KV 的 0..plen-1，中间和尾部留空洞（mask=0）即可。
        这条是 _dbg_mask3 验证的直接推论 —— 填充本来只是"没有逐行 mask 时的替代品"。
        代价：attention 仍会扫过 [0, level) 整段（masked 项跳过，但 QK^T 照算），
        所以稀疏的是**显存里的空洞**，不是计算。

        为什么要成批 prefill：eager prefill 每条要付约 26 ms 的固定开销
        （Python dispatch + kernel launch），而 130 token 的真实计算只要 1.7 ms。
        96 条请求若逐条 prefill，光开销就 2.5 s —— 占整轮墙钟的 2/3。
        """
        k = len(reqs)
        L = max(r.plen for r in reqs)
        ids = torch.zeros(k, L, dtype=torch.long, device=self.dev)
        pmask = torch.zeros(k, self.cap, dtype=torch.long, device=self.dev)
        ppos = torch.zeros(k, L, dtype=torch.long, device=self.dev)
        for i, r in enumerate(reqs):
            ids[i, L - r.plen:] = torch.tensor(r.ids, device=self.dev)
            pmask[i, L - r.plen:L] = 1
            ppos[i, L - r.plen:] = torch.arange(r.plen, device=self.dev)

        small = StaticCache(self.model.config, max_cache_len=self.cap)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(input_ids=ids, cache=small, use_cache=True,
                             attention_mask=pmask, position_ids=ppos)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        nxt = out.logits[:, -1, :].argmax(dim=-1)          # (k,)
        for i, r in enumerate(reqs):
            sid, plen = r.slot, r.plen
            for li in range(self.n_layers):
                src, dst = small.layers[li], self.cache.layers[li]
                dst.keys[sid, :, :plen, :] = src.keys[i, :, L - plen:L, :]
                dst.values[sid, :, :plen, :] = src.values[i, :, L - plen:L, :]
            self.ids[sid, 0] = nxt[i]
            self.mask[sid].zero_()
            self.mask[sid, :plen] = 1
            self.pos[sid, 0] = plen          # 下一个要喂的 token 的位置正是 plen
        del small
        gaps = sum(max(0, level - r.plen) for r in reqs)
        return dt, gaps


def build_requests(tok, n: int, pmin: int, pmax: int, newens: list) -> list[Req]:
    """prompt 长度 4 档轮转、max_new 4 档轮转，内容互不共享前缀。"""
    lens = [pmin, (pmin + pmax) // 2, pmax - 32, pmax]
    words = FILLER.split()
    reqs = []
    for i in range(n):
        L = lens[i % len(lens)]
        off = (i * 37) % max(1, len(words) - L)
        text = " ".join(words[off:off + L])
        ids = tok(text, return_tensors="pt").input_ids[0][:L].tolist()
        reqs.append(Req(rid=i, plen=len(ids), max_new=newens[i % len(newens)], ids=ids))
    return reqs


def reference_tokens(model, req: Req) -> list:
    """不填充、不分批：纯 eager + DynamicCache 的贪心参考（正确性基准）。"""
    dev = next(model.parameters()).device
    ids = torch.tensor(req.ids, device=dev).view(1, -1)
    out_tokens = []
    cache = DynamicCache()
    with torch.no_grad():
        out = model(ids, cache=cache)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        out_tokens.append(int(nxt.item()))
        for _ in range(req.max_new - 1):
            out = model(nxt, cache=cache)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            out_tokens.append(int(nxt.item()))
    return out_tokens


def check_tokens(model, reqs: list[Req]) -> int:
    """逐条比对，返回不一致的条数。"""
    bad = 0
    for r in reqs:
        ref = reference_tokens(model, r)
        if len(r.tokens) != len(ref) or r.tokens != ref:
            bad += 1
            if bad <= 3:
                k = next((i for i, (a, b) in enumerate(zip(r.tokens, ref)) if a != b), -1)
                print(f"  [x] rid={r.rid} plen={r.plen} max_new={r.max_new} "
                      f"首个分叉步={k}  ours={r.tokens[:8]} ref={ref[:8]}")
    print(f"  对拍：{len(reqs) - bad}/{len(reqs)} 条逐位一致"
          + ("" if bad == 0 else f"  **{bad} 条不一致**"))
    return bad


def run(engine: Engine, reqs: list[Req], slots: int, policy: str,
        verify: bool = False) -> Stat:
    st = Stat(name=policy)
    if verify:                             # 同一批 Req 会被多个模式复用，必须清干净
        for r in reqs:
            r.tokens.clear()
            r.gen = 0
            r.done_ms = 0.0
            r.admit_ms = 0.0
    # 水位只需覆盖"第一波最长的 prompt"：之后每步写在 level 处，
    # 谁也不会覆盖自己 prompt 的那 plen 个 KV。mask 是逐行的，所以不必填充对齐。
    level = max(r.plen for r in reqs[:slots])
    assert level + max(r.max_new for r in reqs) <= engine.cap, "CAP 不够"

    waiting = list(reqs)
    running: list[Req] = []
    free = list(range(slots))
    engine._set_level(level)
    t0_all = time.perf_counter()

    def admit_batch(cands):
        # 一次 prefill 放完这一批（省掉逐条的 ~26 ms 固定开销），再各自记账
        for r in cands:
            r.slot = free.pop(0)
            r.admit_ms = (time.perf_counter() - t0_all) * 1000
        dt, gaps = engine.admit_many(cands, level)
        st.prefill_ms += dt
        st.pad_tokens += gaps
        for r in cands:
            r.gen = 1                      # prefill 已经给出第 1 个 token
            if verify:
                r.tokens.append(int(engine.ids[r.slot, 0].item()))
            running.append(r)

    admit_batch(waiting[:slots])
    waiting = waiting[slots:]

    while running:
        torch.cuda.synchronize()
        if os.environ.get("M2B_DEBUG") and st.steps < 3:
            print(f"[dbg] step={st.steps} level={level} "
                  f"cum={engine.cache.layers[0].cumulative_length.item()} "
                  f"ids={engine.ids[:, 0].tolist()} "
                  f"running={[(r.rid, r.slot, r.gen, r.max_new) for r in running]}")
        t0 = time.perf_counter()
        # 进图之前把这一行要写的位置打开、位置编号给对（都是原地改，形状不变）
        for r in running:
            engine.mask[r.slot, level] = 1
            engine.pos[r.slot, 0] = r.plen + r.gen - 1
        engine.g.replay()
        torch.cuda.synchronize()
        st.steps += 1
        st.slot_steps += len(running)
        level += 1                          # update() 里已把 cumulative_length +1
        if level > engine.cap:
            raise RuntimeError(f"水位 {level} 超过 CAP {engine.cap} —— "
                               f"index_copy_ 会越界，调大 cap 公式")

        done = []
        if verify:
            for r in running:
                r.tokens.append(int(engine.ids[r.slot, 0].item()))
        for r in running:
            r.gen += 1
            if r.gen >= r.max_new:
                r.done_ms = (time.perf_counter() - t0_all) * 1000
                done.append(r)
        for r in done:
            running.remove(r)
            free.append(r.slot)
            st.gen_tokens += r.gen

        if waiting and free:
            k = min(len(free), len(waiting))
            if policy == "continuous" or not running:
                if not running:
                    # 一批跑完了：水位可以重设（新一批会重写 0..level-1，旧 KV 不会被读到）。
                    # 不重设的话 level 会跨波单调上涨，后面每一行都要多读前几波的 KV。
                    lvl = max(r.plen for r in waiting[:k])
                    if lvl < level:
                        level = lvl
                        engine._set_level(level)
                free.sort()
                admit_batch(waiting[:k])
                waiting = waiting[k:]

    st.wall_ms = (time.perf_counter() - t0_all) * 1000
    st.lats = [r.e2e_ms for r in reqs]
    st.steps = st.steps
    return st


def report(rows: list[Stat], slots: int):
    base = rows[0]
    print()
    print("=" * 104)
    print(f"{'模式':<12}{'墙钟ms':>10}{'tok/s':>9}{'相对':>7}{'槽位利用':>11}"
          f"{'填充token':>11}{'prefill ms':>12}{'解码步数':>10}")
    print("-" * 104)
    for s in rows:
        util = s.slot_steps / (s.steps * slots) * 100 if s.steps else 0.0
        print(f"{s.name:<12}{s.wall_ms:>10.1f}{s.gen_tokens / s.wall_ms * 1000:>9.1f}"
              f"{base.wall_ms / s.wall_ms:>7.2f}{util:>10.1f}%"
              f"{s.pad_tokens:>11}{s.prefill_ms:>12.1f}{s.steps:>10}")
    print("-" * 104)
    for s in rows:
        print(f"{s.name:<12} e2e  p50 {pct(s.lats, .5):7.1f}  p95 {pct(s.lats, .95):7.1f}"
              f"  p99 {pct(s.lats, .99):7.1f}  max {max(s.lats):7.1f} ms")
    print("=" * 104)
    floor = W_MB / 1024 / PEAK_BW * 1000
    print(f"参考：权重地板 {W_MB:.0f} MB / {PEAK_BW:.0f} GB/s = {floor:.2f} ms/步（不含 KV）")
    print("注：槽位利用率 = Σ(每步在跑的槽位数) / (步数 × 槽位总数)；wave 组低说明槽位在空转。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=int, default=96)
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--pmin", type=int, default=48)
    ap.add_argument("--pmax", type=int, default=256)
    ap.add_argument("--newens", default="8,16,24,32")
    ap.add_argument("--modes", default="wave,continuous")
    ap.add_argument("--verify", action="store_true",
                    help="每步把生成的 token 读回主机并逐条对拍（会破坏计时，只用于查正确性）")
    ap.add_argument("--out", default="", help="把结果写成 JSON（画图用）")
    args = ap.parse_args()

    newens = [int(x) for x in args.newens.split(",")]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    model.eval()

    # CAP 要覆盖【整轮】的水位上涨，不只是第一波：
    #   连续补位时 level 单调涨，总步数 ≈ Σmax_new/slots + 尾部长；
    #   wave 模式下跨波也会继续涨（只有一批全空时才允许重设水位）。
    waves = -(-len(range(args.reqs)) // args.slots)          # ceil(reqs/slots)
    cap = args.pmax + waves * max(newens) + 4
    reqs = build_requests(tok, args.reqs, args.pmin, args.pmax, newens)
    print(f"{len(reqs)} 条请求  prompt={args.pmin}..{args.pmax}  max_new∈{newens}  "
          f"槽位={args.slots}  CAP={cap}（{waves} 波）")

    rows = []
    bad_total = 0
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        engine = Engine(model, tok, args.slots, cap)
        engine.capture()
        print(f"[{mode}] 捕获 {engine.capture_ms:.0f} ms  "
              f"KV {engine.n_layers} 层 × {engine.heads} head × {engine.hd}")
        st = run(engine, reqs, args.slots, mode, verify=args.verify)
        st.capture_ms = engine.capture_ms
        rows.append(st)
        print(f"[{mode}] 完成 墙钟 {st.wall_ms:.1f} ms  步数 {st.steps}")
        if args.verify:
            bad_total += check_tokens(model, reqs)
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    report(rows, args.slots)
    if args.out:
        import json
        data = {
            "config": {
                "reqs": len(reqs), "slots": args.slots, "cap": cap,
                "pmin": args.pmin, "pmax": args.pmax, "newens": newens,
                "model": MODEL,
            },
            "rows": [{
                "name": s.name,
                "wall_ms": s.wall_ms,
                "tok_s": s.gen_tokens / s.wall_ms * 1000,
                "gen_tokens": s.gen_tokens,
                "steps": s.steps,
                "util": s.slot_steps / (s.steps * args.slots) if s.steps else 0.0,
                "prefill_ms": s.prefill_ms,
                "gaps": s.pad_tokens,
                "capture_ms": s.capture_ms,
                "p50": pct(s.lats, .5), "p95": pct(s.lats, .95),
                "p99": pct(s.lats, .99), "max": max(s.lats),
            } for s in rows],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写入 {args.out}")
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
