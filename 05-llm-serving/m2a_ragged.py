#!/usr/bin/env python3
"""m2a_ragged.py — M2 第一步：变长请求的代价（三种跑法，全部上 CUDA Graph）

为什么必须先测这一步
--------------------
M1/m0f 的所有数字都建在一个不成立的前提上：24 条请求长度相同、生成步数相同。
真实服务里 prompt 长度和服务时长都不同，一上变长，两个地方开始漏钱：

  1. 短请求被长请求拖着走（凑齐一个 batch 要等最慢的那条）
  2. 为凑成定长要 padding，padding 出来的 KV 每步都要被读一遍

本脚本不引入 continuous batching（那是 m2b），只做三件事：
  ① 量出 serial（不 batching）和 pad_max（朴素 batching）的差距
  ② 量出 bucket（按长度分桶）到底救回来多少
  ③ 把「吞吐」和「延迟」分开记 —— 这两个数在变长场景下会背离

三种模式
--------
  serial   : 一条一条跑，batch=1，每条一个 shape → 每条一张图
  pad_max  : 全塞进一个 batch，prompt 补到组内最长，跑到最慢那条结束
  bucket   : 按长度排序后每 8 条一组，组内补到组内最长，组间顺序跑

为什么必须用 CUDA Graph
----------------------
eager 的 ITL 是 19.5 ms，其中 17 ms 是 Python 开销。padding 多读的那点 KV
完全淹没在里面 —— 用 eager 测，pad_max 和 bucket 会显示成"一样快"，
因为两者都被同一个 CPU 地板压着。只有把 CPU 地板拿掉，调度策略的差异才看得见。
这和 P1 的教训是同一个：不先把瓶颈换成可观测的那个，就测不出想测的东西。

padding 怎么做（以及为什么不做全程 token 对拍）
-----------------------------------------------
左填充会破坏"最后一个位置一定是真 token"的前提，右填充会让新 token 写进 pad
后面。正确做法是传 attention_mask，但 HF 的 mask 长度必须等于当前 cache_position+1
（内部按 target_length 展开），长度每步都变 → 又回到"形状不固定"，图捕获不了。
所以这里**用重复真 token 来填充**，完全不用 mask：
  - 行内因果注意力的结果只依赖本行自己的 KV，与 batch 里其他行无关
  - 因此"没有填充的那些行"结果必须和 serial 逐位一致
  - 这条就是本脚本的正确性判据（填充行结果不对，只拿它测时间）

用法
----
    ~/.venvs/torch/bin/python -u m2a_ragged.py 2>&1 | tee ~/m2a.log
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import warnings
from dataclasses import dataclass

import torch

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda"

LENS = (48, 96, 160, 256)      # prompt 长度
NEWS = (8, 16, 24, 32)         # 生成长度
PER_LEN = 6                    # 每个长度 6 条 → 共 24 条


# ---------------------------------------------------------------------------
class Req:
    def __init__(self, rid: int, prompt_len: int, max_new: int, ids: torch.Tensor) -> None:
        self.rid = rid
        self.prompt_len = prompt_len
        self.max_new = max_new
        self.ids = ids
        self.tokens: list[int] = []
        self.latency_ms: float = 0.0   # 自己跑完用了多久（不含排队）
        self.e2e_ms: float = 0.0       # 提交→返回（含排队）—— serving 真正该看的
        self.t_submit: float = 0.0
        self.was_padded = False


@dataclass
class Stat:
    wall_ms: float = 0.0
    capture_ms: float = 0.0   # 图捕获（启动成本，不计入稳态）
    upload_ms: float = 0.0     # 捕获后首次 replay 的图上传（启动成本）
    prefill_ms: float = 0.0    # 本组 prefill 耗时（eager，单次）
    slot_steps: int = 0        # 分配的 token-步（batch × 步数），含浪费
    pad_tokens: int = 0        # 注入的填充 token 数（会影响 KV 读取量）
    steps: int = 0             # decode 步数（算 ITL 用）
    lats: list = None          # 各请求延迟快照（不能共享同一个 list！）

    def __post_init__(self) -> None:
        if self.lats is None:
            self.lats = []

    def __iadd__(self, o: "Stat") -> "Stat":
        self.wall_ms += o.wall_ms
        self.capture_ms += o.capture_ms
        self.upload_ms += o.upload_ms
        self.prefill_ms += o.prefill_ms
        self.slot_steps += o.slot_steps
        self.pad_tokens += o.pad_tokens
        self.steps += o.steps
        self.lats.extend(o.lats)
        return self


# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
             .to(DEV).eval())
    return tok, model


def build_reqs(tok) -> list[Req]:
    """用一段长文本切出不同起点、不同长度的 prompt。

    每条偏移量不同 → 相邻请求不共享前缀。这一条是给 M5（prefix caching）埋的伏笔：
    如果用同一段文本的连续切片，prefix caching 会白捡命中率，测出来的收益是假的。
    """
    text = ("The quick brown fox jumps over the lazy dog. "
            "A journey of a thousand miles begins with a single step. ") * 400
    base = tok(text, return_tensors="pt").input_ids[0]

    reqs: list[Req] = []
    for i in range(PER_LEN * len(LENS)):
        plen = LENS[i % len(LENS)]
        mnew = NEWS[(i * 3) % len(NEWS)]        # 让长度与生成长度交叉组合
        off = (i * 37) % 500
        ids = base[off: off + plen].clone()
        assert ids.numel() == plen, f"切片不够长：要 {plen}，给 {ids.numel()}"
        reqs.append(Req(i, plen, mnew, ids))
    return reqs


# ---------------------------------------------------------------------------
def make_static_cache(model, max_cache_len: int):
    """v5 的 StaticCache 签名：(config, max_cache_len, offloading, ...) —— 已无 max_batch_size。"""
    from transformers import StaticCache

    return StaticCache(config=model.config, max_cache_len=max_cache_len)


class DecodeGraph:
    """把"一次 decode step"捕获成一张图。

    输入只有两个静态缓冲：ids (B,1) 和 pos (1,)。cache 在捕获期就固定了地址，
    replay 时 KV 写进同一块显存 —— 这正是我们要的（形状恒定）。
    """

    def __init__(self, model, cache, cache_len: int, batch: int) -> None:
        self.model = model
        self.cache = cache
        self.ids = torch.zeros(batch, 1, dtype=torch.long, device=DEV)
        self.pos = torch.zeros(1, dtype=torch.long, device=DEV)

        # 热身必须用一次性 cache：热身会真的写 KV，用同一个会把 cache 长度推乱
        # （m0e 第一版就漏了这一点）
        warm = make_static_cache(model, cache_len)
        with torch.no_grad():
            for _ in range(2):
                model(input_ids=self.ids, cache_position=self.pos,
                      cache=warm, use_cache=True)
        del warm
        gc.collect()
        torch.cuda.synchronize()

        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.logits = model(input_ids=self.ids, cache_position=self.pos,
                                cache=self.cache, use_cache=True).logits

    def step(self, tok_ids: torch.Tensor, position: int) -> torch.Tensor:
        self.ids.copy_(tok_ids)
        self.pos.fill_(position)
        self.g.replay()
        return self.logits

# ---------------------------------------------------------------------------
def _upload_ms(step_ms: list[float]) -> float:
    """从每步累计耗时里估出"第一次 replay 的图上传"成本。

    捕获后的第一次 replay 要额外付一次图上传（实测 batch 1 约 70 ms、batch 24 约
    280 ms，视图大小），它只落在第一个 decode 间隔 d[0] 上。用后面所有间隔的中位数
    当稳态单价，超出部分就是上传成本。

    为什么用中位数：decode 间隔本身有抖动，平均会被上传那一格拉偏。
    为什么不在捕获后主动 replay 一次把它"挤掉"：试过了 —— StaticCache 的
    `cumulative_length` 单调累加，多跑一次就 `index_copy_(): index out of bounds`。
    同一块 StaticCache 只能按自然顺序走一遍。
    """
    d = [step_ms[i] - step_ms[i - 1] for i in range(1, len(step_ms))]
    if len(d) < 3:
        return 0.0                      # 步数太少估不出来，就不修正（宁可算进延迟）
    med = statistics.median(d[1:])      # 排除 d[0]：它是唯一被污染的那一格
    return max(0.0, d[0] - med)


# 累计的"构建期成本"（图捕获 + 每组清理）。端到端延迟要把这部分扣掉，
# 否则串行模式会被 24 次捕获（每次约 240 ms）算成排队时间。
_STARTUP = 0.0


def run_group(model, reqs: list[Req], pad_len: int) -> Stat:
    """跑一组（batch = len(reqs)），把结果与延迟写回 Req，返回本组统计。

    顺序：prefill（eager）→ 捕获 decode 图 → decode 循环。
    prefill 必须在捕获之前：StaticCache 是懒初始化的，第一次 `update()` 里做
    `cumulative_length.to(device)`，这一行落进捕获窗口就会报
    `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the
    CPU tensor is pinned`。
    捕获成本（~300 ms/图）与每组清理单独计进 `capture_ms`/`_STARTUP`，
    不进任何延迟数字。
    """
    global _STARTUP
    s0 = _STARTUP                       # 本组开始前已累计的构建期成本（秒）
    batch = len(reqs)
    max_new = max(r.max_new for r in reqs)
    cache_len = pad_len + max_new

    # 左填充：整行先填成该请求的第一个真 token，再把真 token 放到右端
    ids = torch.empty(batch, pad_len, dtype=torch.long, device=DEV)
    for b, r in enumerate(reqs):
        ids[b, :] = r.ids[0]
        ids[b, pad_len - r.prompt_len:] = r.ids
        r.was_padded = r.prompt_len != pad_len
    pos = torch.arange(pad_len, dtype=torch.long, device=DEV)
    pad_tokens = sum(pad_len - r.prompt_len for r in reqs)

    cache = make_static_cache(model, cache_len)
    tok_buf = torch.empty(max_new, batch, dtype=torch.long, device=DEV)
    step_ms = [0.0] * max_new

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, cache_position=pos, cache=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1)
    tok_buf[0] = nxt
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    step_ms[0] = prefill_ms

    t_cap = time.perf_counter()
    dg = DecodeGraph(model, cache, cache_len, batch)
    capture_ms = (time.perf_counter() - t_cap) * 1000.0

    # 位置要对上：prefill 占了 0..pad_len-1，它吐出的第一个 token 属于位置 pad_len，
    # 所以 step=1 这次前向的 cache_position 是 pad_len（不是 pad_len+1）。
    step_ids = torch.empty(batch, 1, dtype=torch.long, device=DEV)
    t1 = time.perf_counter()
    for step in range(1, max_new):
        step_ids[:, 0] = nxt
        logits = dg.step(step_ids, pad_len + step - 1)
        nxt = logits[:, -1, :].argmax(dim=-1)
        tok_buf[step] = nxt
        # 以 prefill 结束为基准累加，捕获成本自然被排除在这一段之外
        step_ms[step] = prefill_ms + (time.perf_counter() - t1) * 1000.0

    torch.cuda.synchronize()
    got = tok_buf.cpu().tolist()        # 整个生成只在这里同步一次

    up = _upload_ms(step_ms)
    step_c = [t - up for t in step_ms]
    wall_ms = step_c[-1]

    now = time.perf_counter()
    for b, r in enumerate(reqs):
        r.tokens = [got[s][b] for s in range(r.max_new)]
        r.latency_ms = step_c[r.max_new - 1]
        # 端到端由 mode_* 用"work-only 时间线"填（见 mode_serial 的注释）
        r.e2e_ms = r.latency_ms

    # 清理也算构建期成本：empty_cache 在 WSL 上每次要几十到几百 ms
    t_td = time.perf_counter()
    del dg, cache
    gc.collect()
    torch.cuda.empty_cache()
    _STARTUP += capture_ms / 1000.0 + (time.perf_counter() - t_td)

    return Stat(wall_ms=wall_ms, capture_ms=capture_ms, upload_ms=up,
                prefill_ms=prefill_ms, slot_steps=batch * max_new,
                pad_tokens=pad_tokens, steps=max_new - 1,
                lats=[r.latency_ms for r in reqs])


# ---------------------------------------------------------------------------
def _submit_all(reqs: list[Req], t0: float) -> None:
    """把所有请求的提交时刻都钉在 t0，并清零构建期成本的累计。

    这是 serving 的真实语义：请求一起到，排队时间必须算进端到端延迟。
    如果按"轮到它了才算提交"，串行模式的排队时间会凭空消失。
    """
    global _STARTUP
    _STARTUP = 0.0
    for r in reqs:
        r.t_submit = t0


def mode_serial(model, reqs: list[Req]) -> Stat:
    """一条一条跑。端到端 = 前面所有请求的墙钟 + 自己跑的时间。

    用"work-only 时间线"算，不读墙钟：这样图捕获、empty_cache 这些构建期成本
    不会混进排队时间（它们一次性的，真实 serving 也只在启动时付一次）。
    """
    _submit_all(reqs, time.perf_counter())
    st = Stat()
    acc = 0.0
    for r in reqs:
        g = run_group(model, [r], r.prompt_len)
        r.e2e_ms = acc + r.latency_ms
        acc += g.wall_ms
        st += g
    return st


def mode_pad_max(model, reqs: list[Req]) -> Stat:
    """全部同时开跑，所以每条都在自己跑完时返回 —— 端到端 = 自跑时长。"""
    _submit_all(reqs, time.perf_counter())
    st = run_group(model, reqs, max(r.prompt_len for r in reqs))
    for r in reqs:
        r.e2e_ms = r.latency_ms
    return st


def mode_bucket(model, reqs: list[Req], bucket: int) -> Stat:
    """按长度排序再切组 —— 组内长度差最小，padding 浪费最小。

    代价在这里显形：第 i 组的请求要等前面 i 组跑完，端到端 = 前面组的墙钟之和
    + 自己跑的时间。分组越细，排队越长 —— 这正是 continuous batching 要消掉的东西。
    """
    ordered = sorted(reqs, key=lambda r: (r.prompt_len, r.max_new))
    _submit_all(ordered, time.perf_counter())
    st = Stat()
    acc = 0.0
    for i in range(0, len(ordered), bucket):
        grp = ordered[i: i + bucket]
        g = run_group(model, grp, max(r.prompt_len for r in grp))
        for r in grp:
            r.e2e_ms = acc + r.latency_ms
        acc += g.wall_ms
        st += g
    return st


# ---------------------------------------------------------------------------
def report(name: str, reqs: list[Req], st: Stat, ref: dict[int, list[int]]) -> None:
    real = sum(len(r.tokens) for r in reqs)
    tps = real / (st.wall_ms / 1000.0)
    lats = sorted(st.lats)
    util = real / st.slot_steps if st.slot_steps else 0.0
    # ITL 只按 decode 段算：墙钟里还含 prefill（大 batch prefill 能到几百 ms，
    # 除进 step 数会把 ITL 抬成假数字）
    dec_ms = max(0.0, st.wall_ms - st.prefill_ms)
    itl = dec_ms / st.steps if st.steps else 0.0
    p99 = lats[int(0.99 * (len(lats) - 1))]

    print(f"\n[{name}]")
    print(f"  墙钟(稳态)  {st.wall_ms:9.1f} ms   捕获(启动) {st.capture_ms:8.1f} ms"
          f"   图上传(启动) {st.upload_ms:7.1f} ms")
    print(f"  每步 ITL    {itl:9.3f} ms   （{st.steps} 个 group-step，只算 decode 段，"
          f"已扣掉首步图上传）")
    print(f"  prefill     {st.prefill_ms:9.1f} ms   （各组 eager prefill 之和）")
    print(f"  真实 token  {real}      有效吞吐 {tps:8.1f} tok/s")
    print(f"  槽位利用率  {util:6.1%}   （真实 token-步 ÷ 分配的 token-步）")
    print(f"  填充 token  {st.pad_tokens}   （每条每步都要多读这些位置的 KV）")
    print(f"  延迟(自跑)  p50 {lats[len(lats) // 2]:8.1f}   p99 {p99:8.1f}   max {lats[-1]:8.1f} ms")
    e2e = sorted(r.e2e_ms for r in reqs)
    e99 = e2e[int(0.99 * (len(e2e) - 1))]
    print(f"  延迟(端到端，含排队) p50 {e2e[len(e2e) // 2]:8.1f}   p99 {e99:8.1f}"
          f"   max {e2e[-1]:8.1f} ms   ← serving 看这个")

    checked = [r for r in reqs if not r.was_padded and ref]
    if checked:
        bad = [r.rid for r in checked if ref.get(r.rid) != r.tokens]
        verdict = "✅ 全部一致" if not bad else f"❌ {len(bad)}/{len(checked)} 不一致"
        print(f"  对拍        无填充的 {len(checked)} 条 → {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, default=8)
    args = ap.parse_args()

    print(f"torch {torch.__version__}   device {torch.cuda.get_device_name(0)}")
    tok, model = load_model()
    reqs = build_reqs(tok)

    w_mb = sum(p.numel() for p in model.parameters()) * 2 / 1024 / 1024
    print(f"权重 {w_mb:.0f} MB   权重地板 {w_mb / 1024 / 334 * 1000:.2f} ms（batch 1，按实测 334 GB/s）")
    print(f"请求 {len(reqs)} 条   prompt {sorted({r.prompt_len for r in reqs})}   "
          f"生成 {sorted({r.max_new for r in reqs})}   总真实 token {sum(r.max_new for r in reqs)}")

    results: dict[str, Stat] = {}
    e2e_by_mode: dict[str, float] = {}

    st = mode_serial(model, reqs)
    results["serial"] = st
    e2e_by_mode["serial"] = max(r.e2e_ms for r in reqs)
    ref = {r.rid: list(r.tokens) for r in reqs}
    report("serial（batch=1，一条一条）", reqs, st, {})      # 自己是参考，不做对拍

    for r in reqs:
        r.tokens, r.latency_ms, r.e2e_ms = [], 0.0, 0.0
    st = mode_pad_max(model, reqs)
    results["pad_max"] = st
    report(f"pad_max（batch={len(reqs)}，补到 {max(r.prompt_len for r in reqs)}）",
           reqs, st, ref)

    e2e_by_mode["pad_max"] = max(r.e2e_ms for r in reqs)

    for r in reqs:
        r.tokens, r.latency_ms, r.e2e_ms = [], 0.0, 0.0
    st = mode_bucket(model, reqs, args.bucket)
    results["bucket"] = st
    report(f"bucket（每 {args.bucket} 条一组，组内补长）", reqs, st, ref)
    e2e_by_mode["bucket"] = max(r.e2e_ms for r in reqs)

    total_real = sum(r.max_new for r in reqs)
    print("\n" + "=" * 84)
    print(f"{'模式':<14}{'墙钟 ms':>11}{'有效 tok/s':>12}{'相对 serial':>12}"
          f"{'max 自跑 ms':>13}{'max 端到端 ms':>15}")
    print("-" * 84)
    base = results["serial"].wall_ms
    for name, st_ in results.items():
        tps = total_real / (st_.wall_ms / 1000.0)
        lats = sorted(st_.lats)
        print(f"{name:<14}{st_.wall_ms:11.1f}{tps:12.1f}{base / st_.wall_ms:11.2f}×"
              f"{lats[-1]:13.1f}{e2e_by_mode[name]:15.1f}")
    print("-" * 84)
    for name, st_ in results.items():
        print(f"  启动成本 {name:<12} 捕获 {st_.capture_ms:7.1f} ms"
              f"  图上传 {st_.upload_ms:6.1f} ms"
              f"   （稳态墙钟不含这两部分）")
    print("=" * 78)
    print("读法：")
    print("  · 吞吐只看「真实 token ÷ 墙钟」，别看分配的槽位数")
    print("  · p99 延迟是变长场景真正的痛点：pad_max 会让 8-token 的请求等到")
    print("    32-token 那条结束才返回 —— 吞吐上去了，单条反而更慢")
    print("  · m2b 加 continuous batching：完成一条立刻补一条进来，目标是不掉吞吐")
    print("    的前提下把延迟压下来")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
