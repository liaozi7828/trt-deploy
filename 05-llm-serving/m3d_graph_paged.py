"""M3d：把整个「分页 decode 一步」捕获成一张 CUDA Graph。

为什么这一步只能在分页之后做
--------------------------
CUDA Graph 要求**所有张量形状恒定**。M2 的 dense 缓存必须维护一个全局水位
`level`（每步 +1），形状就跟着变，图捕获不了。分页之后：

| 张量              | 每步变化        | 能否进图 |
|-------------------|-----------------|----------|
| ids / pos         | 值变、形状不变  | ✅ 原地写 |
| block_table       | 只在换请求时变  | ✅        |
| seq               | 值变、形状不变  | ✅        |
| KV 池             | 全部预分配      | ✅        |

所以本脚本要回答的问题是：**分页把「形状固定」这件事变成可能之后，
一步到底能快到多少？**

做法
----
1. prefill 仍走 HF 原路径（逐条），KV 拷进分页池（复用 m3c 的代码）。
2. 整步写成纯原地操作：seq += 1 → forward → argmax 写 tok → ids ← tok → pos += 1。
3. 先用 eager 跑一段（同进程基线），再 `torch.cuda.graph` 捕获同一段，replay 计时。
4. token 与 HF 原路径逐位对拍（对齐：我的第 k 步对应参考的第 k+1 步）。
"""
from __future__ import annotations

import argparse
import statistics
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3b_paged_attn import paged_attn_triton
from m3c_qwen_paged import (FILLER, MODEL, build_pool, head_dim_of, patch,
                            prefill_and_fill, reference, unpatch)

W_MB = 942.0          # 0.5B 的 fp16 权重体积（GB/s 换算用）
PEAK_BW = 334.0       # m0b 实测等效带宽


# ----------------------------------------------------------------------------
def kv_bytes(cfg, seq: torch.Tensor) -> tuple[float, float]:
    """返回 (按需 MiB, dense 等价 MiB)。"""
    per_tok = 2 * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim_of(cfg) * 2
    used = int(seq.sum().item()) * per_tok
    dense = int(seq.numel()) * int(seq.max().item()) * per_tok
    return used / 2**20, dense / 2**20


def reset_state(seq, pos, ids, first_ids, back: int):
    """把 seq/pos/ids 退回到 back 步之前的状态。

    分页池里的 KV 不用清：attention 读的范围由 `seq` 决定，
    退回去之后再前进会在同样的位置重写同样的内容。
    """
    with torch.no_grad():
        seq.sub_(back)
        pos.sub_(back)
        ids.copy_(first_ids)


# ----------------------------------------------------------------------------
@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--steps", type=int, default=16, help="对拍步数")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--pmin", type=int, default=64)
    ap.add_argument("--pmax", type=int, default=192)
    ap.add_argument("--eager-steps", type=int, default=10, help="eager 基线步数")
    ap.add_argument("--replay", type=int, default=50, help="图 replay 计时步数")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda", attn_implementation="eager",
        local_files_only=True)
    model.eval()
    cfg = model.config
    print(f"{MODEL}  layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"kv_heads={cfg.num_key_value_heads} d={head_dim_of(cfg)}  rows={args.rows}")

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    full = tok(FILLER, return_tensors="pt").input_ids[0].tolist()
    prompts = []
    for r in range(args.rows):
        L = args.pmin + (args.pmax - args.pmin) * (r % 4) // 3
        prompts.append(full[r * 13: r * 13 + L])
    lens = [len(p) for p in prompts]
    print(f"prompt 长度 {min(lens)}..{max(lens)}  （4 档轮转）")

    # 参考答案：HF 原路径逐条
    ref, ref_lg = reference(model, prompts, args.steps)
    print(f"参考 token 前 3 步（第 0 行）: {ref[0, :3].tolist()}")

    cap = max(lens) + args.steps + args.replay + 8
    pool = build_pool(cfg, args.rows, cap, args.block)
    first_ids = prefill_and_fill(model, pool, prompts, args.rows, args.block).view(-1, 1)
    print(f"首 token 与参考一致: {first_ids[:, 0].tolist() == ref[:, 0].tolist()}")

    # 静态缓冲：整步只改这些张量的内容，绝不重建
    seq, pos = pool[3], torch.tensor(lens, dtype=torch.long, device="cuda").view(-1, 1)
    ids = first_ids.clone()
    tokbuf = torch.zeros(args.rows, 1, dtype=torch.long, device="cuda")

    st = {"pk": pool[0], "pv": pool[1], "bt": pool[2], "seq": seq,
          "bs": args.block, "pos": pos.view(-1, 1), "attn": paged_attn_triton}
    saved = patch(model, st)

    def step():
        """一整步 decode，全部原地操作 —— 这是要被捕获的东西。"""
        seq.add_(1)                                   # 新 token 落在 seq-1
        out = model(input_ids=ids, position_ids=pos, use_cache=False,
                    attention_mask=None)
        tokbuf.copy_(out.logits[:, -1, :].argmax(-1, keepdim=True))
        ids.copy_(tokbuf)
        pos.add_(1)

    # ---- 1) eager 基线（同进程、同张量） ----
    for _ in range(3):                                # 热身：Triton JIT 必须在此之前
        step()
    torch.cuda.synchronize()
    reset_state(seq, pos, ids, first_ids, 3)

    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(args.eager_steps):
        step()
    ev1.record()
    torch.cuda.synchronize()
    eager_ms = ev0.elapsed_time(ev1) / args.eager_steps
    reset_state(seq, pos, ids, first_ids, args.eager_steps)
    print(f"eager  每步 {eager_ms:8.3f} ms   {args.rows / (eager_ms / 1000):8.1f} tok/s")

    if args.no_graph:
        unpatch(saved)
        return 0

    # ---- 2) 捕获 ----
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g):
            step()
    except Exception as e:
        print(f"[FAIL] 捕获失败：{type(e).__name__}: {e}")
        unpatch(saved)
        return 1
    torch.cuda.synchronize()
    print("捕获成功")

    # ---- 3) 对拍：replay + 逐步把 tok 抄进 hist ----
    hist = torch.zeros(args.steps, args.rows, dtype=torch.long, device="cuda")
    for i in range(args.steps):
        g.replay()
        hist[i].copy_(tokbuf[:, 0])
    torch.cuda.synchronize()
    mine = hist.transpose(0, 1)                       # (rows, steps)
    ok = bool(torch.equal(mine[:, :-1], ref[:, 1:]))
    step_ok = (mine[:, :-1] == ref[:, 1:]).all(dim=0).sum().item()
    bad = (mine[:, :-1] != ref[:, 1:]).any(dim=1).nonzero().flatten().tolist()
    print(f"图路径 token 逐位={ok}  全程一致的步={step_ok}/{args.steps - 1}  不一致的行={bad}")
    print(f"  第 0 行 mine {mine[0, :5].tolist()}")
    print(f"  第 0 行 ref  {ref[0, 1:6].tolist()}")

    # ---- 4) 计时 ----
    reset_state(seq, pos, ids, first_ids, args.steps)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(args.replay):
        g.replay()
    ev1.record()
    torch.cuda.synchronize()
    graph_ms = ev0.elapsed_time(ev1) / args.replay

    used, dense = kv_bytes(cfg, seq)
    print(f"图路径 每步 {graph_ms:8.3f} ms   {args.rows / (graph_ms / 1000):8.1f} tok/s"
          f"   相对 eager {eager_ms / graph_ms:.2f}×")
    print(f"每步读权重下界 {W_MB / 1024 / PEAK_BW * 1000:.2f} ms"
          f"  → 本步是它的 {graph_ms / (W_MB / 1024 / PEAK_BW * 1000):.1f} 倍")
    print(f"KV 显存：按需 {used:.1f} MiB / dense 等价 {dense:.1f} MiB")

    unpatch(saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
