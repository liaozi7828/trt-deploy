#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w3_export_decode.py —— W3 第一步：导出**带 KV cache** 的 decode 专用 ONNX

为什么不能只用一个图（关键，别绕过这一条）：
    gpt.py 里 mask 是 Python 层的条件：
        if past_len == 0:
            att = att.masked_fill(mask[:T,:T] == 0, -inf)
    `past_len` 来自张量尺寸，**导出时会被冻结成追踪那一刻的分支**。
    所以：
      · prefill（past=0, T>1）必须有因果掩码
      · decode （past>0, T=1）必须**没有**掩码
    一个 ONNX 图无法同时满足 → 必须是两个图。
    这也正是 vLLM / TRT-LLM 把 prefill 和 decode 拆成两个引擎的根源。

本脚本产出：
    decode 图：
      输入：input_ids[1,1] + past_k_i[1,H,P,hd] + past_v_i[1,H,P,hd]   (i = 0..L-1)
      输出：logits[1,1,V] + new_k_i[1,H,P+1,hd] + new_v_i[1,H,P+1,hd]
    P 是动态维度（past_len），profile 时单独设 min/opt/max。

验证（这一步比导出更重要）：
    ① 用 ONNXRuntime 跑 decode 图，逐步喂回 cache，跑出 N 个 token 的 logits
    ② 用 PyTorch 做同样的逐步解码（它自己就是"有 cache"的实现）
    ③ 两者必须一致 —— 否则说明 cache 的 in/out 接错了

用法：
    python w3_export_decode.py                      # 导出 + 验证
    python w3_export_decode.py --past-dummy 4       # 换 dummy past_len（一般不用改）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gpt import GPT, DEFAULT_CFG

# 权重的随机种子。w3_export_decode.py 和 w3_cache.py **必须用同一个值**，
# 否则 engine 里是权重 A、参考模型是权重 B，输出会"量级对、数值全错"。
WEIGHT_SEED = 1234


# ---------------------------------------------------------------------------
class DecodeWrapper(nn.Module):
    """把 (cache_list, logits) 的 Python 接口摊平成"一列张量"。

    ONNX / TensorRT 的 IO 都是扁平的，不能收 tuple 列表，
    所以这里用 *flat 把 2L 个 cache 张量摊开，内部再配对。
    """

    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model
        self.n_layer = model.cfg.n_layer

    def forward(self, idx: torch.Tensor, *flat):
        caches = [(flat[2 * i], flat[2 * i + 1]) for i in range(self.n_layer)]
        logits, new = self.model(idx, kv_caches=caches, use_cache=True)
        out = [logits]
        for k, v in new:
            out.append(k)
            out.append(v)
        return tuple(out)


def io_names(n_layer: int) -> tuple[list[str], list[str]]:
    ins = ["input_ids"]
    outs = ["logits"]
    for i in range(n_layer):
        ins += [f"past_k_{i}", f"past_v_{i}"]
        outs += [f"new_k_{i}", f"new_v_{i}"]
    return ins, outs


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="w1_out/gpt_decode.onnx")
    ap.add_argument("--past-dummy", type=int, default=4,
                    help="导出时的 dummy past_len（必须 >0，否则 mask 分支被冻结进去）")
    args = ap.parse_args()

    cfg = DEFAULT_CFG
    L, H, C = cfg.n_layer, cfg.n_head, cfg.n_embd
    hd = C // H
    V = cfg.vocab_size

    # ★★ 权重种子必须固定，且 w3_cache.py 用**同一个**：
    #    GPT(cfg) 是随机初始化的，如果两次构造的种子不同，
    #    engine 里的权重和"标准答案"的权重就是两套模型 —— 数值全错但不报错。
    torch.manual_seed(WEIGHT_SEED)
    model = GPT(cfg).eval()
    wsum = sum(float(p.detach().abs().sum()) for p in model.parameters())
    print(f"权重校验和 : {wsum:.6f}   ← 和 w3_cache.py 打印的必须一致")
    wrapped = DecodeWrapper(model).eval()
    in_names, out_names = io_names(L)

    print("=" * 78)
    print("导出带 KV cache 的 decode ONNX")
    print("=" * 78)
    print(f"配置   : layer={L} head={H} embd={C} head_dim={hd} block={cfg.block_size}")
    print(f"参数量 : {sum(p.numel() for p in model.parameters()):,}")
    print(f"IO     : {len(in_names)} 输入 / {len(out_names)} 输出"
          f"（各含 {2 * L} 个 cache 张量）")

    P = args.past_dummy
    dummy_ids = torch.randint(0, V, (1, 1))
    dummy_caches = []
    for _ in range(L):
        dummy_caches += [torch.randn(1, H, P, hd), torch.randn(1, H, P, hd)]

    # ---- dynamic_axes ----
    # 输入侧的 past 维度叫 past_len；输出侧是 past_len+1，必须用**不同的名字**，
    # 否则 ONNX 会认为两者必须相等（同名字 = 同一个符号）。
    # ★ 三类维度，命名规则不同，很容易搞反：
    #   · batch    ：输入和输出是**同一个值** → 必须用**同名** "batch"
    #   · past_len ：输入是 P，输出是 P+1 → 必须用**不同名**，否则 ONNX 会断言相等
    #   （这和对错刚好相反，是这里最容易出错的地方）
    dyn = {"input_ids": {0: "batch", 1: "seq"}}
    for i in range(L):
        dyn[f"past_k_{i}"] = {0: "batch", 2: "past_len"}
        dyn[f"past_v_{i}"] = {0: "batch", 2: "past_len"}
    for i in range(L):
        dyn[f"new_k_{i}"] = {0: "batch", 2: "total_len"}
        dyn[f"new_v_{i}"] = {0: "batch", 2: "total_len"}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] 导出 ONNX（dummy past_len={P}）")

    # ★ PyTorch 2.9+ 默认走 dynamo 导出器，它依赖 onnxscript；
    #   而 legacy 导出器（dynamo=False）对 dynamic_axes 的支持最稳，
    #   现有的 export_gpt_onnx.py 也是这么做的（先试 legacy，失败再退回）。
    def _export(use_legacy: bool) -> str:
        kw = {"dynamo": False} if use_legacy else {}
        with torch.no_grad():
            torch.onnx.export(
                wrapped,
                (dummy_ids, *dummy_caches),
                str(out_path),
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dyn,
                opset_version=17,
                do_constant_folding=True,
                **kw,
            )
        return "legacy (dynamo=False)" if use_legacy else "dynamo (默认)"

    try:
        backend = _export(True)
    except Exception as e:
        print(f"      legacy 导出器失败: {type(e).__name__}: {str(e)[:150]}")
        print(f"      改用 dynamo（需要 pip install onnxscript）")
        backend = _export(False)
    print(f"      导出器: {backend}")
    print(f"      产出: {out_path} ({out_path.stat().st_size / 1048576:.1f} MB)")

    # ---- 检查图里没有不该有的算子 ----
    import onnx
    m = onnx.load(str(out_path))
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"      节点数: {len(m.graph.node)}")
    print(f"      算子: {ops}")
    if "Range" in ops:
        print("      [WARN] 出现 Range 算子 —— 动态 shape 下 TRT 可能拒绝")

    # ---- ORT 跑通 + 与 PyTorch 逐步解码对拍 ----
    print(f"\n[2/4] ONNXRuntime 加载")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        print(f"      OK，输入 {len(sess.get_inputs())} 个，输出 {len(sess.get_outputs())} 个")
    except Exception as e:
        print(f"      [FAIL] {type(e).__name__}: {str(e)[:200]}")
        print(f"      （TRT 可能会容忍，但没有第三方验证 —— 和 P3 的 FP16 问题同类）")
        return 1

    print(f"\n[3/4] 逐步解码对拍（PyTorch vs ORT，同一个初始 cache）")
    P0 = 16
    N_STEPS = 5
    rng = torch.Generator().manual_seed(2026)
    prompt_ids = torch.randint(0, V, (1, P0), generator=rng)

    with torch.no_grad():
        # PyTorch：先整段 prefill，拿到初始 cache 与首 token 的 logits
        logits_prefill, caches_pt = model(prompt_ids, kv_caches=None, use_cache=True)
        print(f"      prefill {P0} 个 token → logits {tuple(logits_prefill.shape)}，"
              f"cache 长度 {caches_pt[0][0].shape[2]}")

        # 下一 token 的输入：拿 prefill 最后一个位置的 argmax
        next_id = logits_prefill[:, -1, :].argmax(dim=-1, keepdim=True)

        pt_logits = []
        cache_list = caches_pt
        for _ in range(N_STEPS):
            lg, cache_list = model(next_id, kv_caches=cache_list, use_cache=True)
            pt_logits.append(lg[:, -1, :].clone())
            next_id = lg[:, -1, :].argmax(dim=-1, keepdim=True)

        # ORT：从**同一份** cache 开始，同样的 next_id 序列
        next_id = logits_prefill[:, -1, :].argmax(dim=-1, keepdim=True)
        feed_caches = []
        for k, v in caches_pt:
            feed_caches += [k.numpy(), v.numpy()]

        print(f"\n      {'step':>5} {'max|diff|':>12} {'相对误差':>10} {'cache 长度':>11}")
        print("      " + "-" * 44)
        worst = 0.0
        for s in range(N_STEPS):
            feed = {"input_ids": next_id.numpy().astype(np.int64)}
            for i in range(L):
                feed[f"past_k_{i}"] = feed_caches[2 * i]
                feed[f"past_v_{i}"] = feed_caches[2 * i + 1]
            outs = sess.run(None, feed)

            y = outs[0]                                  # (1,1,V)
            d = float(np.abs(y - pt_logits[s].numpy()).max())
            scale = max(float(np.abs(pt_logits[s].numpy()).max()), 1e-9)
            worst = max(worst, d / scale)
            print(f"      {s:5d} {d:12.3e} {d / scale:10.3e} "
                  f"{outs[1].shape[2]:11d}")
            if s + 1 < N_STEPS:
                # 用 ORT 自己算出的 cache 继续下一步（这才是真实的自回归循环）
                feed_caches = []
                for i in range(L):
                    feed_caches += [outs[1 + 2 * i], outs[2 + 2 * i]]
                next_id = torch.from_numpy(y.argmax(axis=-1))

    print(f"\n[4/4] 判定")
    if worst < 1e-3:
        print(f"      [OK] 最大相对误差 {worst:.2e} < 1e-3 —— cache 的 in/out 接对了")
        print(f"           逐步解码 5 步，每一步都和 PyTorch 增量解码一致")
        return 0
    print(f"      [WARN] 相对误差 {worst:.2e} 偏大")
    print(f"             常见原因：past 维度接错、或 mask 分支被冻结进了 decode 图")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
