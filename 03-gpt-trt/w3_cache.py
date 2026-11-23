#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w3_cache.py —— W3：把带 KV cache 的 decode 图建成 TRT 引擎，并逐步解码

这个脚本要回答三个问题：
    ① cache 的 in/out 在 TRT 里能不能跑通（形状 P 每步都变）
    ② **每 token 的延迟**是多少（LLM 服务的核心指标）
    ③ 延迟里有多少是**计算**、多少是**host 侧搬 cache**？

第 ③ 问最重要，也是这个脚本和前面几个的区别：
    朴素的 KV cache 管理是「每步把 old 和 new 拼起来」（torch.cat），
    这个拷贝量是 **O(P)** 的 —— P 越长，每步越慢。
    这正是 PagedAttention 要解决的问题：**不拼，只改指针**。

所以本脚本把每步拆成三段计时：
    [cat]   把上一步的 new_k/new_v 拼进 past      ← O(P) 拷贝
    [setup] set_input_shape × 24 + set_tensor_address × 50   ← host 调用
    [exec]  execute_async_v3 + sync                ← 真正的 GPU 计算

用法：
    python w3_cache.py                         # 建引擎（若不存在）+ 解码 + 计时
    python w3_cache.py --steps 20 --p0 16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

from gpt import GPT, DEFAULT_CFG

# 和 w3_export_decode.py 保持一致的权重种子（见那边的说明）
WEIGHT_SEED = 1234

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
def cache_names(n_layer: int) -> tuple[list[str], list[str]]:
    ins, outs = [], []
    for i in range(n_layer):
        ins += [f"past_k_{i}", f"past_v_{i}"]
        outs += [f"new_k_{i}", f"new_v_{i}"]
    return ins, outs


def build_decode_engine(onnx_path: Path, engine_path: Path,
                        n_layer: int, n_head: int, head_dim: int,
                        p_min: int, p_opt: int, p_max: int) -> trt.ICudaEngine:
    builder = trt.Builder(TRT_LOGGER)
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    except Exception:
        network = builder.create_network()

    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(f"  解析错误: {parser.get_error(i)}")
        raise SystemExit("[FAIL] ONNX 解析失败")
    print(f"  解析成功: {network.num_layers} 层, "
          f"{network.num_inputs} 输入, {network.num_outputs} 输出")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except Exception as e:
        print(f"  [WARN] workspace 设置失败: {e}")

    # ★ past_len 是新的动态维度。24 个 cache 输入**都必须**在同一组 [p_min, p_max] 内。
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (1, 1), (1, 1))
    for name in cache_names(n_layer)[0] + cache_names(n_layer)[1]:
        profile.set_shape(name,
                          (1, n_head, p_min, head_dim),
                          (1, n_head, p_opt, head_dim),
                          (1, n_head, p_max, head_dim))
    config.add_optimization_profile(profile)
    print(f"  profile: input_ids=(1,1)  past_len=[{p_min}, {p_opt}, {p_max}]")

    t0 = time.time()
    print("  正在编译（50 个 IO 张量，会比 prefill 慢一些）...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("[FAIL] 编译失败（把 TRT_LOGGER 改成 INFO 看原因）")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"  编译完成: {engine_path} "
          f"({engine_path.stat().st_size / 1048576:.1f} MB, {time.time() - t0:.1f}s)")

    rt = trt.Runtime(TRT_LOGGER)
    eng = rt.deserialize_cuda_engine(engine_path.read_bytes())
    if eng is None:
        raise SystemExit("[FAIL] 反序列化失败")
    return eng


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="w1_out/gpt_decode.onnx")
    ap.add_argument("--engine", default="w1_out/gpt_decode.engine")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--p0", type=int, default=16, help="起始 cache 长度")
    ap.add_argument("--steps", type=int, default=20, help="解码多少个 token")
    ap.add_argument("--p-max", type=int, default=511)
    args = ap.parse_args()

    cfg = DEFAULT_CFG
    L, H = cfg.n_layer, cfg.n_head
    hd, V = cfg.n_embd // H, cfg.vocab_size

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[FAIL] 找不到 {onnx_path}，先跑 w3_export_decode.py")
        return 1

    engine_path = Path(args.engine)
    if args.rebuild or not engine_path.exists():
        print("=" * 78)
        print("建 decode 引擎")
        print("=" * 78)
        engine = build_decode_engine(onnx_path, engine_path, L, H, hd,
                                     p_min=1, p_opt=64, p_max=args.p_max)
    else:
        rt = trt.Runtime(TRT_LOGGER)
        engine = rt.deserialize_cuda_engine(engine_path.read_bytes())
        print(f"复用已有引擎: {engine_path} "
              f"({engine_path.stat().st_size / 1048576:.1f} MB)")

    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    in_cache, out_cache = cache_names(L)

    # ---- PyTorch 侧：产生初始 cache，并给出"标准答案" ----
    # ★★ 必须和 w3_export_decode.py 用**同一个权重种子**（见那边的说明）
    torch.manual_seed(WEIGHT_SEED)
    model = GPT(cfg).eval()
    wsum = sum(float(p.detach().abs().sum()) for p in model.parameters())
    print(f"权重校验和 : {wsum:.6f}   ← 必须等于 w3_export_decode.py 打印的那个")
    rng = torch.Generator().manual_seed(2026)
    prompt = torch.randint(0, V, (1, args.p0), generator=rng)
    with torch.no_grad():
        lg, caches = model(prompt, kv_caches=None, use_cache=True)
        pt_logits = []
        pt_feed = []          # ★ 每步"喂进去的 token"，用于锁死两条路径的输入
        c = caches
        nid = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(args.steps):
            pt_feed.append(nid.clone())
            lg, c = model(nid, kv_caches=c, use_cache=True)
            pt_logits.append(lg[:, -1, :].clone())
            nid = lg[:, -1, :].argmax(dim=-1, keepdim=True)

    # TRT 侧从**同一份**初始 cache 出发
    # ★★ 必须搬到 GPU：TRT 的 set_tensor_address 要的是**设备指针**。
    #    PyTorch 侧模型在 CPU 上，caches / next_id 全是 host 指针；
    #    直接把 host 指针喂给 TRT 会读到非法地址 → cudaErrorIllegalAddress
    #    （而且报错会在后面的 synchronize 才冒出来，位置偏离真因）。
    cache = [(k.float().cuda().contiguous(), v.float().cuda().contiguous())
             for k, v in caches]
    pt_feed_g = [tok.to(torch.int64).cuda() for tok in pt_feed]
    out_logits = torch.empty((1, 1, V), dtype=torch.float32, device="cuda")
    argmax_match = 0      # TRT 自己的 argmax 和 PyTorch 一致了几步

    print("\n" + "=" * 78)
    print("逐步解码（TRT，每步把 new_* 拼回 past_*）")
    print("=" * 78)
    print(f"起始 past_len = {args.p0}，共 {args.steps} 步")
    print(f"\n  {'step':>4} {'P':>5} {'setup(ms)':>10} {'exec(ms)':>9} "
          f"{'合计(ms)':>10} {'cache 体积':>11} {'相对误差':>10}")
    print("  " + "-" * 70)

    worst = 0.0
    samples = []       # (P, setup, exec, cache_mb)
    for s in range(args.steps):
        P = int(cache[0][0].shape[2])
        if P + 1 > args.p_max:
            print(f"  P+1={P + 1} 超出 profile max={args.p_max}，停止")
            break

        # ---- [setup] 设输入形状（24 个）→ 查输出形状 → 设地址（50 个）----
        t_set0 = time.perf_counter()
        ctx.set_input_shape("input_ids", (1, 1))
        for i in range(L):
            ctx.set_input_shape(in_cache[2 * i], tuple(cache[i][0].shape))
            ctx.set_input_shape(in_cache[2 * i + 1], tuple(cache[i][1].shape))

        # ★ 输出形状**问 TRT 要**，不要自己按 P+1 猜。
        #   猜错就是非法内存访问（而且报错会出现在后面的 synchronize）。
        new_k, new_v = [], []
        for i in range(L):
            sk = tuple(ctx.get_tensor_shape(out_cache[2 * i]))
            sv = tuple(ctx.get_tensor_shape(out_cache[2 * i + 1]))
            new_k.append(torch.empty(sk, dtype=torch.float32, device="cuda"))
            new_v.append(torch.empty(sv, dtype=torch.float32, device="cuda"))

        ctx.set_tensor_address("input_ids", pt_feed_g[s].data_ptr())
        ctx.set_tensor_address("logits", out_logits.data_ptr())
        for i in range(L):
            ctx.set_tensor_address(in_cache[2 * i], cache[i][0].data_ptr())
            ctx.set_tensor_address(in_cache[2 * i + 1], cache[i][1].data_ptr())
            ctx.set_tensor_address(out_cache[2 * i], new_k[i].data_ptr())
            ctx.set_tensor_address(out_cache[2 * i + 1], new_v[i].data_ptr())
        t_setup = (time.perf_counter() - t_set0) * 1000

        # ---- [exec] 真正算（★ 里面含图的 Concat，所以 O(P) 的拷贝在这里）----
        t_ex0 = time.perf_counter()
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise SystemExit("[FAIL] execute_async_v3 返回 False")
        stream.synchronize()
        t_exec = (time.perf_counter() - t_ex0) * 1000

        # ---- 精度 ----
        ref = pt_logits[s]                      # CPU 上的参考值
        d = float((out_logits.cpu() - ref).abs().max())
        scale = max(float(ref.abs().max()), 1e-9)
        rel = d / scale
        worst = max(worst, rel)

        # cache 体积（全部层的 k+v 总和）
        cache_mb = sum(t.numel() * 4 for kt, vt in zip(new_k, new_v)
                       for t in (kt, vt)) / 1048576.0

        # 不拿 TRT 的 argmax 喂回去（否则一旦 TF32 让某个 token 不同，后面就比错了）；
        # 只**统计**它和 PyTorch 一致了多少步 —— 这才是"数值差异是否影响决策"的证据。
        if s + 1 < args.steps:
            if int(out_logits.argmax()) == int(pt_feed[s + 1]):
                argmax_match += 1

        total = t_setup + t_exec
        samples.append((P, t_setup, t_exec, cache_mb))
        print(f"  {s:4d} {P:5d} {t_setup:10.3f} {t_exec:9.3f} "
              f"{total:10.3f} {cache_mb:9.1f}MB {rel:10.2e}")

        # ★★ new_k/new_v 已经是**全量历史**（gpt.py 里 k = cat([past_k, k])），
        #    所以直接替换，**不要**再 cat 一次 —— 再 cat 会让 P 每步近似翻倍
        #    （16 → 33 → 67 → 135 → …），几轮就冲出 profile 并 OOM。
        cache = [(new_k[i], new_v[i]) for i in range(L)]

    # ---- 汇总 ----
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    if worst < 1e-2:
        print(f"  精度: [OK] 最大相对误差 {worst:.2e} < 1e-2（TRT 是 TF32，比 ORT 的 1e-6 大属正常）")
    else:
        print(f"  精度: [WARN] 最大相对误差 {worst:.2e} 偏大")

    early = [x for x in samples if x[0] < args.p0 + 5]
    late = [x for x in samples if x[0] > args.p0 + args.steps - 6]
    if early and late:
        def avg(rows, idx):
            return sum(r[idx] for r in rows) / len(rows)
        print(f"\n  P 从 {early[0][0]} 到 {late[-1][0]} 的变化（看 O(P) 的代价）:")
        print(f"    {'':12} {'P≈' + str(early[0][0]):>12} {'P≈' + str(late[-1][0]):>12}")
        print(f"    {'setup(ms)':12} {avg(early, 1):12.3f} {avg(late, 1):12.3f}   ← 74 次 host 调用")
        print(f"    {'exec(ms)':12} {avg(early, 2):12.3f} {avg(late, 2):12.3f}   ← 含图的 Concat（O(P)）")
        print(f"    {'cache(MB)':12} {avg(early, 3):12.1f} {avg(late, 3):12.1f}")

    print(f"\n  argmax 一致: {argmax_match}/{max(1, args.steps - 1)} 步"
          f"  （TF32 下 logits 有 1e-3 级差异，但没改变任何一次选词）"
          if argmax_match == args.steps - 1 else
          f"\n  argmax 一致: {argmax_match}/{max(1, args.steps - 1)} 步  ⚠️ 有分歧")

    print("\n  怎么读:")
    print("    · new_k/new_v 是**全量历史**（gpt.py 里 k = cat([past_k,k])），所以直接替换 cache。")
    print("      那个 O(P) 的拼接发生在**引擎内部**（图里的 Concat），代价体现在 exec 列——")
    print("      看 exec 随 P 是否增长，就能看出朴素 KV cache 的不可扩展性。")
    print("      这正是 PagedAttention 要消除的东西：不拼，只改指针（页表）。")
    print("    · setup 是 host 侧开销：24 次 set_input_shape + 50 次 set_tensor_address。")
    print("      形状每步都变，所以 **CUDA Graph 在这里用不了**（图会把形状钉死）。")
    print("    · exec 才是 GPU 真正计算的时间。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
