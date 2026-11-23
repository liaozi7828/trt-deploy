#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w3_batch.py —— W3：batch 对 decode 的影响（验证"batch 放大 cache 拷贝"的推论）

上一轮发现：单请求时 O(P) 的 cache 拷贝只占 16%，被"读 154 MB 权重"压住了。
推论：**batch 会把 cache 拷贝成倍放大，而权重只读一次** —— 所以
      · per-step 延迟随 batch **次线性**增长（权重被摊薄）
      · per-token 延迟随 batch **显著下降**（这是 LLM 服务吞吐的根本来源）

本脚本要验证这个推论：batch × P 的二维矩阵。

计量方式和 w3_latency_vs_p.py 一致：
    · 固定 (batch, P)，重复 decode I 次（不更新 cache）
    · 两趟法拆 host / exec：第一趟全做，第二趟只做 host，exec = 差
    · R 轮取中位数 + σ + 95% CI

用法：
    python w3_batch.py --rebuild                      # 第一次要重建引擎（batch profile 变了）
    python w3_batch.py                                # 复用引擎直接测
    python w3_batch.py --batches 1 2 4 --ps 16 511
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import tensorrt as trt

from gpt import GPT, DEFAULT_CFG

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
WEIGHT_SEED = 1234          # ★ 和 w3_export_decode.py 一致


def cache_names(n_layer: int):
    ins, outs = [], []
    for i in range(n_layer):
        ins += [f"past_k_{i}", f"past_v_{i}"]
        outs += [f"new_k_{i}", f"new_v_{i}"]
    return ins, outs


def ci95(xs):
    n = len(xs)
    return 0.0 if n < 2 else 1.96 * statistics.stdev(xs) / (n ** 0.5)


def build(onnx_path: Path, engine_path: Path, L, H, hd, b_max, p_max):
    builder = trt.Builder(TRT_LOGGER)
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    except Exception:
        network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("[FAIL] ONNX 解析失败")
    print(f"  解析成功: {network.num_layers} 层, "
          f"{network.num_inputs} 输入, {network.num_outputs} 输出")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except Exception:
        pass

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (b_max, 1), (b_max, 1))
    for name in cache_names(L)[0] + cache_names(L)[1]:
        profile.set_shape(name,
                          (1, H, 1, hd),
                          (b_max, H, min(64, p_max), hd),
                          (b_max, H, p_max, hd))
    config.add_optimization_profile(profile)
    print(f"  profile: input_ids batch[1,{b_max}], past_len[1,{p_max}]")

    t0 = time.time()
    print("  正在编译...")
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit("[FAIL] 编译失败")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(ser)
    print(f"  编译完成: {engine_path} "
          f"({engine_path.stat().st_size / 1048576:.1f} MB, {time.time() - t0:.1f}s)")
    rt = trt.Runtime(TRT_LOGGER)
    return rt.deserialize_cuda_engine(engine_path.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="w1_out/gpt_decode.onnx")
    ap.add_argument("--engine", default="w1_out/gpt_decode_b4.engine")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--ps", nargs="+", type=int, default=[16, 511])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--p-max", type=int, default=511)
    args = ap.parse_args()

    cfg = DEFAULT_CFG
    L, H = cfg.n_layer, cfg.n_head
    hd, V = cfg.n_embd // H, cfg.vocab_size
    b_max = max(args.batches)

    engine_path = Path(args.engine)
    if args.rebuild or not engine_path.exists():
        print("=" * 88)
        print("建 decode 引擎（batch 动态）")
        print("=" * 88)
        engine = build(Path(args.onnx), engine_path, L, H, hd, b_max, args.p_max)
    else:
        rt = trt.Runtime(TRT_LOGGER)
        engine = rt.deserialize_cuda_engine(engine_path.read_bytes())
        print(f"复用引擎: {engine_path.name}")

    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    in_cache, out_cache = cache_names(L)

    torch.manual_seed(WEIGHT_SEED)
    model = GPT(cfg).eval()
    wsum = sum(float(p.detach().abs().sum()) for p in model.parameters())
    print(f"权重校验和: {wsum:.6f}   ← 必须等于 w3_export_decode.py 的那个")

    print("\n" + "=" * 88)
    print("batch × P 矩阵")
    print("=" * 88)
    print(f"  {'batch':>5} {'P':>5} {'host':>8} {'exec':>8} {'合计':>8} "
          f"{'σ%':>6} {'µs/token':>10} {'权重读取摊薄':>12}")
    print("  " + "-" * 76)

    base_per_token = None
    grid = {}
    for P in args.ps:
        if P > args.p_max:
            continue
        # 用真实 prefill 造 batch=1 的 cache，再沿 batch 维复制
        with torch.no_grad():
            prompt = torch.randint(0, V, (1, P))
            _lg, caches = model(prompt, kv_caches=None, use_cache=True)
        ck1 = [k.float().contiguous() for k, _ in caches]
        cv1 = [v.float().contiguous() for _, v in caches]

        for B in args.batches:
            if B > b_max:
                continue
            # ★ 沿 batch 维复制：所有请求共享同一份 cache（内容不重要，只测延迟）
            ck = [t.repeat(B, 1, 1, 1).cuda() for t in ck1]
            cv = [t.repeat(B, 1, 1, 1).cuda() for t in cv1]
            ids = torch.randint(0, V, (B, 1), dtype=torch.int64, device="cuda")
            out_logits = torch.empty((B, 1, V), dtype=torch.float32, device="cuda")

            def one(do_exec: bool) -> None:
                ctx.set_input_shape("input_ids", (B, 1))
                for i in range(L):
                    ctx.set_input_shape(in_cache[2 * i], tuple(ck[i].shape))
                    ctx.set_input_shape(in_cache[2 * i + 1], tuple(cv[i].shape))
                nk, nv = [], []
                for i in range(L):
                    nk.append(torch.empty(tuple(ctx.get_tensor_shape(out_cache[2 * i])),
                                          dtype=torch.float32, device="cuda"))
                    nv.append(torch.empty(tuple(ctx.get_tensor_shape(out_cache[2 * i + 1])),
                                          dtype=torch.float32, device="cuda"))
                ctx.set_tensor_address("input_ids", ids.data_ptr())
                ctx.set_tensor_address("logits", out_logits.data_ptr())
                for i in range(L):
                    ctx.set_tensor_address(in_cache[2 * i], ck[i].data_ptr())
                    ctx.set_tensor_address(in_cache[2 * i + 1], cv[i].data_ptr())
                    ctx.set_tensor_address(out_cache[2 * i], nk[i].data_ptr())
                    ctx.set_tensor_address(out_cache[2 * i + 1], nv[i].data_ptr())
                if do_exec:
                    if not ctx.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError("execute_async_v3 失败")

            try:
                for _ in range(2):
                    one(True)
                stream.synchronize()
            except Exception as e:
                print(f"  {B:5d} {P:5d}   跳过（{type(e).__name__}: {str(e)[:50]}）")
                continue

            full = []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    one(True)
                stream.synchronize()
                full.append((time.perf_counter() - t0) * 1000 / args.iters)
            host = []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    one(False)
                host.append((time.perf_counter() - t0) * 1000 / args.iters)

            f_med = statistics.median(full)
            h_med = statistics.median(host)
            e_med = max(f_med - h_med, 0.0)
            sd = statistics.stdev(full) / f_med * 100 if len(full) > 1 else 0.0
            per_tok = f_med / B
            if B == 1:
                base_per_token = per_tok
            ratio = (base_per_token / per_tok) if base_per_token else 1.0
            grid[(B, P)] = (h_med, e_med, f_med)
            print(f"  {B:5d} {P:5d} {h_med:8.3f} {e_med:8.3f} {f_med:8.3f} "
                  f"{sd:6.1f} {per_tok * 1000:10.0f} {ratio:11.2f}×")
            del ck, cv
        print()

    print("=" * 88)
    print("怎么读：")
    print("  · 『µs/token』下降 = 权重读取被摊薄了。**这就是 batching 提升吞吐的全部秘密**。")
    print("  · 『exec』随 batch 增长但**次线性**（不是 4 倍）——因为权重只读一次。")
    print("  · 对比不同 P：P 越大，cache 拷贝占比越高，exec 的 batch 曲线越接近线性。")
    print("  · 『host』与 batch、P 都无关（74 次调用是固定的），所以 batch 越大它占比越低。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
