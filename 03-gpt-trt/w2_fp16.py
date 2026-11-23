#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w2_fp16.py —— W2：把 GPT 转成 FP16 并建引擎，验证「decode 是访存受限」这个推论

为什么值得做（承接上一轮的带宽核算）：
    FP32 引擎的 decode（seq=1）= 1.076 ms
        155 MB 权重 ÷ 1.076 ms = 144 GB/s（峰值 384，用了 37%）
    如果权重变成 FP16（78 MB），同样的带宽下 decode 应该接近 0.54 ms。

    → 这是一个**可预测、可验证**的推论，不是"试试看"。

为什么不用 BuilderFlag.FP16：
    TRT 11 删掉了 `BuilderFlag.FP16`（本机实测可用 flag 列表里就是没有）。
    它是 strong typing：**精度由模型里声明的 dtype 决定**。
    所以要转的是 ONNX，不是引擎配置。这和 P3 里 YOLO 的 INT8 是同一套逻辑。

哪些算子必须留在 FP32：
    · LayerNormalization —— 方差要在 fp32 里算，否则长序列会漂
    · Softmax           —— exp/sum 的动态范围大
    · Erf               —— GELU 里的误差函数，fp16 精度不够
    其余（MatMul / Add / Mul / Reshape / Transpose）转 FP16 是安全的。

用法：
    python w2_fp16.py                          # 转 + 建 + 测
    python w2_fp16.py --baseline w1_out/gpt_opt1_1.engine   # 和 FP32 引擎对比
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# 复用 build_engine.py 里已经调好的构建/测速逻辑，避免两份实现分叉
from build_engine import BENCH_SHAPES, MAX_SHAPE, MIN_SHAPE, OPT_SHAPE, benchmark, build_engine, describe

# 默认留在 FP32 的算子
FP32_OPS = ["LayerNormalization", "Softmax", "Erf"]


# ---------------------------------------------------------------------------
def _convert_with(which: str, m, block_ops: list[str]):
    """按名字选一个转换器并调用。

    为什么要两个：
      · onnxexec_common 的老实现在 LayerNorm 的 weight 上会插错 Cast
        （报 `Type Error: Type (tensor(float16)) of output arg (model.ln_f.weight) of node (Identity)`），
        和 P3 里 Resize 节点上的问题同类。
      · onnxruntime.transformers.float16 是微软为 Transformer 单独维护的版本，
        对 LayerNorm/Attention 的处理更正确。
    所以先试 ORT，失败再退回 onnxconverter_common。
    """
    if which == "ort":
        from onnxruntime.transformers.float16 import convert_float_to_float16 as conv
    else:
        from onnxconverter_common.float16 import convert_float_to_float16 as conv
    return conv(m, keep_io_types=True, op_block_list=block_ops)


def convert_fp16(src: Path, dst: Path, block_ops: list[str]) -> bool:
    try:
        import onnx
    except ImportError as e:
        print(f"[FAIL] 缺依赖: {e}")
        return False

    print(f"[1/4] ONNX → FP16")
    print(f"      源: {src} ({src.stat().st_size / 1048576:.1f} MB)")
    m = onnx.load(str(src))
    before = {n.op_type for n in m.graph.node}

    m16 = None
    used = ""
    errors = []
    for which in ("ort", "onnxconverter_common"):
        try:
            m16 = _convert_with(which, m, block_ops)
            used = which
            break
        except Exception as e:
            errors.append(f"{which}: {type(e).__name__}: {str(e)[:120]}")
    if m16 is None:
        print(f"[FAIL] 两个转换器都失败：")
        for e in errors:
            print(f"       {e}")
        return False

    onnx.save(m16, str(dst))
    after = {n.op_type for n in m16.graph.node}
    added = sorted(after - before)

    print(f"      转换器: {used}")
    print(f"      目标: {dst} ({dst.stat().st_size / 1048576:.1f} MB)")
    print(f"      节点: {len(m.graph.node)} → {len(m16.graph.node)}"
          f"（新增 {len(m16.graph.node) - len(m.graph.node)} 个 Cast）")
    print(f"      保持 FP32 的算子: {block_ops}")
    print(f"      新增算子种类: {added if added else '无'}")
    print(f"      权重体积: {src.stat().st_size / 1048576:.1f} MB → "
          f"{dst.stat().st_size / 1048576:.1f} MB "
          f"（省 {100 * (1 - dst.stat().st_size / src.stat().st_size):.0f}%）")
    return True


# ---------------------------------------------------------------------------
def verify_ort(fp32_path: Path, fp16_path: Path, shapes, vocab: int) -> bool:
    """用 ONNXRuntime 跑一遍 FP16 模型，确认它能被第三方运行时接受。

    ★ 这一步不是可选的：
      P3 里 polygraphy 转出来的 FP16 模型 **TRT 能建、ORT 不能加载**
      （在 Resize 节点上插错了 Cast）。所以"TRT 能跑"不等于"模型是对的"。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("      [SKIP] 没有 onnxruntime，跳过第三方验证")
        return True

    print(f"\n[2/4] ONNXRuntime 交叉验证（FP32 vs FP16）")
    try:
        s32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        s16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"      [WARN] ORT 加载失败: {type(e).__name__}: {str(e)[:160]}")
        print(f"             → 和 P3 的 FP16 问题同类：TRT 容忍、ORT 不容忍。")
        print(f"             → 引擎仍然可以建，但**没有第三方证据**证明模型是对的。")
        return False

    print(f"      {'batch':>5} {'seq':>5} {'max|diff|':>12} {'相对误差':>10}")
    print("      " + "-" * 32)
    worst_rel = 0.0
    for bs, seq in shapes:
        if bs > MAX_SHAPE[0] or seq > MAX_SHAPE[1]:
            continue
        ids = np.random.randint(0, vocab, (bs, seq)).astype(np.int64)
        a = s32.run(None, {"input_ids": ids})[0]
        b = s16.run(None, {"input_ids": ids})[0]
        d = float(np.abs(a - b).max())
        scale = max(float(np.abs(a).max()), 1e-9)
        rel = d / scale
        worst_rel = max(worst_rel, rel)
        print(f"      {bs:5d} {seq:5d} {d:12.3e} {rel:10.3e}")

    if worst_rel < 1e-2:
        print(f"\n      [OK] 最大相对误差 {worst_rel:.2e} < 1e-2 —— FP16 转换在数值上站得住")
        return True
    print(f"\n      [WARN] 相对误差 {worst_rel:.2e} 偏大，考虑把更多算子加入 FP32 白名单")
    return False


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="w1_out/gpt.onnx")
    ap.add_argument("--onnx-fp16", default="w1_out/gpt_fp16.onnx")
    ap.add_argument("--engine", default="w1_out/gpt_fp16_opt1_1.engine")
    ap.add_argument("--baseline", default=None,
                    help="FP32 引擎路径（可选）；给了就做同形状对比")
    ap.add_argument("--block-ops", nargs="*", default=FP32_OPS,
                    help="保持 FP32 的算子（默认 LayerNormalization/Softmax/Erf）")
    ap.add_argument("--min", nargs=2, type=int, default=[1, 1])
    ap.add_argument("--opt", nargs=2, type=int, default=[1, 1],
                    help="默认按 decode 优化（上一轮证明 decode 最敏感）")
    ap.add_argument("--max", nargs=2, type=int, default=list(MAX_SHAPE))
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    vocab = args.vocab
    if vocab is None:
        try:
            from gpt import DEFAULT_CFG
            vocab = DEFAULT_CFG.vocab_size
        except Exception:
            vocab = 4096

    src = Path(args.onnx)
    if not src.exists():
        print(f"[FAIL] 找不到 {src}，先跑 export_gpt_onnx.py")
        return 1

    print("=" * 78)
    print("W2：GPT → FP16（ONNX 转精度 + TRT strong typing）")
    print("=" * 78)
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"profile: min={tuple(args.min)} opt={tuple(args.opt)} max={tuple(args.max)}")

    dst = Path(args.onnx_fp16)
    if not convert_fp16(src, dst, args.block_ops):
        return 1

    verify_ort(src, dst, BENCH_SHAPES, vocab)

    print(f"\n[3/4] 从 FP16 ONNX 建引擎")
    engine_path = Path(args.engine)
    engine = build_engine(dst, engine_path, use_fp16=True,
                          min_shape=args.min, opt_shape=args.opt, max_shape=args.max)
    describe(engine)

    print(f"\n[4/4] 延迟 / 吞吐（vocab={vocab}）")
    benchmark(engine, vocab=vocab)

    # ---------------- 和 FP32 引擎对比 ----------------
    if args.baseline:
        base_path = Path(args.baseline)
        if not base_path.exists():
            print(f"\n[SKIP] 找不到基线引擎 {base_path}")
        else:
            print(f"\n[对比] FP32 vs FP16（两个引擎，同脚本内依次跑）")
            import tensorrt as trt
            rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            base = rt.deserialize_cuda_engine(base_path.read_bytes())
            print(f"  基线: {base_path.name} ({base_path.stat().st_size / 1048576:.1f} MB)")
            print(f"  FP16: {engine_path.name} ({engine_path.stat().st_size / 1048576:.1f} MB)")
            print(f"\n  {'batch':>5} {'seq':>5} {'FP32(ms)':>10} {'FP16(ms)':>10} "
                  f"{'加速比':>8} {'FP32带宽':>10} {'FP16带宽':>10} {'数值 rel':>10}")
            print("  " + "-" * 78)
            w32 = base_path.stat().st_size / 1048576.0
            w16 = engine_path.stat().st_size / 1048576.0
            worst_rel = 0.0
            for bs, seq in BENCH_SHAPES:
                if bs > MAX_SHAPE[0] or seq > MAX_SHAPE[1]:
                    continue
                ctx32 = base.create_execution_context()
                ctx16 = engine.create_execution_context()
                stream = torch.cuda.Stream()
                ids = torch.randint(0, vocab, (bs, seq), dtype=torch.int64, device="cuda")

                def run(ctx, eng):
                    ctx.set_input_shape("input_ids", (bs, seq))
                    osh = tuple(ctx.get_tensor_shape("logits"))
                    out = torch.empty(osh, dtype=torch.float32, device="cuda")
                    ctx.set_tensor_address("input_ids", ids.data_ptr())
                    ctx.set_tensor_address("logits", out.data_ptr())
                    ctx.execute_async_v3(stream.cuda_stream)
                    stream.synchronize()
                    return out

                def t_of(ctx, eng):
                    for _ in range(5):
                        run(ctx, eng)
                    torch.cuda.synchronize()
                    import time
                    t0 = time.perf_counter()
                    for _ in range(args.iters):
                        run(ctx, eng)
                    torch.cuda.synchronize()
                    return (time.perf_counter() - t0) * 1000 / args.iters

                # ★ 先用同一份 ids 比一次数值，再测时间
                #   （时间可以跨引擎比，但数值只在这一步算得出来）
                o32 = run(ctx32, base).float().clone()
                o16 = run(ctx16, engine).float().clone()
                d = float((o16 - o32).abs().max())
                scale = max(float(o32.abs().max()), 1e-9)
                rel = d / scale
                worst_rel = max(worst_rel, rel)

                t32 = t_of(ctx32, base)
                t16 = t_of(ctx16, engine)
                # decode（seq=1）时每 token 要读一遍全部权重 → 用权重体积/耗时算有效带宽
                bw32 = w32 / (t32 / 1000.0) / 1024.0
                bw16 = w16 / (t16 / 1000.0) / 1024.0
                print(f"  {bs:5d} {seq:5d} {t32:10.3f} {t16:10.3f} "
                      f"{t32 / t16:8.2f}x {bw32:8.0f}GB/s {bw16:8.0f}GB/s "
                      f"{rel:10.2e}")

            print()
            if worst_rel < 1e-2:
                print(f"  [OK] FP16 引擎 vs FP32 引擎：最大**相对**误差 {worst_rel:.2e} < 1e-2")
                print(f"       （FP16 尾数 10 位 ≈ 1e-3，12 层累积到 1e-2 量级属正常）")
                print(f"       ★ 这是「两个真实引擎对同一输入的输出差」——")
                print(f"         不依赖 onnxruntime，绕开了转换器在 ln_f 上插错 Cast 的问题。")
            else:
                print(f"  [WARN] 相对误差 {worst_rel:.2e} 偏大：")
                print(f"         考虑把更多算子加入 FP32 白名单（--block-ops）")

            print("\n  怎么读：")
            print("    · 带宽 = 权重体积 / 耗时。**decode（seq=1）时应该接近显存峰值（~384 GB/s）**；")
            print("      如果 FP16 的带宽明显高于 FP32，说明它确实在吃同样的带宽、搬更少的字节。")
            print("    · seq=512 是计算受限，FP16 靠 Tensor Core 提速，和带宽无关。")
            print("    · 数值 rel 是 FP16 引擎相对 FP32 引擎的最大相对误差，必须 < 1e-2。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
