#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_engine.py —— P1-a W4：ONNX → TensorRT Engine（含 dynamic profile）

这份脚本把 TensorRT 的五个核心对象串起来，正好对应计划 W4 的验收点：

    Builder   工厂，负责"造引擎"
    Network   从 ONNX 解析出来的计算图（还没优化）
    Profile   告诉 TRT 输入尺寸的范围 —— **动态 shape 必须建它**
    Engine    编译产物，绑定 GPU 架构 + TRT 版本
    Context   执行上下文，一次只能被一个线程 enqueue

前置（只需一次）:
    ~/.venvs/torch/bin/pip install tensorrt-cu12

用法:
    ~/.venvs/torch/bin/python build_engine.py                 # 首次构建 + 验证
    ~/.venvs/torch/bin/python build_engine.py --rebuild       # 强制重建
    ~/.venvs/torch/bin/python build_engine.py --fp32          # 不用 FP16

为什么用 torch 那个 venv:
    脚本里要用 torch 分配 GPU 显存、拿 CUDA stream、和 ONNXRuntime 对精度，
    所以 torch 和 tensorrt 必须在同一个环境里。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 动态维度的三档尺寸。名字必须和 ONNX 里的 dim_param 一致（batch / seq）
MIN_SHAPE = (1, 1)     # batch=1, seq=1（decode 单步）
OPT_SHAPE = (1, 256)   # OPT_SHAPE 决定 TRT 挑哪套 kernel，按"最常见的长序列"选；曾用 8 导致长序列吃亏
MAX_SHAPE = (4, 512)   # block_size=512，batch 上限 4

# 要测的形状：从 decode 单步到 prefill 满窗口，再到 batch 提升吞吐
BENCH_SHAPES = [(1, 1), (1, 64), (1, 256), (1, 512), (4, 512)]


# =============================================================================
# 1. 构建 engine
# =============================================================================
def build_engine(onnx_path: Path, engine_path: Path,
                 use_fp16: bool = True, min_shape=None, opt_shape=None,
                 max_shape=None) -> trt.ICudaEngine:
    t0 = time.time()

    builder = trt.Builder(TRT_LOGGER)
    # TRT 10+ 默认就是 explicit batch；老写法要显式传 flag，两种都兼容一下
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    except Exception:
        network = builder.create_network()

    parser = trt.OnnxParser(network, TRT_LOGGER)
    data = onnx_path.read_bytes()
    if not parser.parse(data):
        print("[FAIL] ONNX 解析失败：")
        for i in range(parser.num_errors):
            print(f"   - {parser.get_error(i)}")
        raise SystemExit(1)
    print(f"  解析成功: {network.num_layers} 层, "
          f"{network.num_inputs} 输入, {network.num_outputs} 输出")

    config = builder.create_builder_config()

    # workspace：建引擎时的"草稿纸"，太小会影响优化，太大浪费显存
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    except Exception as e:
        print(f"  [WARN] 设置 workspace 失败（本版本 API 可能不同）: {e}")

    # 精度模式：各版本差异极大，用 getattr 探测而不是硬写
    #   TRT 8/10: BuilderFlag.FP16 存在
    #   TRT 11 : 该枚举被移除，精度改由别的方式控制
    if use_fp16:
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
        if fp16_flag is not None:
            config.set_flag(fp16_flag)
            print("  精度模式: FP16")
        else:
            avail = sorted(n for n in dir(trt.BuilderFlag) if not n.startswith("_"))
            print("  精度模式: 默认（本版本没有 BuilderFlag.FP16）")
            print(f"  本版本可用 BuilderFlag: {avail}")

    # ★ 动态 shape 必须建 profile，否则 build 会直接报错
    mn = tuple(min_shape or MIN_SHAPE)
    op = tuple(opt_shape or OPT_SHAPE)
    mx = tuple(max_shape or MAX_SHAPE)
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", min=mn, opt=op, max=mx)
    config.add_optimization_profile(profile)
    print(f"  profile: min={mn} opt={op} max={mx}")

    print("  正在编译（这一步最慢，几十秒到几分钟）...")
    try:
        serialized = builder.build_serialized_network(network, config)
    except AttributeError:
        # 极老/极新版本可能没有这个入口，退回 build_engine + serialize
        print("  [WARN] 没有 build_serialized_network，退回 build_engine")
        eng = builder.build_engine(network, config)
        if eng is None:
            print("[FAIL] 编译失败，把 TRT_LOGGER 改成 INFO 级看详细原因")
            raise SystemExit(1)
        serialized = eng.serialize()
    if serialized is None:
        print("[FAIL] 编译失败，把 TRT_LOGGER 改成 INFO 级看详细原因")
        raise SystemExit(1)

    engine_path.write_bytes(serialized)
    dt = time.time() - t0
    print(f"  编译完成: {engine_path} ({engine_path.stat().st_size/1024/1024:.1f} MB, {dt:.1f}s)")

    # 立刻反序列化回来，验证存盘文件真的能用
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        print("[FAIL] engine 反序列化失败")
        raise SystemExit(1)
    return engine


# =============================================================================
# 2. 打印 engine 信息
# =============================================================================
def describe(engine: trt.ICudaEngine) -> None:
    print(f"\n  IO 张量 ({engine.num_io_tensors} 个):")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = "INPUT " if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"    {mode} {name:12s} dtype={engine.get_tensor_dtype(name)}")
    print(f"  optimization profile 数量: {engine.num_optimization_profiles}")


# =============================================================================
# 3. 推理（TRT 10+ 的 set_tensor_address + execute_async_v3）
# =============================================================================
def infer(context: trt.IExecutionContext, engine: trt.ICudaEngine,
          input_ids: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
    assert input_ids.is_cuda, "输入必须在 GPU 上，否则性能全花在拷贝上"

    # 每次换形状都要先告诉 context —— 这是动态 shape 的代价
    context.set_input_shape("input_ids", tuple(input_ids.shape))

    out_shape = tuple(context.get_tensor_shape("logits"))
    logits = torch.empty(out_shape, dtype=torch.float32, device="cuda")

    context.set_tensor_address("input_ids", input_ids.data_ptr())
    context.set_tensor_address("logits", logits.data_ptr())

    # ★ 必须用非默认流。TRT 在默认(legacy)流上会插入 cudaStreamSynchronize，
    #   官方日志里那句 "Using default stream ... may lead to performance issues" 就是这个。
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("execute_async_v3 失败")
    stream.synchronize()
    return logits


def benchmark(engine: trt.ICudaEngine, vocab: int,
              warmup: int = 10, iters: int = 50) -> None:
    """测延迟：先 warmup，再跑 N 次取中位数。

    不 warmup 的第一次调用会包含 CUDA context 初始化、kernel 加载、
    显存分配，可能比稳态慢 10 倍以上 —— 那个数字没有任何意义。
    """
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    print(f"\n  {'batch':>5} {'seq':>5} {'median(ms)':>12} {'p90(ms)':>9} "
          f"{'tokens/s':>11} {'ms/token':>10}")
    print("  " + "-" * 62)
    for bs, seq in BENCH_SHAPES:
        if bs > MAX_SHAPE[0] or seq > MAX_SHAPE[1]:
            continue
        ids = torch.randint(0, vocab, (bs, seq), device="cuda")

        for _ in range(warmup):
            infer(context, engine, ids, stream)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer(context, engine, ids, stream)
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        med = times[len(times) // 2]
        p90 = times[int(len(times) * 0.9)]
        # ★ 不能用 samples/s：每次调用的 token 数不同，横着比毫无意义。
        #   tokens/s = 每秒钟处理多少 token；ms/token = 单 token 成本（decode 的核心指标）
        toks = bs * seq
        print(f"  {bs:5d} {seq:5d} {med:12.3f} {p90:9.3f} "
              f"{toks * 1000.0 / med:11.0f} {med / toks:10.4f}")


def run_compare(engine: trt.ICudaEngine, onnx_path: Path, vocab: int = 128) -> None:
    import onnxruntime as ort

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print(f"\n  {'batch':>5} {'seq':>5} {'max|diff|':>12}")
    print("  " + "-" * 30)
    worst = 0.0
    for bs, seq in [(1, 1), (1, 64), (1, 256), (1, 512), (4, 512)]:
        ids_cpu = torch.randint(0, vocab, (bs, seq))
        ids_gpu = ids_cpu.cuda()

        y_trt = infer(context, engine, ids_gpu, stream).cpu().numpy()
        # ORT（同一份 ONNX，权重必然一致，所以能直接比）
        y_ort = sess.run(None, {"input_ids": ids_cpu.numpy()})[0]

        d = float(abs(y_trt - y_ort).max())
        worst = max(worst, d)
        flag = "" if d < 1e-2 else "  <-- 偏大"
        print(f"  {bs:5d} {seq:5d} {d:12.3e}{flag}")

    print()
    if worst < 1e-2:
        print("  [OK] TRT 与 ONNXRuntime 输出一致")
        print("       （当前精度约 TF32 量级，1e-4 属正常；FP32 会是 1e-7，FP16 约 1e-3）")
    else:
        print("  [WARN] 差异偏大，检查精度模式或 profile 是否覆盖这些形状")


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="w1_out/gpt.onnx")
    ap.add_argument("--engine", default="w1_out/gpt.engine")
    ap.add_argument("--rebuild", action="store_true", help="强制重新编译")
    ap.add_argument("--fp32", action="store_true", help="不用 FP16")
    ap.add_argument("--min", nargs=2, type=int, default=list(MIN_SHAPE),
                    help="profile min (batch seq)")
    ap.add_argument("--opt", nargs=2, type=int, default=list(OPT_SHAPE),
                    help="profile opt (batch seq) —— 决定 TRT 挑哪套 kernel")
    ap.add_argument("--max", nargs=2, type=int, default=list(MAX_SHAPE),
                    help="profile max (batch seq)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="跳过 ONNXRuntime 精度对比（只比 opt_shape 时省时间）")
    ap.add_argument("--vocab", type=int, default=None,
                    help="词表大小；默认从 gpt.DEFAULT_CFG 读")
    args = ap.parse_args()

    vocab = args.vocab
    if vocab is None:
        try:
            from gpt import DEFAULT_CFG
            vocab = DEFAULT_CFG.vocab_size
        except Exception:
            vocab = 4096

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.exists():
        print(f"[FAIL] 找不到 {onnx_path}，先跑 export_gpt_onnx.py")
        return 1

    print("=" * 66)
    print("ONNX → TensorRT Engine")
    print("=" * 66)
    print(f"TRT 版本   : {trt.__version__}")
    print(f"输入 ONNX  : {onnx_path} ({onnx_path.stat().st_size/1024:.1f} KB)")
    print(f"GPU        : {torch.cuda.get_device_name(0)}")

    print("\n[1/3] 构建 engine")
    engine = build_engine(onnx_path, engine_path, use_fp16=not args.fp32,
                          min_shape=args.min, opt_shape=args.opt, max_shape=args.max)

    print("\n[2/3] engine 信息")
    describe(engine)

    if args.skip_verify:
        print("\n[3/4] 精度验证：已跳过（--skip-verify）")
    else:
        print("\n[3/4] 精度验证（TRT vs ONNXRuntime）")
        run_compare(engine, onnx_path, vocab=vocab)

    print(f"\n[4/4] 延迟测试（warmup=10, iters=50, 取中位数；vocab={vocab}）")
    benchmark(engine, vocab=vocab)
    print(f"\n  【一句话摘要】engine={engine_path.name} "
          f"opt={tuple(args.opt)} min={tuple(args.min)} max={tuple(args.max)}")

    print("\n下一步（W2）：")
    print("  1) 三档 opt_shape 对比（这是 prefill/decode 拆分的实证）：")
    for o in ["1 1", "1 256", "1 512"]:
        print(f"     python build_engine.py --opt {o} --skip-verify \\")
        print(f"            --engine w1_out/gpt_opt{o.replace(' ', '_')}.engine")
    print("  2) CUDA Graph 对比：python w2_graph.py --engine w1_out/gpt.engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
