#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w2_graph.py —— W2：CUDA Graph 对 Transformer 推理有多少收益？

为什么值得测（承接 P3 的发现）：
    YOLOv8 那边的结论是「2.014 ms 里 1.385 ms 是 CPU 在派发 kernel」。
    这里要问：Transformer 也有同样的问题吗？

    但答案会不一样，因为两者的瓶颈不同：
      · YOLO      : 100+ 个 kernel，每个只跑几微秒  → 派发开销占 69%
      · GPT(seq=1): 12 层 × ~6 kernel = 72 次启动，但单次要读 155 MB 权重
                    → 派发只占一部分，访存才是大头
      · GPT(seq=512): 计算量大，派发占比更低

    所以**预期是「seq 越短，Graph 的相对收益越大」**——这条曲线本身就是结论。

怎么测（和 P3 一样的规矩：同进程内配对比较）：
    ① 直连：每次调用都做「set_input_shape + set_tensor_address + execute + sync」
    ② Graph：形状和地址只设一次，捕获后只 replay
    两条路径用**同一份显存**（同一个 ids/out 张量），
    并且先跑一次直连、再捕获、再比对输出（必须一致，否则说明图是空的）。

用法：
    python w2_graph.py --engine w1_out/gpt.engine
    python w2_graph.py --engine w1_out/gpt.engine --shapes 1x1 1x64 1x256 1x512
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
def load_engine(path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise SystemExit(f"[FAIL] 反序列化失败: {path}")
    return engine


def io_names(engine: trt.ICudaEngine) -> tuple[str, str]:
    """返回 (输入名, 输出名)。用 TensorIOMode 判断，不硬编码名字。"""
    inp, out = None, None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inp = name
        else:
            out = name
    if inp is None or out is None:
        raise SystemExit("[FAIL] 找不到输入/输出张量")
    return inp, out


# ---------------------------------------------------------------------------
def bench_loop(fn, iters: int, warmup: int = 5) -> float:
    """跑 iters 次取**总耗时/iters**（单次 sync 放在两端）。

    为什么不每次 sync：每次 sync 会引入固定的驱动往返，会把两条路径的差异抹平。
    这里测的是"在循环里连续调用"的真实吞吐，和 C++ 侧的 benchmark 口径一致。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="w1_out/gpt.engine")
    ap.add_argument("--shapes", nargs="*", default=["1x1", "1x64", "1x256", "1x512"],
                    help="形如 1x512（batch x seq）")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()

    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"[FAIL] 找不到 {engine_path}")
        return 1

    engine = load_engine(engine_path)
    in_name, out_name = io_names(engine)
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()

    shapes = []
    for s in args.shapes:
        b, t = s.lower().split("x")
        shapes.append((int(b), int(t)))

    print("=" * 78)
    print("CUDA Graph vs 直连（同进程配对比较）")
    print("=" * 78)
    print(f"引擎   : {engine_path} ({engine_path.stat().st_size / 1048576:.1f} MB)")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"IO     : 输入={in_name}  输出={out_name}")
    print(f"iters  : {args.iters}（两端各一次 sync）")

    print(f"\n  {'batch':>5} {'seq':>5} {'直连(ms)':>11} {'Graph(ms)':>11} "
          f"{'加速比':>8} {'省下(ms)':>10} {'校验':>6}")
    print("  " + "-" * 64)

    for bs, seq in shapes:
        try:
            ids = torch.randint(0, args.vocab, (bs, seq), dtype=torch.int64, device="cuda")
            ctx.set_input_shape(in_name, (bs, seq))
            out_shape = tuple(ctx.get_tensor_shape(out_name))
            out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
            ctx.set_tensor_address(in_name, ids.data_ptr())
            ctx.set_tensor_address(out_name, out.data_ptr())
        except Exception as e:
            print(f"  {bs:5d} {seq:5d}   形状超出 profile: {e}")
            continue

        # ---------- ① 直连：每次调用都重设形状和地址 ----------
        def run_direct():
            ctx.set_input_shape(in_name, (bs, seq))
            ctx.set_tensor_address(in_name, ids.data_ptr())
            ctx.set_tensor_address(out_name, out.data_ptr())
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("execute_async_v3 失败")
            # ★ 注意：这里**故意不加 stream.synchronize()**。
            #   直连路径也只在批末同步一次，否则每次 sync 的固定开销会把两者差异抹平。

        direct_ms = bench_loop(run_direct, args.iters)

        # 存一份直连的输出，供捕获后校验
        run_direct()
        torch.cuda.synchronize()
        ref = out.clone()

        # ---------- ② CUDA Graph：捕获一次，之后只 replay ----------
        graph_ok = False
        graph_err = ""
        g = torch.cuda.CUDAGraph()
        try:
            # 捕获前的热身必须在**非默认流**上，且捕获期间不能有任何同步
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    if not ctx.execute_async_v3(side.cuda_stream):
                        raise RuntimeError("捕获前热身失败")
            torch.cuda.current_stream().wait_stream(side)

            with torch.cuda.graph(g):
                if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                    raise RuntimeError("捕获期间 execute_async_v3 返回 False")
            graph_ok = True
        except Exception as e:
            graph_err = f"{type(e).__name__}: {e}"

        if not graph_ok:
            print(f"  {bs:5d} {seq:5d} {direct_ms:11.3f} {'—':>11} {'—':>8} "
                  f"{'—':>10} {'✗':>6}   {graph_err[:40]}")
            continue

        # 校验：先清空 out，再 replay，看能不能算回参考值
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        max_abs = float((out - ref).abs().max())
        ok = max_abs < 1e-2          # TF32 量级，重放应该**逐位相同**，这里给足余量

        graph_ms = bench_loop(lambda: g.replay(), args.iters)
        speedup = direct_ms / graph_ms if graph_ms > 0 else 0.0

        print(f"  {bs:5d} {seq:5d} {direct_ms:11.3f} {graph_ms:11.3f} "
              f"{speedup:8.2f}x {direct_ms - graph_ms:10.3f} "
              f"{'✓' if ok else '✗'}")

    print("\n" + "=" * 78)
    print("怎么读：")
    print("  · 加速比 = 直连 / Graph。**seq 越短，Graph 的相对收益应该越大**——")
    print("    因为短序列的计算量小，派发开销占比高。")
    print("  · 如果 seq=512 的加速比接近 1.0，说明长序列是计算受限，Graph 帮不上忙。")
    print("  · 校验列必须是 ✓：Graph 重放应该和直连**逐位相同**（max|abs| 很小）。")
    print("    如果 ✗，说明图是空的或者指针绑错了——这时候那个加速比毫无意义。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
