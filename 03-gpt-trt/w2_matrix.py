#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w2_matrix.py —— W2 收尾：{FP32, FP16} × {直连, Graph} 四路交错对比

为什么要有这个脚本（而不是拿之前零散的数字凑一张表）：
    1. FP16 引擎被重建过（换转换器后节点数变了），旧数字作废
    2. 之前都是「先测 A 再测 B」——机器连续跑两小时会降频，后测的系统性吃亏
       （我们实测过：同一个 FP32 引擎文件，两次运行差 16%）
    3. 所以必须：**同进程 + 交错 + 轮转起点 + 每轮只 sync 一次**

测法（照搬 P3 定下的规矩）：
    · 对每个 shape，准备 4 条路径：(fp32,direct) (fp32,graph) (fp16,direct) (fp16,graph)
    · 跑 R 轮，每轮把 4 条路径都跑一遍（每条约 iters 次），**起点按 (r+k)%4 轮转**
    · 每条路径取"逐轮中位数"，比值取"逐轮比值的集合"→ 中位数 + 95% CI
    · 每轮只在末尾 sync 一次（每次调用都 sync 会把差异抹平）
    · Graph 路径先做校验：清空输出 → replay → 必须和直连一致

用法：
    python w2_matrix.py \
        --engines w1_out/gpt_opt1_1.engine w1_out/gpt_fp16_opt1_1.engine \
        --shapes 1x1 1x64 1x256 1x512 --rounds 6 --iters 20
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path: Path) -> trt.ICudaEngine:
    rt = trt.Runtime(TRT_LOGGER)
    eng = rt.deserialize_cuda_engine(path.read_bytes())
    if eng is None:
        raise SystemExit(f"[FAIL] 反序列化失败: {path}")
    return eng


def io_names(eng: trt.ICudaEngine) -> tuple[str, str]:
    inp = out = None
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            inp = n
        else:
            out = n
    if inp is None or out is None:
        raise SystemExit("[FAIL] 找不到 IO 张量")
    return inp, out


def ci95(xs: list[float]) -> float:
    """95% 置信半宽 = 1.96·σ/√n"""
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / (n ** 0.5)


class Runner:
    """一条被测路径：某引擎 + 直连或 Graph"""

    def __init__(self, label: str, eng: trt.ICudaEngine, use_graph: bool,
                 ids: torch.Tensor):
        """★ ids 由调用方传入并**在所有路径间共享**。

        这是必须的：Graph 重放和直连比的是"同样输入下的输出"。
        如果每个 Runner 各生成一份随机 ids，比的就是两个不同的问题——
        表现出来的现象是 rel ≈ 0.8（两边都算出了真实 logits，只是不一样），
        很容易被误判成"图是空的"。
        """
        self.label = label
        self.use_graph = use_graph
        self.bs, self.seq = int(ids.shape[0]), int(ids.shape[1])

        self.in_name, self.out_name = io_names(eng)
        # 直连和 Graph 用**各自独立的 context**：
        # 直连每次 set_input_shape 可能会让已捕获的图失效，所以不能共用
        self.ctx = eng.create_execution_context()
        self.ctx.set_input_shape(self.in_name, (self.bs, self.seq))
        osh = tuple(self.ctx.get_tensor_shape(self.out_name))

        self.ids = ids                      # ★ 共享，不新建
        self.out = torch.empty(osh, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.in_name, self.ids.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.stream = torch.cuda.Stream()

        self.graph = None
        self.err = ""
        if use_graph:
            self._capture()

    def _capture(self) -> None:
        try:
            # 捕获前热身必须在非默认流；捕获期间禁止任何同步。
            #
            # ★ 次数很关键：TRT 的 workspace 是「按需增长」的，前几次执行可能还在扩容。
            #   如果扩容正好落在捕获期间，那次 enqueue 会被**静默丢弃**——
            #   execute_async_v3 仍返回 True，但图是空的。
            #   w2_graph.py 能成功，是因为它先测了 30 次直连、workspace 早已稳定；
            #   而这里只热身 3 次，不够。所以提到 20 次，并显式同步。
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(20):
                    if not self.ctx.execute_async_v3(side.cuda_stream):
                        raise RuntimeError("热身时 execute_async_v3 返回 False")
            side.synchronize()                      # ★ 确保热身全部落地
            torch.cuda.current_stream().wait_stream(side)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                    raise RuntimeError("捕获时 execute_async_v3 返回 False")
            self.graph = g
        except Exception as e:
            self.err = f"{type(e).__name__}: {str(e)[:80]}"

    def run_once(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        else:
            # 直连：每次都重设形状和地址（这是"最朴素"的用法）
            self.ctx.set_input_shape(self.in_name, (self.bs, self.seq))
            self.ctx.set_tensor_address(self.in_name, self.ids.data_ptr())
            self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
            if not self.ctx.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("execute_async_v3 失败")

    def time_block(self, iters: int) -> float:
        """跑 iters 次，返回单次均值（ms）。**只在块末 sync 一次。**"""
        t0 = time.perf_counter()
        for _ in range(iters):
            self.run_once()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["w1_out/gpt_opt1_1.engine",
                                                    "w1_out/gpt_fp16_opt1_1.engine"])
    ap.add_argument("--shapes", nargs="*", default=["1x1", "1x64", "1x256", "1x512"])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()

    engs = []
    for p in args.engines:
        path = Path(p)
        if not path.exists():
            print(f"[SKIP] 找不到 {path}")
            continue
        # ★ 只记 path 和体积；engine 实例在每条路径里**各自反序列化一份**
        #   原因：TRT 的 workspace 分配器是「每引擎」共享的，
        #   同一个引擎建两个 context 时，第二个 context 在捕获期间可能需要扩容
        #   → 分配显存 → 捕获被拒 → 图是空的。
        engs.append((path, path.stat().st_size / 1048576.0))
    if len(engs) < 1:
        print("[FAIL] 没有可用的引擎")
        return 1

    shapes = []
    for s in args.shapes:
        b, t = s.lower().split("x")
        shapes.append((int(b), int(t)))

    print("=" * 96)
    print("四路交错对比：{FP32, FP16} × {直连, Graph}")
    print("=" * 96)
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"轮数   : {args.rounds}，每轮每路 {args.iters} 次，起点按 (r+k)%N 轮转")
    if args.rounds < 6:
        print(f"  ⚠️ 轮数 < 6，σ 和 CI 不可信（一个离群点就能让 σ 超过 100%）")
    for path, mb in engs:
        print(f"引擎   : {path.name} ({mb:.1f} MB)")

    for bs, seq in shapes:
        print(f"\n{'=' * 96}")
        print(f"shape = batch={bs}, seq={seq}")
        print("=" * 96)

        # 每条引擎 → 直连 + Graph 两条路径
        paths: list[Runner] = []
        # ★ 一个 shape 只生成一份输入，所有引擎/所有路径共用
        shared_ids = torch.randint(0, args.vocab, (bs, seq),
                                   dtype=torch.int64, device="cuda")
        for path_, mb in engs:
            tag = "fp16" if "fp16" in str(path_).lower() else "fp32"
            for use_graph in (False, True):
                lbl = f"{tag}/{'graph ' if use_graph else 'direct'}"
                try:
                    own_eng = load_engine(path_)      # ★ 每条路径一份独立引擎
                    p = Runner(lbl, own_eng, use_graph, shared_ids)
                except Exception as e:
                    print(f"  [SKIP] {lbl}: {e}")
                    continue
                if use_graph and p.graph is None:
                    print(f"  [SKIP] {lbl}: 捕获失败（{p.err}）")
                    continue
                paths.append(p)

        if not paths:
            print("  没有任何可测路径")
            continue

        # ---- 校验：Graph 重放必须和同一引擎的直连一致 ----
        #
        # ★ 校验失败必须**把这条路径剔除**，不能只打个 ✗ 然后继续测速——
        #   空图会跑得"很快"，那个加速比是假的。
        print("  [校验] Graph 重放 vs 同引擎直连")
        failed: set[str] = set()
        for p in paths:
            if p.graph is None:
                continue
            direct = [q for q in paths if not q.use_graph
                      and q.label.split("/")[0] == p.label.split("/")[0]]
            if not direct:
                continue
            d = direct[0]
            d.run_once()
            torch.cuda.synchronize()
            ref = d.out.clone()
            p.out.zero_()
            p.graph.replay()
            torch.cuda.synchronize()
            max_abs = float((p.out - ref).abs().max())
            scale = max(float(ref.abs().max()), 1e-9)
            rel = max_abs / scale
            ok = rel < 1e-2
            print(f"    {p.label:14s} max|diff|={max_abs:.3e} rel={rel:.2e} "
                  f"{'✓' if ok else '✗ 图是空的，剔除'}")
            if not ok:
                failed.add(p.label)

        if failed:
            paths = [q for q in paths if q.label not in failed]
            print(f"  → 剔除了 {len(failed)} 条路径，剩 {len(paths)} 条")
            print(f"     可能原因：TRT 在长序列上用的 kernel 内部要临时 workspace，")
            print(f"     而 cudaStreamBeginCapture 期间禁止分配显存")
            print(f"     （和 P3 里 EfficientNMS plugin 不能被图捕获是同一类问题）")
            if not paths:
                print("  没有可测路径，跳过这个 shape")
                continue

        # ---- 交错测量 ----
        # 预热：直连路径的第一次执行可能包含 workspace 分配 / tactic 落地，
        # 只用 3 次会留下一个巨大的离群点（实测 σ 会飙到 100%+）。
        for p in paths:
            p.time_block(10)
        torch.cuda.synchronize()

        times: dict[str, list[float]] = {p.label: [] for p in paths}
        n = len(paths)
        for r in range(args.rounds):
            for k in range(n):
                p = paths[(r + k) % n]        # ★ 轮转起点
                times[p.label].append(p.time_block(args.iters))

        print(f"\n  {'路径':<16} {'中位数(ms)':>12} {'σ%':>7} {'CI95%':>8}")
        print("  " + "-" * 48)
        med: dict[str, float] = {}
        for p in paths:
            v = times[p.label]
            m = statistics.median(v)
            med[p.label] = m
            sd = statistics.stdev(v) / m * 100 if len(v) > 1 and m > 0 else 0.0
            print(f"  {p.label:<16} {m:12.3f} {sd:7.1f} {ci95(v) / m * 100:8.1f}")

        # ---- 相对基线的比值（逐轮配对，消漂移）----
        base_lbl = next((q.label for q in paths
                         if q.label.startswith("fp32") and "direct" in q.label), paths[0].label)
        if base_lbl in times:
            print(f"\n  相对基线 {base_lbl} 的比值（逐轮配对，中位数 ± CI）")
            print("  " + "-" * 48)
            for p in paths:
                if p.label == base_lbl:
                    continue
                ratios = [a / b for a, b in zip(times[p.label], times[base_lbl]) if b > 0]
                if not ratios:
                    continue
                rr = statistics.median(ratios)
                c = ci95(ratios) / rr * 100 if len(ratios) > 1 else 0.0
                verdict = "显著" if abs(1 - rr) > ci95(ratios) / rr else "在噪声内"
                print(f"  {p.label:<16} {rr:6.3f}x  ±{c:4.1f}%   CPU/GPU 时间 {1 / rr:5.2f}×  → {verdict}")

    print("\n" + "=" * 96)
    print("怎么读：")
    print("  · 每一列都是「同一进程内、轮转起点、每轮只 sync 一次」测的，比跨运行的数字可信。")
    print("  · 比值是**逐轮配对**算的（第 r 轮的 FP16 除以第 r 轮的 FP32），所以温度漂移被约掉。")
    print("  · CI95% 是这个数字的不确定度：比值 ±CI 若跨过 1.0，就说明差异在噪声内。")
    print("  · 校验列必须全是 ✓，否则那个 Graph 路径的加速比没有意义。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
