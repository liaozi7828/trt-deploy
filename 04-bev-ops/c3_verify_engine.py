#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c3_verify_engine.py —— C3 收尾：**BEV pooling 引擎算得对吗？**

为什么这一步是决定性的：
    trtexec 报 PASSED 只说明引擎建出来了。
    而 ONNX 里那个 `ScatterElements` 带一个 `reduction` 属性：
        reduction='add'  → 累加（我们要的）
        reduction='none' → **覆盖写**（结果只剩最后一个来源）
    TRT 如果忽略这个属性，引擎照样能建、能跑，**但输出全错**。

    rank 的重复率 68.3%（每个格子平均 ~3 个来源），所以"覆盖写"和"累加"
    的结果差别极大 —— 这个测试能一刀切开。

检查项：
    ① 引擎输出 vs PyTorch scatter_add
    ② 与"覆盖写"的期望值对比 —— 如果引擎匹配覆盖写而不是累加，就实锤了
    ③ 连跑两次一致（排除未初始化输出）

用法：
    python c3_verify_engine.py --engine c_out/bev_pool.engine
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ★★ 必须初始化 TRT 的插件库，否则含 plugin 的引擎**加载不了**。
#    ONNX 的 ScatterElements 被 TRT 实现成一个内置 plugin（ScatterElements v2），
#    所以这个引擎依赖插件注册表。
#    现象：trtexec 能建能存（它自己初始化过），但纯 Python 进程里
#          deserialize 之后 createExecutionContext 直接失败：
#          "Cannot find plugin: ScatterElements, version: 2"
#    —— 这也是为什么"trtexec 报 PASSED"不等于"你的代码能用这个引擎"。
_PLUGIN_OK = False
try:
    _PLUGIN_OK = bool(trt.init_libnvinfer_plugins(TRT_LOGGER, ""))
except Exception as _e:      # 某些构建里签名不同
    print(f"[WARN] init_libnvinfer_plugins 失败: {_e}")


def bev_pool_scatter(feat, weight, rank, bev_n):
    B, C, M = feat.shape
    src = feat * weight
    idx = rank.unsqueeze(1).expand(B, C, M).contiguous()
    return torch.zeros(B, C, bev_n, dtype=feat.dtype).scatter_add(2, idx, src)


def bev_pool_overwrite(feat, weight, rank, bev_n):
    """故意做"覆盖写"（不累加）：用 scatter 的 reduction='none' 语义。

    如果 TRT 忽略了 reduction='add'，它的输出应该等于这个。
    """
    B, C, M = feat.shape
    src = feat * weight
    idx = rank.unsqueeze(1).expand(B, C, M).contiguous()
    return torch.zeros(B, C, bev_n, dtype=feat.dtype).scatter(2, idx, src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="c_out/bev_pool_inputs.pt")
    ap.add_argument("--engine", default="c_out/bev_pool.engine")
    args = ap.parse_args()

    for p in (args.inputs, args.engine):
        if not Path(p).exists():
            print(f"[FAIL] 找不到 {p}")
            return 1

    d = torch.load(args.inputs, weights_only=False)
    feat, weight, rank, bev_n = d["feat"], d["weight"], d["rank"], d["bev_n"]

    print("=" * 78)
    print("C3 收尾：BEV pooling 引擎算得对吗？")
    print(f"插件库初始化: {_PLUGIN_OK}   ← 含 plugin 的引擎需要它")
    print("=" * 78)
    print(f"输入: feat {tuple(feat.shape)}  weight {tuple(weight.shape)}  "
          f"rank {tuple(rank.shape)}  bev_n={bev_n}")
    n_dup = int((1.0 - len(torch.unique(rank)) / rank.numel()) * 100)
    print(f"rank 重复率: {n_dup}%  → 累加与覆盖写的差别应该很大")

    y_add = bev_pool_scatter(feat, weight, rank, bev_n)
    y_ow = bev_pool_overwrite(feat, weight, rank, bev_n)
    gap = float((y_add - y_ow).abs().max())
    print(f"\n用 PyTorch 造出两个参照答案：")
    print(f"  scatter_add（累加）    max={y_add.abs().max():.4f}")
    print(f"  scatter（覆盖写）      max={y_ow.abs().max():.4f}")
    print(f"  两者的差 max|diff|     = {gap:.4e}   "
          f"{'← 差别足够大，能区分' if gap > 1e-3 else '⚠️ 差别太小，测试无区分力'}")

    # ---------------- 加载引擎 ----------------
    rt = trt.Runtime(TRT_LOGGER)
    engine = rt.deserialize_cuda_engine(Path(args.engine).read_bytes())
    if engine is None:
        print("[FAIL] 反序列化失败")
        return 1
    ctx = engine.create_execution_context()

    ins, outs = [], []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        (ins if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else outs).append(n)
    print(f"\nengine IO: 输入 {ins}  输出 {outs}")

    stream = torch.cuda.Stream()
    gpu = {"feat": feat.cuda().contiguous(),
           "weight": weight.cuda().contiguous(),
           "rank": rank.cuda().contiguous()}
    out_shape = tuple(ctx.get_tensor_shape(outs[0]))
    out_buf = torch.empty(out_shape, dtype=torch.float32, device="cuda")

    def run():
        for n in ins:
            ctx.set_tensor_address(n, gpu[n].data_ptr())
        ctx.set_tensor_address(outs[0], out_buf.data_ptr())
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 失败")
        stream.synchronize()

    run()
    y_trt = out_buf.clone()

    d_add = float((y_trt.cpu() - y_add).abs().max())
    d_ow = float((y_trt.cpu() - y_ow).abs().max())

    print("\n" + "-" * 78)
    print("[①] 引擎输出 vs 两个参照答案")
    print(f"    vs 累加   max|diff| = {d_add:.4e}")
    print(f"    vs 覆盖写 max|diff| = {d_ow:.4e}")

    print("\n[②] 判定")
    # ★ 用"谁明显更小"来判，而不是绝对阈值 ——
    #   本例里 add 与 overwrite 的差是 1.07e-3，绝对阈值 1e-3 太紧，容易误判。
    closer = "add" if d_add < d_ow else "overwrite"
    print(f"    引擎更接近: {closer}（差 d_add={d_add:.3e} / d_ow={d_ow:.3e}）")
    if d_add < 1e-3:
        print(f"    ✅ TRT 的 ScatterElements **正确实现了 reduction='add'**")
        print(f"       → BEV pooling 也不需要 plugin，TRT 11 原生支持")
        verdict = "add_supported"
    elif d_ow < 1e-3:
        print(f"    ❌ 引擎的输出匹配的是**覆盖写**，不是累加！")
        print(f"       → TRT 忽略了 reduction='add'（这是最危险的情况：能建、能跑、算错）")
        print(f"       → **这才是写 plugin 的真正理由**")
        verdict = "add_ignored"
    else:
        print(f"    ⚠️ 与两个答案都对不上 —— 有别的语义差异（可能是索引维度的处理不同）")
        verdict = "unknown"

    # ---------------- 连跑两次 ----------------
    out_buf.zero_()
    run()
    d_rep = float((out_buf - y_trt).abs().max())
    # ★ 不能用 "== 0" 判据：scatter-add 是**原子浮点累加**，
    #   浮点加法不满足结合律（(a+b)+c ≠ a+(b+c)），
    #   线程到达顺序每次不同 → 结果差最后一个比特（实测 1e-10 量级）。
    #   这是原子累加算子的固有性质，不是 bug。
    rel_rep = d_rep / max(float(y_trt.abs().max()), 1e-9)
    print(f"\n[③] 连跑两次: max|diff| = {d_rep:.3e}  相对 = {rel_rep:.2e}")
    if d_rep == 0.0:
        print("    ✅ 逐位一致")
    elif rel_rep < 1e-6:
        print("    ✅ 在 float32 机器精度内一致 —— **原子浮点累加的固有非确定性**")
        print("       （(a+b)+c ≠ a+(b+c)，线程顺序每次不同 → 差最后一位）")
        print("       ★ 所以 scatter 类算子**做不到逐位复现**，验证判据要相应放宽")
    else:
        print("    ⚠️ 差异超出机器精度，需排查")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    if verdict == "add_supported":
        print("""  TRT 11 原生支持 ScatterElements(reduction='add') → BEV pooling 不需要 plugin。

  那么"必须写 plugin"的真正边界是**动态输出规模**：
    · BEV pooling     ：输出固定 bev_H×bev_W（预分配）→ ScatterElements 能做 ✅
    · 点云 pillarization：**有几根 pillar 由数据决定** → 需要原子分配编号 ❌
  两者都是"scatter"，区别只在**输出形状是否静态可知**。
  这才是判据，而不是"这个操作听起来复杂"。""")
    elif verdict == "add_ignored":
        print("""  TRT 接受了 ScatterElements 这个算子，但**忽略了 reduction='add'**，
  输出是覆盖写而不是累加 —— 能建、能跑、算错。

  这是本项目第五次遇到"能建引擎但输出错"，也是 C4 写 plugin 的**正当理由**：
  不是"TRT 没有这个算子"，而是"TRT 的实现语义不对"。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
