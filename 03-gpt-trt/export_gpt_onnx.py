#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_gpt_onnx.py —— 把 gpt.py 导成 ONNX，并逐步验证

这是 P1-a W3 的交付物：
    「导出一个 dynamic batch + dynamic seq 的模型，onnxsim 化简成功」

用法:
    cd ~/projects/p1
    ~/.venvs/torch/bin/python export_gpt_onnx.py

产出:
    w1_out/gpt.onnx            原始导出
    w1_out/gpt.sim.onnx        onnxsim 化简后（装了 onnxsim 才有）

你会在这份脚本里看到四件事，对应四个知识点:
    1. dynamic_axes  —— 让 batch 和 seq 两个维度都不固定
    2. 包装类        —— 导出要求 forward 的参数/返回值尽量简单
    3. 多 shape 验证  —— 真正验证"动态"是不是生效了（只测一个 shape 等于没测）
    4. 图对比        —— 原始图 vs 化简图，算子数和文件大小差多少
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from gpt import GPT, GPTConfig, DEFAULT_CFG


# =============================================================================
# 包装类：只输出 logits
# =============================================================================
class LogitsOnly(torch.nn.Module):
    """ONNX 导出的第一个坑：forward 的签名要简单。

    原始 GPT.forward 是 (idx, kv_caches, use_cache) -> (logits, new_caches)，
    返回值里有 list 和 tuple 混合，KV cache 还是"输入输出同名但形状变"的结构，
    导出器处理起来很容易出问题（而且第一次没必要导 cache）。

    所以包一层：只吃 idx，只吐 logits。
    """

    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx, kv_caches=None, use_cache=False)
        return logits


# =============================================================================
# 导出
# =============================================================================
def export_onnx(model: torch.nn.Module, path: Path, vocab_size: int,
                opset: int = 17) -> str:
    model.eval()
    dummy = torch.randint(0, vocab_size, (1, 8))   # 形状随便，dynamic_axes 会覆盖

    kwargs = dict(
        input_names=["input_ids"],
        output_names=["logits"],
        # 两个动态维度：batch 和 seq。名字会写进 ONNX 的 dim_param
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits":    {0: "batch", 1: "seq"},
        },
        opset_version=opset,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
        return "legacy (dynamo=False)"
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)
        return "default (dynamo)"


# =============================================================================
# 校验
# =============================================================================
def inspect(path: Path) -> None:
    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    print(f"  onnx.checker    : 通过")
    print(f"  文件大小        : {path.stat().st_size/1024:.1f} KB")
    print(f"  节点数          : {len(m.graph.node)}")
    print(f"  算子种类        : {sorted({n.op_type for n in m.graph.node})}")

    # 输入/输出的动态维度标注
    for io, label in ((m.graph.input, "input"), (m.graph.output, "output")):
        for t in io:
            dims = []
            for d in t.type.tensor_type.shape.dim:
                dims.append(d.dim_param if d.dim_param else d.dim_value)
            print(f"  {label:15s} : {t.name} {dims}")

    # 教学点：Range 算子的有无，直接决定这张图好不好部署
    ops = {n.op_type for n in m.graph.node}
    if "Range" in ops:
        print("  [!] 图里有 Range 算子 —— TRT 支持较弱，考虑改用切片写法")
    else:
        print("  [OK] 没有 Range 算子，TRT 友好")


def verify(path: Path, model: torch.nn.Module, vocab_size: int) -> float:
    """在多个 (batch, seq) 组合上对比 PyTorch 与 onnxruntime。

    只测一个 shape 是自欺欺人：动态维度可能只在一个取值上碰巧对了。
    """
    import onnxruntime as ort

    print(f"  onnxruntime     : {ort.__version__}")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    model.eval()
    worst = 0.0
    print(f"\n  {'batch':>5} {'seq':>5} {'max|diff|':>12}")
    print("  " + "-" * 26)
    with torch.no_grad():
        for bs, seq in [(1, 1), (1, 4), (2, 8), (3, 16), (1, 32)]:
            ids = torch.randint(0, vocab_size, (bs, seq))
            y_torch = model(ids).cpu().numpy()
            y_ort = sess.run(None, {"input_ids": ids.numpy()})[0]
            d = float(abs(y_torch - y_ort).max())
            worst = max(worst, d)
            flag = "OK" if d < 1e-4 else "FAIL"
            print(f"  {bs:5d} {seq:5d} {d:12.3e}  {flag}")
    return worst


def simplify(src: Path, dst: Path) -> None:
    try:
        import onnx
        from onnxsim import simplify as onnxsim_simplify
    except ImportError:
        print("  跳过（没装 onnxsim，可 pip install onnxsim）")
        return

    m = onnx.load(str(src))
    simplified, ok = onnxsim_simplify(m)
    if not ok:
        print("  [WARN] onnxsim 化简失败，保留原图")
        return
    onnx.save(simplified, str(dst))
    print(f"  化简成功: {dst.stat().st_size/1024:.1f} KB "
          f"（原 {src.stat().st_size/1024:.1f} KB）")


# =============================================================================
def main() -> int:
    out_dir = Path("w1_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ★ 用部署配置（大模型）——导出正确性和模型大小无关，
    #   但下一步要测「seq_len 对延迟的影响」，必须让计算量可见。
    cfg = DEFAULT_CFG
    model = GPT(cfg).eval()

    print("=" * 66)
    print("gpt.py → ONNX 导出")
    print("=" * 66)
    print(f"配置: layer={cfg.n_layer} head={cfg.n_head} embd={cfg.n_embd} "
          f"block={cfg.block_size} vocab={cfg.vocab_size}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    wrapper = LogitsOnly(model).eval()
    onnx_path = out_dir / "gpt.onnx"

    print("\n[1/4] 导出")
    how = export_onnx(wrapper, onnx_path, cfg.vocab_size)
    print(f"  导出器: {how}")

    print("\n[2/4] 检查图结构")
    inspect(onnx_path)

    print("\n[3/4] 多 shape 精度验证")
    # 必须传 wrapper：GPT.forward 返回 (logits, caches) 元组，
    # 直接传 GPT 会在 .cpu() 那步炸掉
    worst = verify(onnx_path, wrapper, cfg.vocab_size)
    if worst < 1e-4:
        print(f"\n  [OK] 最大误差 {worst:.3e} < 1e-4，动态维度确实生效")
    else:
        print(f"\n  [FAIL] 最大误差 {worst:.3e}，动态维度可能没生效")

    print("\n[4/4] onnxsim 化简")
    simplify(onnx_path, out_dir / "gpt.sim.onnx")

    print("\n产出:")
    for p in sorted(out_dir.glob("gpt*.onnx")):
        print(f"  {p}  ({p.stat().st_size/1024:.1f} KB)")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
