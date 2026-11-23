#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_preprocess.py —— 前处理正确性验证：我的 CUDA 实现 vs torchvision 标准实现

⚠️ 这个脚本回答的是 **"我写的前处理对不对"**，
   而 compare_precision.py 回答的是 **"TRT 和 PyTorch 数值等价吗"**。
   两者是不同的问题，缺一不可：

     compare_preprocess.py : 前处理 vs 标准实现   → 算法对不对
     compare_precision.py  : TRT     vs PyTorch    → 引擎等价不等价

   只做后者的话，两边喂的是"同一份（可能错的）输入"，所以永远发现不了前处理的错。

用法：
    # 1) C++ 侧导出原始图 + 前处理结果
    ./build/bin/resnet50_demo --engine models/resnet50.engine \
        --dump-raw dump/raw.bin --dump-input dump/in.bin --iters 2

    # 2) 比对
    ~/.venvs/torch/bin/python scripts/compare_preprocess.py \
        --raw dump/raw.bin --mine dump/in.bin
"""
from __future__ import annotations

import argparse
import struct

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_tensor(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        (nd,) = struct.unpack("i", f.read(4))
        dims = struct.unpack(f"{nd}i", f.read(4 * nd))
        n = int(np.prod(dims))
        data = np.frombuffer(f.read(4 * n), dtype=np.float32)
    return data.reshape(dims).copy()


def stats(a: np.ndarray, b: np.ndarray, label: str,
          atol: float = 1e-3) -> float:
    assert a.shape == b.shape, f"形状不一致 {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    md, mean = float(d.max()), float(d.mean())
    ndiff = int((d != 0).sum())
    verdict = "OK" if md < atol else "差异较大"
    print(f"  {label}")
    print(f"     max|diff| = {md:.3e}   mean|diff| = {mean:.3e}   [{verdict}]")
    print(f"     有差异元素 {ndiff}/{d.size} ({100.0 * ndiff / d.size:.1f}%)")
    return md


def build_reference(img_u8: np.ndarray, size: int, antialias: bool,
                    float_resize: bool):
    """用 torchvision 复现标准前处理，返回 (1,3,H,W) 的 numpy

    ⚠️ 关键：v2.Resize 会**保持输入 dtype**。
       uint8 进 → uint8 出（中间做了量化）
       想要 float 域 resize，必须**先 ToDtype(float) 再 Resize**
       —— 这正是我第一版脚本的 bug，导致两个变体结果一样。
    """
    import torch
    from torchvision.transforms import v2

    t = v2.ToImage()(img_u8)          # HWC uint8 numpy → CHW uint8 tensor

    if float_resize:
        # 先在 [0,1] 的 float 域 resize（无量化）—— 我们的实现走的是这条路
        t = v2.ToDtype(torch.float32, scale=True)(t)
        t = v2.Resize((size, size), antialias=antialias)(t)
    else:
        # 先在 uint8 域 resize（PIL 的行为，会量化回整数）再转 float
        t = v2.Resize((size, size), antialias=antialias)(t)
        t = v2.ToDtype(torch.float32, scale=True)(t)

    t = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(t)
    return t.unsqueeze(0).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="--dump-raw 导出的原始图 (H,W,3)")
    ap.add_argument("--mine", required=True, help="--dump-input 导出的 (1,3,H,W)")
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    raw = read_tensor(args.raw)
    mine = read_tensor(args.mine)

    img_u8 = np.clip(np.round(raw), 0, 255).astype(np.uint8)
    size = mine.shape[2]
    print("=" * 66)
    print("前处理正确性：我的 CUDA 实现 vs torchvision")
    print("=" * 66)
    print(f"原始图     : {img_u8.shape} (uint8, 0–255)  范围 [{img_u8.min()}, {img_u8.max()}]")
    print(f"我的输出   : {mine.shape}  范围 [{mine.min():.3f}, {mine.max():.3f}]\n")

    results = {}
    for label, antialias, float_resize in [
        ("torchvision  antialias=True ,  float 域 resize", True,  True),
        ("torchvision  antialias=True ,  uint8 域 resize", True,  False),
        ("torchvision  antialias=False,  float 域 resize", False, True),
        ("torchvision  antialias=False,  uint8 域 resize", False, False),
    ]:
        ref = build_reference(img_u8, size, antialias, float_resize)
        results[label] = stats(ref, mine, label, args.atol)
        print()

    best = min(results, key=results.get)
    print("=" * 66)
    print(f"最接近的变体：{best}")
    print(f"   max|diff| = {results[best]:.3e}")
    print("=" * 66)
    print("""
怎么读这个结果：

  · antialias=False + float 域 最小（~1e-3 或更好）
      → 你的双线性公式和 torchvision 一致，差异来自 uint8 量化

  · antialias=True 明显更小
      → torchvision 默认开了抗锯齿（对 PIL 输入是默认行为）
        你的实现少了这一步 → 缩小图像时丢高频 → 输入分布和训练时不同

  · 四档都差 >0.1
      → 公式不同（插值方式 / align_corners / 归一化参数 / 是否 /255）

  · antialias=True 那档差 2 以上
      → 说明抗锯齿在小图上的影响极大（2.86 倍缩小时它会明显改变像素）

⚠️ 部署判断：
   ResNet50 的官方推理脚本用的是 Resize(256) + CenterCrop(224) + antialias。
   我们这里是直接 Resize 到 224 且无抗锯齿 —— 两者语义不同。
   要复现官方精度，前处理必须照抄官方脚本，而不是"数学上看起来对"。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
