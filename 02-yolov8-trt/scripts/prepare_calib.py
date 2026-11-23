#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_calib.py —— 生成 INT8 校准集（P3 W3 的前置）

产出：calib/*.ppm，每张都是 **640×640 letterboxed** 的 RGB 图。
      这样 C++ 校准器只需要"读 PPM + /255"，不用再复现前处理。

为什么校准集要提前 letterbox：
    校准用的是"真实喂给模型的张量"，所以必须和推理时的输入完全一致。
    如果校准器自己再做一遍前处理，就容易和推理侧不一致——那是 precision 陷阱。

用法：
    # ① 从一张图增强出 N 张（能跑通管线，但多样性差，只适合调试）
    python scripts/prepare_calib.py --src models/bus.jpg -n 100

    # ② 从本地图片目录（推荐：真实场景的图）
    python scripts/prepare_calib.py --src /path/to/images -n 500

    # ③ 拿到 COCO val2017 之后
    python scripts/prepare_calib.py --src /path/to/val2017 -n 500

⚠️ 校准集的**分布**决定了 INT8 的量化区间：
   只用 bus.jpg 增强，得到的 scale 会偏向"街道场景"，
   在真实场景上可能掉点严重。生产必须用覆盖实际场景的 500–1000 张图。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


def letterbox(img: Image.Image, size: int, pad: int = 114) -> Image.Image:
    """等比缩放 + 居中填充（和 preprocess::letterbox 一致）"""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """轻度增强：模拟不同光照和构图，让量化区间更稳"""
    # 随机裁剪 85%–100%
    if rng.random() < 0.6:
        w, h = img.size
        r = rng.uniform(0.85, 1.0)
        cw, ch = int(w * r), int(h * r)
        x = rng.randint(0, w - cw)
        y = rng.randint(0, h - ch)
        img = img.crop((x, y, x + cw, y + ch))

    # 水平翻转
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 亮度 / 对比度
    if rng.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))

    return img


def collect_sources(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="一张图，或一个图片目录")
    ap.add_argument("-n", "--num", type=int, default=100, help="要生成多少张")
    ap.add_argument("--size", type=int, default=640, help="输入尺寸（和模型一致）")
    ap.add_argument("--pad", type=int, default=114, help="letterbox 填充值")
    ap.add_argument("--out", default="calib", help="输出目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true", help="不增强（原图直接 letterbox）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[FAIL] 找不到 {src}")
        return 1

    sources = collect_sources(src)
    if not sources:
        print(f"[FAIL] {src} 里没有图片")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成 INT8 校准集")
    print("=" * 60)
    print(f"源        : {src}  （{len(sources)} 个文件）")
    print(f"输出      : {out_dir}/  （{args.num} 张，{args.size}×{args.size}）")
    print(f"增强      : {'关' if args.no_augment else '开（裁剪/翻转/亮度）'}")

    rng = random.Random(args.seed)
    written = 0
    for i in range(args.num):
        p = sources[i % len(sources)]
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue

        if not args.no_augment:
            img = augment(img, rng)

        img = letterbox(img, args.size, args.pad)
        dst = out_dir / f"calib_{i:04d}.ppm"
        img.save(dst)
        written += 1

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{args.num}")

    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.ppm")) / 1024 / 1024
    print(f"\n完成：{written} 张，共 {total_mb:.1f} MB")

    if len(sources) < 20:
        print(f"""
⚠️ 警告：源图只有 {len(sources)} 张，增强出来的校准集**多样性不足**。
   INT8 的量化区间会偏向这几张图的场景，真实数据上可能掉点严重。
   生产必须用 500–1000 张覆盖实际场景的图。

   想要真实图：
     · COCO val2017（5000 张，约 1GB）：http://images.cocodataset.org/zips/val2017.zip
     · 或者用自己项目/车上录的图
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
