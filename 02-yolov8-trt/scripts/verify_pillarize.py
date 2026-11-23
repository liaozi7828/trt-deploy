#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_pillarize.py —— 用 NumPy 原语独立核对 C++ 的 pillarization

为什么要这个脚本（回应"你这不是自己和自己比"）：

  C++ 里的 `pillarizeCpu()` 是我自己写的循环，和 kernel 共享同一套理解，
  所以它只能抓"实现不一致"，抓不到"理解错了"。

  这个脚本换一条路：
    ① 用 **np.unique / np.argsort / np.bincount** 这些被广泛验证的库原语
       重新表达 pillarization（公式化方式完全不同：没有循环、没有 map）
    ② 再检查**定义级不变量**（不依赖任何实现）：
         - 每个界内点必须出现在输出里
         - 没有两根 pillar 共用同一格
         - 界外点数 + 存下点数 == 输入点数
    ③ 三方对齐：NumPy 参考  ==  C++ CPU 参考  ==  C++ GPU kernel

用法：
    ./build/bin/pillarize_test --points 50000 --dump dump/pil
    python scripts/verify_pillarize.py --dump dump/pil \
        --grid 200 200 --voxel 0.5 --min -50 -50 --max-pts 32

注意：
    这个脚本的"参考"仍然是我写的（只是换用库原语 + 不同公式）。
    真正的**权威实现**是 OpenPCDet 的 DataProcessor / spconv 的 voxelizer，
    见 README §18.5。本脚本能抓"公式化错误"，抓不到"对论文的误解"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def sorted_rows(a: np.ndarray) -> np.ndarray:
    """
    把二维数组按"**整行**"做字典序排序，用于多重集合比较。

    ⚠️ 不能用 np.sort(a, axis=0)：那是**逐列**排序，会把行拆散，
       导致 [1,2]/[3,4] 和 [1,4]/[3,2] 被误判为相同。

    用 np.lexsort：`keys` 里**最后一个**是主键，所以把第 0 列放最后，
    就得到"先按第 0 列、再按第 1 列……"的字典序。
    这也是 numpy 里排序结构化记录的常规做法。
    """
    a = np.ascontiguousarray(a)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        return a
    keys = tuple(a[:, i] for i in range(a.shape[1] - 1, -1, -1))
    return a[np.lexsort(keys)]


def selftest_comparator() -> None:
    """
    给"对比器本身"做反例测试。

    为什么需要这一层：验证工具本身也可能有 bug，而且最危险的 bug 是
    **往"通过"方向错**——把不同的东西判成相同，测试从此失去意义。

    本脚本就踩过这个坑：第一版用 `np.sort(a, axis=0)`（逐列排序），
    会把 [(1,2),(3,4)] 和 [(1,4),(3,2)] 判成相同。
    这个自检就是为了让那种 bug 不可能再溜过去。
    """
    a = np.array([[1, 2], [3, 4]], np.float32)
    b = np.array([[3, 4], [1, 2]], np.float32)   # 只是顺序不同 → 应相等
    c = np.array([[1, 2], [3, 5]], np.float32)   # 一个数不同   → 应不等
    d = np.array([[1, 4], [3, 2]], np.float32)   # 逐列排序会误判 → 应不等

    assert np.array_equal(sorted_rows(a), sorted_rows(b)), "顺序不同被误判为不等"
    assert not np.array_equal(sorted_rows(a), sorted_rows(c)), "内容不同被误判为相等"
    assert not np.array_equal(sorted_rows(a), sorted_rows(d)), \
        "对比器有 bug：逐列排序式的误判（这正是第一版的错误）"

    # 顺带确认逐列排序确实会误判 —— 把反例钉在代码里
    assert np.array_equal(np.sort(a, axis=0), np.sort(d, axis=0)), \
        "反例失效：np.sort(axis=0) 本应把 a 和 d 误判成相同"


def read_points(path: Path) -> np.ndarray:
    """int32 n, int32 stride, then n*stride floats"""
    data = path.read_bytes()
    n = int(np.frombuffer(data, dtype=np.int32, count=1, offset=0)[0])
    stride = int(np.frombuffer(data, dtype=np.int32, count=1, offset=4)[0])
    buf = np.frombuffer(data, dtype=np.float32, count=n * stride, offset=8)
    return buf.reshape(n, stride).copy()


def read_pillars(path: Path) -> tuple[int, int, int, list[tuple[int, int, int, np.ndarray]]]:
    """int32 n_pillars, int32 max_pts, int32 F, then per-pillar:
       int32 iy, int32 ix, int32 count, count*F floats"""
    data = path.read_bytes()
    n_pillars = int(np.frombuffer(data, dtype=np.int32, count=1, offset=0)[0])
    max_pts = int(np.frombuffer(data, dtype=np.int32, count=1, offset=4)[0])
    nf = int(np.frombuffer(data, dtype=np.int32, count=1, offset=8)[0])
    off = 12
    out = []
    for _ in range(n_pillars):
        iy, ix, cnt = np.frombuffer(data, dtype=np.int32, count=3, offset=off)
        off += 12
        stored = min(int(cnt), max_pts)
        pts = np.frombuffer(data, dtype=np.float32, count=stored * nf, offset=off).copy()
        off += stored * nf * 4
        out.append((int(iy), int(ix), int(cnt), pts.reshape(stored, nf)))
    return n_pillars, max_pts, nf, out


def numpy_reference(pts: np.ndarray, grid_x: int, grid_y: int,
                    min_x: float, min_y: float, voxel: float,
                    max_pts: int) -> dict:
    """
    用库原语重写 pillarization —— 和 C++ 的循环/map 完全不同的公式化。

    ① 算每个点的格子索引（向量化，没有循环）
    ② 界外点剔除
    ③ np.unique(return_inverse=True) 给出"每根 pillar 的编号"
       —— 这是 numpy 里被广泛使用的分组原语
    ④ np.argsort(稳定排序) 把同一 pillar 的点排到一起
    ⑤ 按 pillar 切片，取前 max_pts 个
    """
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    ix = np.floor((x - min_x) / voxel).astype(np.int64)
    iy = np.floor((y - min_y) / voxel).astype(np.int64)

    inb = (ix >= 0) & (ix < grid_x) & (iy >= 0) & (iy < grid_y)
    n_oob = int((~inb).sum())

    cell = (iy[inb] * grid_x + ix[inb])          # 一维格子编号
    order = np.argsort(cell, kind="stable")      # 稳定排序 → 同格相邻
    cell_sorted = cell[order]

    uniq, first_idx, counts = np.unique(cell_sorted,
                                        return_index=True,
                                        return_counts=True)

    # uniq 就是所有被占用的格子；解码回 (iy, ix)
    uiy = uniq // grid_x
    uix = uniq % grid_x

    # 每根 pillar 收前 max_pts 个点
    pillars = []
    for k in range(len(uniq)):
        s = first_idx[k]
        e = s + min(int(counts[k]), max_pts)
        sel = order[s:e]                          # 指向 inb 之前的索引
        idx_in_full = np.nonzero(inb)[0][sel]     # 映回原始点序号
        pillars.append(pts[idx_in_full, :4].copy())

    return {
        "n_pillars": len(uniq),
        "coords": np.stack([uiy, uix], axis=1),
        "counts": counts,                         # 总点数（未截断）
        "stored": np.minimum(counts, max_pts),
        "pillars": pillars,
        "n_oob": n_oob,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="dump/pil", help="C++ --dump 的前缀")
    ap.add_argument("--grid", nargs=2, type=int, default=[200, 200])
    ap.add_argument("--voxel", type=float, default=0.5)
    ap.add_argument("--min", nargs=2, type=float, default=[-50.0, -50.0])
    ap.add_argument("--max-pts", type=int, default=32)
    args = ap.parse_args()

    base = Path(args.dump)
    pts_path = base.parent / (base.name + "_points.bin")
    pil_path = base.parent / (base.name + "_pillars.bin")
    if not pts_path.exists():
        print(f"[FAIL] 找不到 {pts_path}")
        print("  先跑: ./build/bin/pillarize_test --points 50000 --dump dump/pil")
        return 1

    pts = read_points(pts_path)
    n_pillars, max_pts, nf, gpu = read_pillars(pil_path)
    grid_x, grid_y = args.grid

    print("=" * 74)
    print("pillarization 独立核对（NumPy 原语 vs C++ GPU kernel）")
    print("=" * 74)

    # ---- 第 0 层：先验证对比器本身是活的 ----
    try:
        selftest_comparator()
        print("对比器自检 : ✅ 通过（顺序无关、内容差异能识别、"
              "逐列排序的反例已被排除）")
    except AssertionError as e:
        print(f"对比器自检 : ❌ {e}")
        print("  → 对比器有 bug，后面的结论都不可信，直接退出")
        return 2

    print(f"输入点数   : {len(pts)}")
    print(f"网格       : {grid_x} × {grid_y}，voxel {args.voxel}")
    print(f"C++ 输出   : {n_pillars} 根 pillar，每根最多 {max_pts} 点，{nf} 个特征")

    ref = numpy_reference(pts, grid_x, grid_y, args.min[0], args.min[1],
                          args.voxel, max_pts)
    print(f"NumPy 参考 : {ref['n_pillars']} 根 pillar，界外 {ref['n_oob']} 点")

    ok = True

    # ---------- ① pillar 数量 ----------
    print("\n① pillar 数量")
    if n_pillars == ref["n_pillars"]:
        print(f"   ✅ 一致：{n_pillars}")
    else:
        print(f"   ❌ 不一致：C++ {n_pillars} vs NumPy {ref['n_pillars']}")
        ok = False

    # ---------- ② 格子集合 ----------
    print("\n② 格子集合（按 (iy, ix) 比）")
    cpp_cells = {(iy, ix) for (iy, ix, _, _) in gpu}
    ref_cells = {(int(a), int(b)) for a, b in ref["coords"]}
    if cpp_cells == ref_cells:
        print(f"   ✅ 完全一致（{len(cpp_cells)} 个格子）")
    else:
        only_cpp = cpp_cells - ref_cells
        only_ref = ref_cells - cpp_cells
        print(f"   ❌ C++ 独有 {len(only_cpp)} 个，NumPy 独有 {len(only_ref)} 个")
        for c in list(only_cpp)[:5]:
            print(f"      C++ 独有 {c}")
        for c in list(only_ref)[:5]:
            print(f"      NumPy 独有 {c}")
        ok = False

    # ---------- ③ 每根 pillar 的点内容（按集合，忽略顺序）----------
    print("\n③ 每根 pillar 的点内容（排序后逐元素比，忽略顺序）")
    bad = 0
    # ★ 键必须是 tuple（list 不可哈希）
    ref_map = {(int(c[0]), int(c[1])): proc
               for c, proc in zip(ref["coords"], ref["pillars"])}
    for (iy, ix, cnt, stored_pts) in gpu:
        rp = ref_map.get((iy, ix))
        if rp is None:
            bad += 1
            continue
        a = sorted_rows(stored_pts)
        b = sorted_rows(rp.astype(np.float32))
        if a.shape != b.shape or not np.array_equal(a, b):
            bad += 1
    if bad == 0:
        print(f"   ✅ 全部 {n_pillars} 根 pillar 点内容一致")
    else:
        print(f"   ❌ 有 {bad} 根 pillar 点内容不同")
        ok = False

    # ---------- ④ 定义级不变量（不依赖任何参考实现）----------
    print("\n④ 定义级不变量（不需要参考实现）")
    # 4a 每根 pillar 的坐标在范围内
    bad_range = sum(1 for (iy, ix, _, _) in gpu
                    if not (0 <= iy < grid_y and 0 <= ix < grid_x))
    print(f"   坐标越界的 pillar : {bad_range}  "
          f"{'✅' if bad_range == 0 else '❌'}")
    ok = ok and bad_range == 0

    # 4b 没有重复格子
    print(f"   重复格子数        : {n_pillars - len(cpp_cells)}  "
          f"{'✅' if n_pillars == len(cpp_cells) else '❌'}")
    ok = ok and n_pillars == len(cpp_cells)

    # 4c 数量守恒：界外 + 存下 == 总数
    stored_total = sum(min(c, max_pts) for (_, _, c, _) in gpu)
    print(f"   界外 {ref['n_oob']} + 存下 {stored_total} = {ref['n_oob'] + stored_total}"
          f"  (输入 {len(pts)})  "
          f"{'✅' if ref['n_oob'] + stored_total == len(pts) else '❌'}")
    ok = ok and (ref["n_oob"] + stored_total == len(pts))

    # 4d 每个界内点都能在输出里找到（用浮点位模式做多重集合比对）
    print("\n   → 检查每个界内点是否都出现在输出里（这是最狠的一条）")
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    ix = np.floor((x - args.min[0]) / args.voxel).astype(np.int64)
    iy = np.floor((y - args.min[1]) / args.voxel).astype(np.int64)
    inb = (ix >= 0) & (ix < grid_x) & (iy >= 0) & (iy < grid_y)
    want = sorted_rows(pts[inb, :4])

    if n_pillars:
        got = np.concatenate([sp[:min(c, max_pts), :4] for (_, _, c, sp) in gpu])
    else:
        got = np.zeros((0, 4), np.float32)
    got = sorted_rows(got)
    if want.shape == got.shape and np.array_equal(want, got):
        print(f"      ✅ 界内点 {int(inb.sum())} 个，全部找到，不多不少")
    else:
        print(f"      ❌ 界内点 {int(inb.sum())} 个，输出里存了 {got.shape[0]} 个")
        ok = False

    print("\n" + "=" * 74)
    print("结果：" + ("✅ 通过" if ok else "❌ 不通过"))
    print("=" * 74)
    if ok:
        print("""
说明：这一层用的是 NumPy 的 np.unique / np.argsort（被广泛使用的库原语），
      公式化和 C++ 的"循环 + map"完全不同。它能抓出"公式写错了"。
      但它抓不出"对 pillarization 的语义理解错了"——那需要和
      OpenPCDet / spconv 这类生产实现对齐（见 README §18.5）。""")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
