#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/ci_checks.py —— 仓库自检（可在本地直接跑，CI 里也跑同一份）

为什么这些检查值得做：
    这个仓库的"内容"是四份长 README（1200+ 行的项目②、581 行的项目③…）。
    长文档最常见的腐烂方式不是写错，是**交叉引用失效**——改了目录名、
    删了一个脚本，某处的 `[链接](路径)` 就静默指向不存在的文件，没人发现。
    （本项目在把 p1/p2/p3/p4 重命名成 01/02/03/04 时，就有 6 处这样的引用。）

检查项：
    ① Python 语法      —— 用 compile() 在内存里编，不落 __pycache__
    ② Markdown 相对链接 —— 所有 [x](y) 的本地目标必须存在
    ③ 仓库卫生          —— 不许混入 *.onnx / *.engine / *.pt / 大文件
    ④ 文本编码          —— 全部 UTF-8、且没有 CRLF

用法：
    python tools/ci_checks.py
    python tools/ci_checks.py --root .          # 指定根目录
退出码：0 = 全过；1 = 有问题
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---- ③ 不允许出现的产物类型（体积大且能由脚本重新生成）----
FORBIDDEN_EXT = {
    ".onnx", ".engine", ".plan", ".trt",
    ".pt", ".pth", ".safetensors", ".pyc",
}
FORBIDDEN_DIRS = {"build", "__pycache__", ".venv", "venv", "w1_out", "c_out"}
MAX_FILE_MB = 5.0

# ---- ② markdown 链接：[文字](目标) ----
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# 跳过这些：外链、锚点、mailto
SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "data:")

SKIP_DIRS = {".git", "build", "__pycache__", ".venv", "venv", "node_modules"}


def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS):
            yield p


def check_python(root: Path) -> list[str]:
    errs = []
    n = 0
    for p in iter_files(root):
        if p.suffix != ".py":
            continue
        n += 1
        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            errs.append(f"{p.relative_to(root)}: 不是 UTF-8（{e}）")
            continue
        try:
            compile(src, str(p), "exec")          # 只编不执行，不落盘
        except SyntaxError as e:
            errs.append(f"{p.relative_to(root)}:{e.lineno}: 语法错误: {e.msg}")
    print(f"[①] Python 语法      : {n} 个文件")
    return errs


def check_links(root: Path) -> list[str]:
    errs = []
    n_md = 0
    n_link = 0
    for p in iter_files(root):
        if p.suffix.lower() != ".md":
            continue
        n_md += 1
        text = p.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            t = target.strip()
            if t.startswith(SKIP_PREFIX):
                continue
            # 去掉标题锚点和尖括号写法
            t = t.split("#", 1)[0].strip().strip("<>").strip()
            if not t:
                continue
            n_link += 1
            resolved = (p.parent / t).resolve()
            if not resolved.exists():
                line = text[: text.find(target)].count("\n") + 1
                errs.append(f"{p.relative_to(root)}:{line}: 链接目标不存在 → {target}")
    print(f"[②] Markdown 相对链接: {n_md} 个文档 / {n_link} 条本地链接")
    return errs


def check_hygiene(root: Path) -> list[str]:
    errs = []
    n = 0
    for p in iter_files(root):
        n += 1
        rel = p.relative_to(root)
        if p.suffix.lower() in FORBIDDEN_EXT:
            errs.append(f"{rel}: 不该提交的产物类型（{p.suffix}）")
        if set(rel.parts) & FORBIDDEN_DIRS:
            errs.append(f"{rel}: 在应被忽略的目录里（{set(rel.parts) & FORBIDDEN_DIRS}）")
        mb = p.stat().st_size / 1048576
        if mb > MAX_FILE_MB:
            errs.append(f"{rel}: 文件过大 {mb:.1f} MB > {MAX_FILE_MB} MB")
    print(f"[③] 仓库卫生        : {n} 个文件")
    return errs


def check_encoding(root: Path, warns: list[str]) -> list[str]:
    errs = []
    n = 0
    for p in iter_files(root):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".ico"}:
            continue
        n += 1
        raw = p.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            errs.append(f"{p.relative_to(root)}: 不是合法 UTF-8")
            continue
        if b"\r\n" in raw:
            # 只记警告：.gitattributes 已用 `* text=auto eol=lf` 保证**仓库内**是 LF，
            # 本地在 Windows 上编辑时工作区可能是 CRLF，那是正常的。
            # 想让本地也干净：git add --renormalize . && git checkout -- .
            warns.append(f"{p.relative_to(root)}: 工作区含 CRLF（仓库内已被 .gitattributes 规范为 LF）")
    print(f"[④] 编码/换行       : {n} 个文本文件")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    print(f"仓库自检: {root}\n" + "-" * 58)
    all_errs: list[str] = []
    warns: list[str] = []
    all_errs += check_python(root)
    all_errs += check_links(root)
    all_errs += check_hygiene(root)
    all_errs += check_encoding(root, warns)
    print("-" * 58)

    if warns:
        print(f"\n[WARN]  {len(warns)} 条警告（不影响通过）：")
        for w in warns[:10]:
            print("  " + w)
        if len(warns) > 10:
            print(f"  … 还有 {len(warns) - 10} 条")

    if all_errs:
        print(f"\n[FAIL] {len(all_errs)} 个问题：\n")
        for e in all_errs[:60]:
            print("  " + e)
        if len(all_errs) > 60:
            print(f"  … 还有 {len(all_errs) - 60} 条")
        return 1
    print("\n[OK] 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
