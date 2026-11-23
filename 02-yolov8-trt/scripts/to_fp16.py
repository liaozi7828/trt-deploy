#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
to_fp16.py —— 把 ONNX 转成 FP16（产出一个 **ORT 和 TRT 都认**的模型）

背景：这个坑踩过两次，两次报错一模一样
--------------------------------------------------
    Type Error: Type (tensor(float)) of output arg (/model.10/Resize_output_0)
    of node (/model.10/Resize_output_cast0) does not match expected type (tensor(float16))

发生了什么：
    YOLO 的 neck 里有个 Resize（上采样），它的输出名字 `/model.10/Resize_output_0`
    被转换器拿去当**新插入的 Cast 节点**的输出名，但那个 Cast 的 value_info 类型
    没跟着改成 float16 —— 于是模型自相矛盾（图里说是 float，下游要 float16）。

为什么 `op_block_list=["Resize"]` 救不了：
    把 Resize 留在 fp32 之后，转换器**必须在它后面插一个 Cast** 才能接回 fp16 主图，
    而这个"插 Cast"的路径正是出 bug 的那条。**所以 block 列表越全，越容易触发。**

解法：换 ORT 自己的转换器
    onnxruntime.transformers.float16.convert_float_to_float16
    它用的是另一套实现（先做符号形状推导，再按子图处理），对 Resize/Shape 这类
    "形状相关"算子处理正确。

本脚本的做法（重要）：
    不是"转完就信"，而是**转完立刻让 ORT 试着加载**，失败了就换下一个 backend。
    三个 backend 依次试，第一个能加载的胜出：
        ① ort  —— onnxruntime.transformers.float16（首选，本项目已验证可用）
        ② occ  —— onnxconverter_common（旧路径，Resize 上会出错）
        ③ none —— 不转换，直接复制（只用于诊断：确认源模型本身能加载）

用法：
    python scripts/to_fp16.py --model models/yolov8n.onnx --out models/yolov8n_fp16.onnx
    python scripts/to_fp16.py --backend ort      # 只用 ORT 的转换器
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def convert_ort(m, block, keep_io):
    """ORT 的 convert_float_to_float16（函数式）。

    ★ 不同 ORT 版本的**签名不一样**（1.31 就没有 use_symbolic_shape_infer），
      所以这里用 inspect 查一下，只传这个版本认识的参数。
      —— 之前直接写死关键字参数，换台机器/升个版本就 TypeError。
    """
    import inspect
    from onnxruntime.transformers.float16 import convert_float_to_float16

    sig = inspect.signature(convert_float_to_float16)
    print(f"  实际签名: {sig}")
    p = sig.parameters
    kw = {}
    if "keep_io_types" in p:
        kw["keep_io_types"] = keep_io
    if "op_block_list" in p:
        kw["op_block_list"] = block
    if "use_symbolic_shape_infer" in p:
        kw["use_symbolic_shape_infer"] = True
    return convert_float_to_float16(m, **kw)


def convert_ort_model(m, block, keep_io):
    """ORT 的 OnnxModel 方法式 API（最稳，支持 use_symbolic_shape_infer）"""
    from onnxruntime.transformers.onnx_model import OnnxModel

    om = OnnxModel(m)
    om.convert_float_to_float16(
        keep_io_types=keep_io,
        use_symbolic_shape_infer=True,
        op_block_list=block,
    )
    return om.model


def convert_occ(m, block, keep_io):
    """onnxconverter-common（旧路径，保留以便对比）"""
    from onnxconverter_common import float16
    return float16.convert_float_to_float16(
        m, keep_io_types=keep_io, op_block_list=block)


def convert_none(m, block, keep_io):
    """不转换（诊断用：确认源模型能加载）"""
    return m


BACKENDS = {"ort": convert_ort, "ort_model": convert_ort_model,
            "occ": convert_occ, "none": convert_none}


def try_load(path: Path):
    """返回 (ok, 说明)"""
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    try:
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        i, o = s.get_inputs()[0], s.get_outputs()[0]
        return True, f"输入 {i.name} {i.shape} {i.type} / 输出 {o.name} {o.shape} {o.type}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n.onnx")
    ap.add_argument("--out", default="models/yolov8n_fp16.onnx")
    ap.add_argument("--backend", default="auto", choices=["auto", "ort", "occ", "none"])
    ap.add_argument("--keep-op-fp32",
                    default="Resize,Upsample,Range,NonMaxSuppression",
                    help="这些算子保持 fp32（逗号分隔）")
    ap.add_argument("--no-keep-io-types", action="store_true",
                    help="默认输入输出保持 float32；加这个则也转成 fp16")
    args = ap.parse_args()

    import onnx

    src, dst = Path(args.model), Path(args.out)
    if not src.exists():
        print(f"[FAIL] 找不到 {src}")
        return 1
    keep_io = not args.no_keep_io_types
    block = [s.strip() for s in args.keep_op_fp32.split(",") if s.strip()]

    print("=" * 66)
    print("ONNX → FP16（转完立刻用 ORT 加载验证，不通过就换 backend）")
    print("=" * 66)
    print(f"输入     : {src}  ({src.stat().st_size / 1048576:.1f} MB)")
    print(f"保持 fp32: {block}")
    print(f"IO 类型  : {'保持 float32' if keep_io else '一并转 fp16'}")

    m = onnx.load(str(src))

    order = (["ort", "ort_model", "occ", "none"]
             if args.backend == "auto" else [args.backend])

    with tempfile.TemporaryDirectory() as td:
        for b in order:
            print(f"\n--- backend = {b} ---")
            try:
                m2 = BACKENDS[b](m, block, keep_io)
            except Exception as e:
                print(f"  转换抛异常: {type(e).__name__}: {str(e)[:160]}")
                continue

            if b == "ort":
                # ORT 转换器会顺手做一遍形状推导，出来的模型更"结实"
                try:
                    m2 = onnx.shape_inference.infer_shapes(m2)
                except Exception as e:
                    print(f"  （形状推导跳过: {str(e)[:80]}）")

            tmp = Path(td) / f"{b}.onnx"
            onnx.save(m2, str(tmp))
            ok, msg = try_load(tmp)
            print(f"  ORT 加载: {'OK' if ok else 'FAIL'} — {msg}")
            if not ok:
                continue

            # ★★ `none` 是诊断用的：它能加载只说明**源模型**没问题，
            #    绝不代表"转换成功"。写进 --out 会得到一个"文件名叫 fp16、内容是 fp32"的假模型，
            #    而且它会安静地产生"FP16 零掉点"这种看起来合理的假结论。
            if b == "none":
                print("  [诊断] 源模型可加载（说明问题在转换器，不在源模型）")
                print("         但这是 fp32，**不写入 --out**")
                continue

            # 成功：写到目标路径，并用最终路径再验一次
            shutil.copyfile(tmp, dst)
            ok2, msg2 = try_load(dst)
            mb_src = src.stat().st_size / 1048576
            mb_dst = dst.stat().st_size / 1048576
            ratio = mb_dst / mb_src
            print(f"  写入 {dst}  ({mb_dst:.1f} MB, 源 {mb_src:.1f} MB, "
                  f"比 {ratio:.2f})  复验: {'OK' if ok2 else 'FAIL'}")
            # ★ 体积自检：真 fp16 大约是 fp32 的 45%~60%。接近 1.0 说明压根没转换。
            if ratio > 0.85:
                print(f"  [WARN] 体积比 {ratio:.2f} 太接近 1 —— 这不像真的转成了 fp16")
            if not ok2:
                continue

            print(f"\n[OK] 用 backend={b} 转换成功（体积比 {ratio:.2f}）")
            print(f"""
下一步（算 mAP，只跑这一档）：
  python scripts/eval_map_coco.py --data ~/datasets/coco \\
      --models {dst} --base yolov8n.onnx --out map_coco_results.json
""")
            return 0

    print("\n[FAIL] 所有 backend 都产不出 ORT 能加载的模型")
    print("  建议：先用 --backend none 确认源模型本身可加载；")
    print("        再试 --no-keep-io-types（IO 也转 fp16）；")
    print("        或把 Resize 从 op_block_list 里去掉（让 ORT 转换器自己决定）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
