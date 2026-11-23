#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_int8.py —— P3 W3：INT8 量化（QDQ 格式）

⚠️ 为什么不用 IInt8EntropyCalibrator2：
   TensorRT 11 删掉了"隐式量化"的整套接口 ——
     grep IInt8Calibrator / setInt8Calibrator / BuilderFlag::kINT8 → 0 处
     trtexec --int8 → 也没了
   TRT 11 全面转向 strong typing：**精度必须由 ONNX 自己携带**。
   所以 INT8 要这样做：
     ① 用 onnxruntime 的量化工具生成 **QDQ 格式** ONNX
        （每个量化点插入 QuantizeLinear / DequantizeLinear，节点里带 scale 和 zero_point）
     ② TRT 直接解析这个 ONNX，自动构建 INT8 engine
   这和 kFP16 被删、改用 polygraphy --fp-to-fp16 是同一件事。

用法：
    # ① 先准备校准集（640×640 letterboxed PPM）
    python scripts/prepare_calib.py --src models/bus.jpg -n 100

    # ② 量化（per-channel，W4/W6 的默认方案）
    python scripts/quantize_int8.py --calib calib --per-channel

    # ③ per-tensor 对比（W6 的实验：per-channel vs per-tensor）
    python scripts/quantize_int8.py --calib calib --per-tensor --out models/yolov8n_int8_pt.onnx

    # ④ 校准集规模的曲线（W6）：--limit 控制用多少张
    python scripts/quantize_int8.py --calib calib --limit 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class PpmCalibReader:
    """把 calib/*.ppm 喂给 onnxruntime 的校准器

    PPM 已经是 640×640 letterboxed 的图，所以这里只做：
      HWC uint8 → CHW float32 /255
    这样和 C++ 侧的前处理完全一致（YOLO 只要 /255，没有 mean/std）。
    """

    def __init__(self, calib_dir: Path, input_name: str, size: int, limit: int = 0):
        from PIL import Image

        self.files = sorted(calib_dir.glob("*.ppm"))
        if limit > 0:
            self.files = self.files[:limit]
        if not self.files:
            raise RuntimeError(f"{calib_dir} 里没有 .ppm 文件，先跑 prepare_calib.py")

        self.input_name = input_name
        self.size = size
        self.index = 0
        self._Image = Image
        print(f"校准集: {len(self.files)} 张 ({size}×{size})")

    def get_next(self):
        if self.index >= len(self.files):
            return None
        p = self.files[self.index]
        self.index += 1

        img = self._Image.open(p).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), self._Image.BILINEAR)

        hwc = np.asarray(img, dtype=np.float32) / 255.0        # [H,W,3]
        chw = np.transpose(hwc, (2, 0, 1))[None, ...]          # [1,3,H,W]
        return {self.input_name: np.ascontiguousarray(chw)}

    def rewind(self):
        self.index = 0

    def __len__(self):
        return len(self.files)


def fold_int32_bias(model):
    """把 "int32 量化 bias 的 DequantizeLinear" 折成 float 常量

    为什么必须折：
      ONNX 的 QDQ 规范里，量化 Conv 的 bias 存成 **int32**（scale = w_scale × a_scale）。
      但 TensorRT 11 的 QDQ 解析器**不接受 int32 作为 DQ 的输入**：
        "input has type Int32 but must have type FP8, FP4, Int4, Int8, or UInt8"

    折叠是安全的：
      bias 只是一个一维向量，量化它省不了多少带宽；
      真正的量化收益在 Conv 权重和激活上，折掉 bias 不影响。

    做法：读 int32 值 + scale + zero_point → 算出 float bias → 换成 float initializer
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    graph = model.graph
    init_map = {i.name: i for i in graph.initializer}

    folded = 0
    new_inits = []
    to_remove = []

    for node in list(graph.node):
        if node.op_type != "DequantizeLinear":
            continue
        x_name = node.input[0]
        if x_name not in init_map:
            continue
        t = init_map[x_name]
        if t.data_type != TensorProto.INT32:
            continue

        # scale（可能 per-channel），zero_point（可能缺省）
        if len(node.input) < 2 or node.input[1] not in init_map:
            continue
        scales = numpy_helper.to_array(init_map[node.input[1]]).astype(np.float64)
        if len(node.input) > 2 and node.input[2] and node.input[2] in init_map:
            zps = numpy_helper.to_array(init_map[node.input[2]]).astype(np.float64)
        else:
            zps = np.zeros_like(scales)

        ints = numpy_helper.to_array(t).astype(np.float64)
        float_val = (ints - zps) * scales
        float_val = float_val.astype(np.float32)

        new_name = node.output[0] + "__f32"
        new_inits.append(numpy_helper.from_array(float_val, new_name))

        # 把这个 DQ 的输出全部改指向新的 float 常量
        for n2 in graph.node:
            for k, nm in enumerate(n2.input):
                if nm == node.output[0]:
                    n2.input[k] = new_name

        to_remove.append(node)
        folded += 1

    for n in to_remove:
        graph.node.remove(n)
    for t in new_inits:
        graph.initializer.append(t)

    # 删掉不再被引用的旧 initializer（可选的清理，不做也不影响正确性）
    return model, folded


def iou_xyxy(a, b) -> float:
    """两个 xyxy 框的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / max(ua + ub - inter, 1e-9)


def match_detections(d32, d8, iou_thres: float = 0.5):
    """按 IoU 匹配两边的检出框

    ⚠️ 不能按"排序后的索引"对应：
       两边的分数顺序不同，d32[0] 和 d8[0] 很可能不是同一个物体，
       直接相减会得出几百像素的假差异（实测 379 px，纯属比错了对象）。

    做法：对每个 FP32 框，在 INT8 里找同类别、IoU 最大且未被占用的框。
    返回 (匹配对列表, 只有 FP32 有的, 只有 INT8 有的)
    """
    used = set()
    pairs = []
    for a in d32:
        best_j, best_u = -1, iou_thres
        for j, b in enumerate(d8):
            if j in used or a[0] != b[0]:
                continue
            u = iou_xyxy(a[2:6], b[2:6])
            if u > best_u:
                best_u, best_j = u, j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((a, d8[best_j]))

    only32 = [a for a in d32 if not any(a is p[0] for p in pairs)]
    only8 = [b for j, b in enumerate(d8) if j not in used]
    return pairs, only32, only8


def decode_v8(y: np.ndarray, conf: float):
    """YOLOv8 decode（和 C++ 的 decodeV8 同逻辑）

    返回 [(cls, score, x1, y1, x2, y2), ...]，按分数降序
    注意通道在前：index(c, i) = c * 8400 + i
        索引 0=cls 1=score 2=x1 3=y1 4=x2 5=y2
    """
    p = y[0]                       # [84, 8400]
    anchors = p.shape[1]
    boxes = p[:4]                  # [4, 8400]  cx, cy, w, h
    scores = p[4:]                 # [80, 8400]

    cls = scores.argmax(axis=0)
    sc = scores.max(axis=0)
    keep = sc > conf

    b = boxes[:, keep]
    x1 = b[0] - b[2] * 0.5
    y1 = b[1] - b[3] * 0.5
    x2 = b[0] + b[2] * 0.5
    y2 = b[1] + b[3] * 0.5

    out = list(zip(cls[keep].tolist(), sc[keep].tolist(),
                   x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()))
    out.sort(key=lambda t: -t[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n.onnx", help="输入的 FP32 ONNX")
    ap.add_argument("--calib", default="calib", help="校准集目录（64×640 ppm）")
    ap.add_argument("--out", default="models/yolov8n_int8.onnx", help="输出的 QDQ ONNX")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 张校准（W6 实验用）")
    ap.add_argument("--per-channel", action="store_true", default=True,
                    help="权重量化按通道（默认，精度更好）")
    ap.add_argument("--per-tensor", dest="per_channel", action="store_false",
                    help="权重按张量量化（W6 的对比实验）")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="量化后用 ORT 跑一张图，对比 FP32 的误差")
    ap.add_argument("--ops", default="Conv,MatMul",
                    help="只量化这些算子类型（默认 Conv,MatMul）。"
                         "别把 Concat/Reshape/Constant 也量化了——"
                         "检测头里的锚点/stride 常量一量化，坐标就废了")
    ap.add_argument("--exclude-head", action="store_true", default=True,
                    help="把 /model.22/（Detect 头）的节点排除在量化之外")
    ap.add_argument("--fold-bias", action="store_true", default=True,
                    help="把 int32 量化 bias 折成 float（TRT 不认 int32）")
    ap.add_argument("--exclude-ids", default="",
                    help="额外排除这些 model 编号的层，如 '0-9' 或 '0-9,21'（W5 分组消融）")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    model_path = Path(args.model)
    calib_dir = Path(args.calib)
    out_path = Path(args.out)

    if not model_path.exists():
        print(f"[FAIL] 找不到 {model_path}")
        return 1
    if not calib_dir.exists():
        print(f"[FAIL] 找不到 {calib_dir}，先跑：")
        print(f"  python scripts/prepare_calib.py --src models/bus.jpg -n 100")
        return 1

    print("=" * 66)
    print("YOLOv8 INT8 量化（QDQ 格式）")
    print("=" * 66)
    print(f"onnxruntime: {ort.__version__}")
    print(f"输入模型   : {model_path}  ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    m = onnx.load(str(model_path))
    input_name = m.graph.input[0].name
    print(f"输入张量   : {input_name}")

    reader = PpmCalibReader(calib_dir, input_name, args.size, args.limit)

    class Reader(CalibrationDataReader):
        def __init__(self, r):
            self.r = r

        def get_next(self):
            return self.r.get_next()

        def rewind(self):
            self.r.rewind()

    print(f"量化格式   : QDQ (QuantizeLinear/DequantizeLinear)")
    print(f"权重精度   : {'per-channel' if args.per_channel else 'per-tensor'}")
    print(f"激活精度   : QInt8")
    op_types = [s.strip() for s in args.ops.split(",") if s.strip()]
    print(f"只量化算子 : {op_types}")

    # ---- 要排除的节点 ----
    # ① 检测头（/model.22/）默认排除
    #    理由：那里的 Constant 是锚点网格和 stride，量化后坐标精度直接崩
    # ② --exclude-ids 额外排除指定的 model 编号（W5 的分组消融用）
    exclude_ids: set[int] = set()
    if args.exclude_ids:
        for part in args.exclude_ids.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                exclude_ids.update(range(int(a), int(b) + 1))
            else:
                exclude_ids.add(int(part))

    import re as _re

    nodes_to_exclude = []
    for n in m.graph.node:
        if args.exclude_head and n.name.startswith("/model.22/"):
            nodes_to_exclude.append(n.name)
            continue
        if exclude_ids:
            mt = _re.match(r"^/model\.(\d+)/", n.name)
            if mt and int(mt.group(1)) in exclude_ids:
                nodes_to_exclude.append(n.name)

    if args.exclude_head:
        print("排除头部节点: /model.22/")
    if exclude_ids:
        print(f"额外排除     : model.{sorted(exclude_ids)}")
    print(f"共排除节点   : {len(nodes_to_exclude)} 个")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(model_path),
        model_output=str(out_path),
        calibration_data_reader=Reader(reader),
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        nodes_to_exclude=nodes_to_exclude,
        # ★ 关键：ORT 默认 ActivationSymmetric=False（uint8 非对称，zero_point≠0），
        #   而 TensorRT 11 只支持对称量化（zero_point=0）。
        #   不开这个，TRT 会在解析第一个 QDQ 节点时直接 assert 失败：
        #     "shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported"
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    print(f"\n已生成: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ---- 折叠 int32 bias（TRT 不认）----
    if args.fold_bias:
        qm0 = onnx.load(str(out_path))
        qm0, folded = fold_int32_bias(qm0)
        onnx.save(qm0, str(out_path))
        print(f"折叠 int32 bias: {folded} 个节点 → 换成 float 常量")
        print(f"  （TRT 的 QDQ 解析器不接受 int32 输入，必须折）")

    # ---- 检查 QDQ 节点数 ----
    qm = onnx.load(str(out_path))
    ops = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    n_q = ops.get("QuantizeLinear", 0)
    n_dq = ops.get("DequantizeLinear", 0)
    print(f"QDQ 节点   : QuantizeLinear={n_q}, DequantizeLinear={n_dq}")
    print(f"节点总数   : {len(qm.graph.node)}（原始 {len(m.graph.node)}）")

    # ---- 用 ORT 验一张图 ----
    #
    # ⚠️ 不要比 raw output 的 max|abs|：
    #    84×8400 = 705600 个数，最大值总落在某个低分 anchor 上（坐标本来就是乱的），
    #    那个数字会把"量化好不好"完全带偏（实测 37%，但实际检出几乎不变）。
    #    正确做法：**先 decode，再比框**——这才是业务关心的东西。
    if args.verify:
        print("\n[ORT 验证] 按 IoU 匹配后比框（不是按索引）")
        reader.rewind()
        feed = reader.get_next()

        # 关掉 ORT 的 INFO 级日志（折叠 bias 后会刷几百行孤儿常量警告）
        ort.set_default_logger_severity(3)

        s32 = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        s8 = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

        y32 = s32.run(None, feed)[0]
        y8 = s8.run(None, feed)[0]

        d32 = decode_v8(y32, conf=0.25)
        d8 = decode_v8(y8, conf=0.25)

        pairs, only32, only8 = match_detections(d32, d8, iou_thres=0.5)

        print(f"  FP32 检出 : {len(d32)} 个框")
        print(f"  INT8 检出 : {len(d8)} 个框")
        print(f"  匹配上    : {len(pairs)} 对（同类别 + IoU > 0.5）")
        print(f"  只有FP32  : {len(only32)} 个（INT8 漏检）")
        print(f"  只有INT8  : {len(only8)} 个（INT8 多检）")

        if pairs:
            print(f"\n  {'类别':>4s} {'FP32分数':>9s} {'INT8分数':>9s} "
                  f"{'分数差':>7s} {'坐标最大差(px)':>13s}")
            worst_cd, worst_sd = 0.0, 0.0
            for a, b in pairs[:10]:
                cd = max(abs(a[2] - b[2]), abs(a[3] - b[3]),
                         abs(a[4] - b[4]), abs(a[5] - b[5]))
                sd = abs(a[1] - b[1])
                worst_cd = max(worst_cd, cd)
                worst_sd = max(worst_sd, sd)
                print(f"  {int(a[0]):>4d} {a[1]:>9.3f} {b[1]:>9.3f} "
                      f"{sd:>7.3f} {cd:>13.1f}")
            print(f"\n  最大坐标差 = {worst_cd:.1f} px   最大分数差 = {worst_sd:.3f}")
            print(f"  判据：坐标 < 15 px 且 无漏检  → 可用")
        else:
            print("  ⚠️ 一对都没匹配上，量化可能已经坏了")

    print("""
下一步：
  # 用 C++ 建 INT8 engine（TRT 解析 QDQ ONNX，不需要 calibrator）
  ./build/bin/yolov8_demo --onnx models/yolov8n_int8.onnx \\
      --save-engine models/yolov8n_int8.engine --image models/bus.ppm

  # 和 FP32/FP16 交错对比（同一次运行内，可纵向比）
  ./build/bin/yolov8_demo --engine models/yolov8n.engine \\
      --engine2 models/yolov8n_int8.engine --image models/bus.ppm
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
