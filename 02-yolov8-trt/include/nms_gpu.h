#pragma once
// ---------------------------------------------------------------------------
// nms_gpu.h —— P3 W2：NMS 的 GPU 版
//
// 设计取舍（重要，面试会问）：
//
//   NMS 的"选择过程"本质是**串行**的：
//     选中分数最高的框 → 抑制和它重叠的 → 再选下一个
//     后一步依赖前一步的结果，无法并行。
//
//   能并行的是 **IoU 矩阵**（n² 个独立计算）。
//
//   所以本实现：
//     · GPU 负责算 IoU 矩阵（O(n²) 并行）
//     · CPU 负责贪心选择（串行，但只查表，不再算 IoU）
//
//   工业方案（TensorRT 的 EfficientNMS plugin）用的是
//   "bitmask + 多轮迭代"，能在 GPU 上完成完整流程；
//   那是 plugin 的内容（P5），这里手写的是教学版。
// ---------------------------------------------------------------------------
#include <vector>

#include "postprocess.h"

namespace yolo {

/// GPU 加速的 NMS
///
/// @param dets       输入/输出：会被按分数降序排序
/// @param iou_thres  通常 0.45
///
/// 结果必须和 nms() 逐位一致——这是 A/B 校验的判据。
std::vector<Detection> nmsGpu(std::vector<Detection>& dets, float iou_thres);

/// 只测 IoU 矩阵的耗时（不含传输、不含贪心），用于分析瓶颈
float benchIouMatrixGpu(const std::vector<Detection>& dets, float iou_thres,
                        int warmup, int iters);

}  // namespace yolo
