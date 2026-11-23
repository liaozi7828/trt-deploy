#pragma once
// ---------------------------------------------------------------------------
// decode_nms_kernel.h —— decode + NMS 融合 kernel（纯 CUDA 接口，不含 STL）
//
// 目的：把 YOLOv8 的后处理整段搬进 GPU，让**引擎输出直接是最终框**。
//   现在：GPU 推理 → GPU decode → 回传 1.1 KB → CPU NMS
//   目标：GPU 推理 → GPU decode+NMS → 回传 120 B（且 CPU 完全不参与）
//
// 为什么输出形状固定 [K][6]：
//   NMS 的结果数量是**数据相关**的（46 也可能是 5 或 30），
//   但 TensorRT 要求每个输出张量的形状在编译期确定。
//   标准解法（TensorRT 自己的 EfficientNMS 也这样）：
//     输出 [1, K, 6] 固定容量 + 一个 [1] 的 int 表示真实数量。
//   K 取 100（YOLOv8 在 COCO 上实际很少超过 100 个框）。
//
// NMS 的并行化（贪心是串行的，但可以这么切）：
//   单个线程块，循环 K 次：
//     ① 全块扫候选 → 归约出当前最高分的未抑制框
//     ② 写出这个框
//     ③ 全块并行标记"同类且 IoU > 阈值"的候选
//   候选数 ≤ 1024 时，一轮只有几百次比较 → 微秒级。
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

namespace yolo {
namespace detail {

/// 候选缓冲区容量上限（= 一次推理里通过 conf 的最大数量）
constexpr int kMaxCandidates = 1024;
/// NMS 输出容量上限
constexpr int kMaxDetections = 100;
/// 单块线程数（NMS kernel 只用一个块）
constexpr int kNmsBlock = 256;

/// 第一步：每个 anchor 取 argmax + 阈值过滤 → 紧凑候选列表
///
/// @param d_cand   设备，[kMaxCandidates][6] = (x1,y1,x2,y2,score,cls)
/// @param d_count  设备 1 个 int，**调用前必须清零**
void launchGatherCandidates(const float* d_out, int num_classes, int num_anchors,
                            float conf_thres,
                            float* d_cand, int* d_count,
                            cudaStream_t stream = nullptr);

/// 第二步：贪心 NMS（单块并行）
///
/// @param d_cand     设备，[n_cand][6]（launchGatherCandidates 的输出）
/// @param d_dets     设备，[kMaxDetections][6] 输出，超出数量的行由主机清零
/// @param d_num_det  设备 1 个 int，**调用前必须清零**；返回真实数量
void launchNms(const float* d_cand, int n_cand, float iou_thres,
               float* d_dets, int* d_num_det,
               cudaStream_t stream = nullptr);

}  // namespace detail
}  // namespace yolo
