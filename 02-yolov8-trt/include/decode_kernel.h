#pragma once
// ---------------------------------------------------------------------------
// decode_kernel.h —— decode 的**纯 CUDA 接口**（只传裸指针，不含 STL）
//
// ⚠️ 会被 .cu include，而 .cu 编译时带 -U_GNU_SOURCE
//    （绕开 glibc 2.41 的 cospi 冲突），所以不能出现 <vector>/<string>/<algorithm>。
//
// 分层和 NMS 一样：
//   decode_kernel.h / decode_kernel.cu  ← kernel 与启动函数（本文件）
//   decode_gpu.h    / decode_gpu.cpp    ← 面向 std::vector 的封装（g++ 编译）
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

namespace yolo {
namespace detail {

/// 启动 decode kernel
///
/// 输入 out 的形状是 [4+nc, num_anchors]（**通道在前**），
/// 每个 anchor 做一次 argmax + 阈值过滤，通过的写入 dets（紧凑排列）。
///
/// @param d_out     设备指针，[4+nc][num_anchors] 的 float
/// @param d_dets    设备输出，[max_dets][6]，每项 = x1,y1,x2,y2,score,cls
/// @param d_count   设备上的 1 个 int，调用前必须清零；返回命中数
/// @param max_dets  容量上限（通常给 num_anchors，最坏情况全命中）
void launchDecode(const float* d_out, int num_classes, int num_anchors,
                  float conf_thres, float* d_dets, int* d_count,
                  int max_dets, cudaStream_t stream = nullptr);

}  // namespace detail
}  // namespace yolo
