#pragma once
// ---------------------------------------------------------------------------
// nms_kernel.h —— GPU NMS 的**纯 CUDA 接口**（只传裸指针，不含 STL）
//
// ⚠️ 这个头会被 .cu 文件 include，而 .cu 编译时带 -U_GNU_SOURCE
//    （绕开 glibc 2.41 的 cospi 冲突），导致 <string>/<vector>/<algorithm>
//    这些头编译失败（<cwchar> 缺声明）。
//
//    所以：**device 代码只用裸指针通信，STL 容器留给 .cpp**。
//    这就是为什么 NMS 被拆成：
//      nms_kernel.h / nms_kernel.cu  ← 纯 CUDA（本文件）
//      nms_gpu.cpp                   ← 面向 std::vector 的封装（g++ 编译）
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

namespace yolo {
namespace detail {

/// 启动 IoU 矩阵计算 kernel
/// @param d_boxes    [n][4] 的设备指针（x1,y1,x2,y2），已按分数降序
/// @param d_cls      [n] 的类别
/// @param d_overlap  [n*n] 的输出（上三角为 1 表示 i 抑制 j）
void launchOverlap(const float* d_boxes, const int* d_cls,
                   unsigned char* d_overlap, int n, float iou_thres,
                   cudaStream_t stream = nullptr);

}  // namespace detail
}  // namespace yolo
