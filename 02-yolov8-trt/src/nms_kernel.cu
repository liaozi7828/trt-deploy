// ---------------------------------------------------------------------------
// nms_kernel.cu —— GPU NMS 的 device 代码（纯 CUDA，不含任何 STL）
//
// ⚠️ 由 nvcc 编译，带 -U_GNU_SOURCE。所以这里**只能 include**：
//      <cuda_runtime.h> / <math.h>（C 风格）
//    绝对不能 include <string> / <vector> / <algorithm> / <cstring>。
//
//    面向 std::vector 的封装在 nms_gpu.cpp（g++ 编译）。
// ---------------------------------------------------------------------------
#include "nms_kernel.h"

#include <math.h>

#include "cuda_utils.h"

namespace yolo {
namespace detail {

namespace {

/// 两个框是否重叠超过阈值（只用 4 个 float，不碰任何容器）
__device__ __forceinline__ bool overlapDev(const float* a, const float* b,
                                           float thres) {
    const float ix1 = fmaxf(a[0], b[0]);
    const float iy1 = fmaxf(a[1], b[1]);
    const float ix2 = fminf(a[2], b[2]);
    const float iy2 = fminf(a[3], b[3]);

    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return false;

    const float inter = iw * ih;
    const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0f) return false;

    return (inter / uni) > thres;
}

/// 每个线程算矩阵的一个元素 (i, j)
/// 只填上三角（i < j），因为 NMS 只需要 "i 抑制 j" 这一个方向
__global__ void overlapKernel(const float* __restrict__ boxes,
                              const int* __restrict__ cls,
                              unsigned char* __restrict__ overlap,
                              int n, float thres) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n;
    if (idx >= total) return;

    const int i = idx / n;
    const int j = idx % n;

    if (i >= j || cls[i] != cls[j]) {   // 下三角 + 跨类：不抑制
        overlap[idx] = 0;
        return;
    }
    overlap[idx] = overlapDev(boxes + i * 4, boxes + j * 4, thres) ? 1 : 0;
}

}  // namespace

void launchOverlap(const float* d_boxes, const int* d_cls,
                   unsigned char* d_overlap, int n, float iou_thres,
                   cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int total = n * n;
    const int grid = (total + block - 1) / block;
    overlapKernel<<<grid, block, 0, stream>>>(d_boxes, d_cls, d_overlap, n, iou_thres);
    CUDA_CHECK_KERNEL();
}

}  // namespace detail
}  // namespace yolo
