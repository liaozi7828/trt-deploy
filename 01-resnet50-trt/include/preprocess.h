#pragma once
// ---------------------------------------------------------------------------
// preprocess.h —— ResNet50 的输入前处理（CUDA 版）
//
// 标准流程：
//   1. 双线性缩放 原图 (H,W,3) → (224,224,3)
//   2. 归一化 (x - mean) / std
//   3. HWC → CHW（PyTorch 要 [3,224,224]，图像天然是 [224,224,3]）
//
// 本文件提供 **多条实现路径**，用于 A/B 对比（每次只改一个变量）：
//   resizeBilinear      —— 朴素版：一个线程算一个输出像素
//   hwcToChwNormalize   —— 朴素版：一个线程处理一个 float
//   hwcToChwNormalizeVec4 —— 向量化版：一个线程处理 4 个 float（float4 存储）
//   letterbox           —— 等比缩放 + 居中填充（YOLO 需要，不拉伸变形）
//
// 所有 CUDA 版都必须和对应的 CPU 版逐元素一致（< 1e-6）。
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

namespace preprocess {

// ===========================================================================
// CUDA 实现
// ===========================================================================

/// 双线性缩放：uint8 HWC → float HWC（拉伸到 dw×dh，不保持宽高比）
void resizeBilinear(const unsigned char* src, int sw, int sh,
                    float* dst, int dw, int dh,
                    cudaStream_t stream = nullptr);

/// HWC float → CHW float + normalize（朴素版，一个线程一个元素）
/// @param mean/std_dev  必须是**设备指针**（长度 = c），主机指针会触发 illegal memory access
void hwcToChwNormalize(const float* src, float* dst,
                       int h, int w, int c,
                       const float* mean, const float* std_dev,
                       cudaStream_t stream = nullptr);

/// HWC float → CHW float + normalize（向量化版，一个线程处理 4 个输出元素）
///
/// 优化点：输出侧用 float4 一次写 16 字节（相比 4 次 4 字节写），
///        减少写事务数量，提升有效写带宽。
/// 要求：dst 必须 16 字节对齐，且每个 (channel, row) 段起始偏移是 4 的倍数。
void hwcToChwNormalizeVec4(const float* src, float* dst,
                           int h, int w, int c,
                           const float* mean, const float* std_dev,
                           cudaStream_t stream = nullptr);

/// letterbox 的几何参数
///
/// 前处理用它做"缩放+填充"，后处理必须用它做**逆运算**（把框坐标还原到原图）。
/// 两边共用同一个函数算出来的参数，才能保证严格互逆。
struct LetterboxParams {
    float scale{1.0f};   // 缩放比
    int new_w{0};        // 缩放后的宽
    int new_h{0};        // 缩放后的高
    int pad_x{0};        // 左侧填充像素数
    int pad_y{0};        // 顶部填充像素数
};

/// 计算 letterbox 参数（前处理与后处理共用，保证互逆）
LetterboxParams computeLetterbox(int src_w, int src_h, int dst_w, int dst_h);

/// letterbox：等比缩放 + 居中填充到 dw×dh，输出 float HWC
///
/// 和 resizeBilinear 的区别：不拉伸。短边缩放后居中，四周填充 pad 值。
/// 这是 YOLO / 大部分检测模型的标准前处理。
/// @param pad 填充值（YOLO 惯例是 114）
void letterbox(const unsigned char* src, int sw, int sh,
               float* dst, int dw, int dh, unsigned char pad,
               cudaStream_t stream = nullptr);

// ===========================================================================
// CPU 参考实现（正确性基准，由 g++ 编译）
// ===========================================================================

void resizeBilinearCpu(const unsigned char* src, int sw, int sh,
                       float* dst, int dw, int dh);

void hwcToChwNormalizeCpu(const float* src, float* dst,
                          int h, int w, int c,
                          const float* mean, const float* std_dev);

/// CPU 版 letterbox（同样输出 float HWC）
void letterboxCpu(const unsigned char* src, int sw, int sh,
                  float* dst, int dw, int dh, unsigned char pad);

}  // namespace preprocess
