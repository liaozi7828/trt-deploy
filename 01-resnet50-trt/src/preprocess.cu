// ---------------------------------------------------------------------------
// preprocess.cu —— 前处理的 **device 代码**（kernel + 启动函数）
//
// ⚠️ 由 nvcc 编译，命令行带 -U_GNU_SOURCE（绕开 glibc 2.41 的 cospi 冲突）。
//    因此这里**不能 include <string> / <vector> / <algorithm>**。
//    CPU 参考实现在 preprocess_cpu.cpp（g++ 编译）。
//
// 本文件包含 4 个 kernel：
//   朴素 resize            —— 一个线程算一个输出像素
//   朴素 normalize         —— 一个线程处理一个 float
//   向量化 normalize       —— 一个线程处理 4 个 float（float4 存储）
//   letterbox              —— 等比缩放 + 居中填充（不拉伸，保持宽高比）
// ---------------------------------------------------------------------------
#include "preprocess.h"

#include <math.h>

#include "cuda_utils.h"

namespace preprocess {

// ===========================================================================
// 归一化的**唯一定义处**
//
// ⚠️ 关键：输入是 uint8 的 [0, 255]，必须先除以 255 缩到 [0, 1]，
//    才能用 ImageNet 的 mean/std（0.485 那些是给 [0,1] 数值准备的）。
//
//    漏掉 /255 的后果：输入被放大 255 倍 → 激活值上千 → 输出 logits 几百
//    （正常应该 ±10），而且 PyTorch 和 TRT 的浮点误差被同步放大，
//    精度对齐会差到 1e-1 量级。
//
//    这个 bug 当初就是被 compare_precision.py 抓出来的：
//    导出的输入范围是 [-1.08, 1128.7] 而不是 [-2.5, 2.5]。
// ===========================================================================
__device__ __forceinline__ float normalizeOne(float v, float m, float s) {
    constexpr float kInv255 = 1.0f / 255.0f;
    return (v * kInv255 - m) / s;
}

// ===========================================================================
// 1. 双线性缩放（朴素版）
//    一个线程算一个输出像素；源图相邻像素会被不同线程重复读取
// ===========================================================================
__global__ void resizeBilinearKernel(const unsigned char* __restrict__ src,
                                     int sw, int sh,
                                     float* __restrict__ dst,
                                     int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // 目标列
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // 目标行
    if (x >= dw || y >= dh) return;

    // half-pixel center 对齐（和 OpenCV / torchvision 一致）
    float fx = (x + 0.5f) * sw / dw - 0.5f;
    float fy = (y + 0.5f) * sh / dh - 0.5f;

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float wx = fx - x0;
    float wy = fy - y0;

    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x0 > sw - 1) x0 = sw - 1;
    if (y0 > sh - 1) y0 = sh - 1;
    if (x1 > sw - 1) x1 = sw - 1;
    if (y1 > sh - 1) y1 = sh - 1;

    for (int c = 0; c < 3; ++c) {
        float v00 = src[(y0 * sw + x0) * 3 + c];
        float v10 = src[(y0 * sw + x1) * 3 + c];
        float v01 = src[(y1 * sw + x0) * 3 + c];
        float v11 = src[(y1 * sw + x1) * 3 + c];

        float top = v00 * (1.0f - wx) + v10 * wx;
        float bot = v01 * (1.0f - wx) + v11 * wx;
        dst[(y * dw + x) * 3 + c] = top * (1.0f - wy) + bot * wy;
    }
}

// ===========================================================================
// 2. HWC → CHW + normalize（朴素版：一个线程一个元素）
// ===========================================================================
__global__ void hwcToChwNormalizeKernel(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        int h, int w, int c,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ std_dev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = h * w * c;
    if (idx >= total) return;

    int ch = idx / (h * w);        // 目标布局 [c][h][w]
    int rem = idx % (h * w);
    int y = rem / w;
    int x = rem % w;

    float v = src[(y * w + x) * c + ch];
    dst[idx] = normalizeOne(v, mean[ch], std_dev[ch]);
}

// ===========================================================================
// 3. HWC → CHW + normalize（向量化版：一个线程处理 4 个连续输出元素）
//
//    为什么能快：
//      - 朴素版每个线程只写 4 字节（float），4 个线程 = 4 次写事务
//      - 向量化版一个线程写 16 字节（float4），1 次写事务搞定
//      - 写带宽利用率提升
//
//    为什么"读"仍然分散：
//      - 输出连续（CHW）意味着同一通道里 4 个相邻 w
//      - 对应源图地址相差 c（=3）个 float，天生无法合并
//      - 所以这项优化只改善**写**，不改善**读**
// ===========================================================================
__global__ void hwcToChwNormalizeVec4Kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            int h, int w, int c,
                                            const float* __restrict__ mean,
                                            const float* __restrict__ std_dev) {
    const int total = h * w * c;
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // ---- 尾部不足 4 个：标量兜底 ----
    if (base + 3 >= total) {
        for (int i = base; i < total; ++i) {
            int ch = i / (h * w);
            int rem = i % (h * w);
            int y = rem / w;
            int x = rem % w;
            float v = src[(y * w + x) * c + ch];
            dst[i] = normalizeOne(v, mean[ch], std_dev[ch]);
        }
        return;
    }

    const int ch = base / (h * w);
    const int rem = base % (h * w);
    const int y = rem / w;
    const int x = rem % w;

    const float m = mean[ch];
    const float s = std_dev[ch];

    // ---- 4 个元素跨行：标量兜底（w 不是 4 的倍数时会发生）----
    //   注意这里必须用**全局下标**重算 (ch,y,x)，不能拿上面的 y 硬加 k，
    //   否则 x+k >= w 时会读到下一行，属于越界逻辑错误。
    if (x + 3 >= w) {
        for (int k = 0; k < 4; ++k) {
            const int i = base + k;
            const int c2 = i / (h * w);
            const int r2 = i % (h * w);
            const int y2 = r2 / w;
            const int x2 = r2 % w;
            const float v = src[(y2 * w + x2) * c + c2];
            dst[i] = normalizeOne(v, mean[c2], std_dev[c2]);
        }
        return;
    }

    const float* p = src + (y * w + x) * c + ch;
    float4 out;
    out.x = normalizeOne(p[0 * c], m, s);
    out.y = normalizeOne(p[1 * c], m, s);
    out.z = normalizeOne(p[2 * c], m, s);
    out.w = normalizeOne(p[3 * c], m, s);
    *reinterpret_cast<float4*>(dst + base) = out;
}

// ===========================================================================
// 4. letterbox：等比缩放 + 居中填充
//
//    new_w / new_h / pad_x / pad_y 都由 host 侧算好传进来，
//    kernel 只负责"这个输出像素该填什么"。
// ===========================================================================
__global__ void letterboxKernel(const unsigned char* __restrict__ src,
                                int sw, int sh,
                                float* __restrict__ dst,
                                int dw, int dh,
                                int new_w, int new_h,
                                int pad_x, int pad_y,
                                float pad_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;

    const int idx = (y * dw + x) * 3;

    // 映射到"缩放后图像"的坐标
    const int sx = x - pad_x;
    const int sy = y - pad_y;

    // 落在填充区
    if (sx < 0 || sy < 0 || sx >= new_w || sy >= new_h) {
        dst[idx + 0] = pad_val;
        dst[idx + 1] = pad_val;
        dst[idx + 2] = pad_val;
        return;
    }

    // 从缩放后坐标映射回源图（half-pixel）
    const float fx = (sx + 0.5f) * sw / new_w - 0.5f;
    const float fy = (sy + 0.5f) * sh / new_h - 0.5f;

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    const float wx = fx - x0;
    const float wy = fy - y0;

    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x0 > sw - 1) x0 = sw - 1;
    if (y0 > sh - 1) y0 = sh - 1;
    if (x1 > sw - 1) x1 = sw - 1;
    if (y1 > sh - 1) y1 = sh - 1;

    for (int c = 0; c < 3; ++c) {
        float v00 = src[(y0 * sw + x0) * 3 + c];
        float v10 = src[(y0 * sw + x1) * 3 + c];
        float v01 = src[(y1 * sw + x0) * 3 + c];
        float v11 = src[(y1 * sw + x1) * 3 + c];
        float top = v00 * (1.0f - wx) + v10 * wx;
        float bot = v01 * (1.0f - wx) + v11 * wx;
        dst[idx + c] = top * (1.0f - wy) + bot * wy;
    }
}

// ===========================================================================
// 启动函数
// ===========================================================================

void resizeBilinear(const unsigned char* src, int sw, int sh,
                    float* dst, int dw, int dh, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
    resizeBilinearKernel<<<grid, block, 0, stream>>>(src, sw, sh, dst, dw, dh);
    CUDA_CHECK_KERNEL();
}

void hwcToChwNormalize(const float* src, float* dst,
                       int h, int w, int c,
                       const float* mean, const float* std_dev,
                       cudaStream_t stream) {
    int total = h * w * c;
    int block = 256;
    int grid = (total + block - 1) / block;
    hwcToChwNormalizeKernel<<<grid, block, 0, stream>>>(src, dst, h, w, c, mean, std_dev);
    CUDA_CHECK_KERNEL();
}

void hwcToChwNormalizeVec4(const float* src, float* dst,
                           int h, int w, int c,
                           const float* mean, const float* std_dev,
                           cudaStream_t stream) {
    const int total = h * w * c;
    const int nthreads = (total + 3) / 4;          // 每个线程处理 4 个元素
    const int block = 256;
    const int grid = (nthreads + block - 1) / block;
    hwcToChwNormalizeVec4Kernel<<<grid, block, 0, stream>>>(
        src, dst, h, w, c, mean, std_dev);
    CUDA_CHECK_KERNEL();
}

void letterbox(const unsigned char* src, int sw, int sh,
               float* dst, int dw, int dh, unsigned char pad,
               cudaStream_t stream) {
    // 几何关系与后处理共用同一个函数，保证严格互逆
    const LetterboxParams p = computeLetterbox(sw, sh, dw, dh);

    dim3 block(16, 16);
    dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
    letterboxKernel<<<grid, block, 0, stream>>>(
        src, sw, sh, dst, dw, dh, p.new_w, p.new_h, p.pad_x, p.pad_y, (float)pad);
    CUDA_CHECK_KERNEL();
}

// ===========================================================================
// 说明：为什么没做 "shared memory tile 版 resize"
//
// tile 版的前提是"多个输出像素复用同一块源图区域"，但：
//   - 缩放比变化时（640×480 → 224×224 是 2.86 倍缩小），一个 32×32 的输出 tile
//     要覆盖约 92×92 的源区域 → 92*92*3 字节 ≈ 25 KB，接近默认 48 KB 上限
//   - 当前前处理整体只占 1.8% 延迟，resize 只占其中一部分
//   - 投入产出比不划算，复杂度却要翻倍
//
// 结论：**优化要有数据支撑**。等 resize 真成为瓶颈（比如 4K 输入）再做。
// ===========================================================================

}  // namespace preprocess
