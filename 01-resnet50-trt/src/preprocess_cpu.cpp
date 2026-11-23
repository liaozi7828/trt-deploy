// ---------------------------------------------------------------------------
// preprocess_cpu.cpp —— 前处理的 CPU 参考实现
//
// 为什么单独一个文件：.cu 由 nvcc 编译，带 -U_GNU_SOURCE，
// 无法 include <vector>/<algorithm>；而 CPU 版需要标准库。
// 拆到 .cpp 由 g++ 编译，两边互不影响。
//
// 用途：验证 CUDA 版是否正确。CUDA 输出与这里逐元素对比，误差应 < 1e-6。
// ---------------------------------------------------------------------------
#include "preprocess.h"

#include <algorithm>
#include <cmath>

namespace preprocess {

// ---------------------------------------------------------------------------
// letterbox 几何参数：前处理和后处理共用这一份，保证严格互逆
// ---------------------------------------------------------------------------
LetterboxParams computeLetterbox(int src_w, int src_h, int dst_w, int dst_h) {
    LetterboxParams p;
    p.scale = std::min(static_cast<float>(dst_w) / src_w,
                       static_cast<float>(dst_h) / src_h);
    p.new_w = static_cast<int>(std::lround(src_w * p.scale));
    p.new_h = static_cast<int>(std::lround(src_h * p.scale));
    p.pad_x = (dst_w - p.new_w) / 2;
    p.pad_y = (dst_h - p.new_h) / 2;
    return p;
}

void resizeBilinearCpu(const unsigned char* src, int sw, int sh,
                       float* dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * sw / dw - 0.5f;
            float fy = (y + 0.5f) * sh / dh - 0.5f;

            int x0 = static_cast<int>(std::floor(fx));
            int y0 = static_cast<int>(std::floor(fy));
            float wx = fx - x0, wy = fy - y0;

            int x1 = std::min(std::max(x0 + 1, 0), sw - 1);
            int y1 = std::min(std::max(y0 + 1, 0), sh - 1);
            x0 = std::min(std::max(x0, 0), sw - 1);
            y0 = std::min(std::max(y0, 0), sh - 1);

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
    }
}

void hwcToChwNormalizeCpu(const float* src, float* dst,
                          int h, int w, int c,
                          const float* mean, const float* std_dev) {
    for (int ch = 0; ch < c; ++ch) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // ⚠️ 必须 x/255 之后再减 mean：ImageNet 的 mean/std 是给 [0,1] 用的
                float v = src[(y * w + x) * c + ch] * (1.0f / 255.0f);
                dst[(ch * h + y) * w + x] = (v - mean[ch]) / std_dev[ch];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// letterbox 的 CPU 参考实现
//
// 几何关系和 kernel 里完全一致：
//   scale   = min(dw/sw, dh/sh)        ← 取小，保证整张图放得下
//   new_w/h = round(sw/sh * scale)
//   pad_x/y = (dw/h - new_w/h) / 2     ← 居中
// ---------------------------------------------------------------------------
void letterboxCpu(const unsigned char* src, int sw, int sh,
                  float* dst, int dw, int dh, unsigned char pad) {
    const float scale = std::min(static_cast<float>(dw) / sw,
                                 static_cast<float>(dh) / sh);
    const int new_w = static_cast<int>(std::lround(sw * scale));
    const int new_h = static_cast<int>(std::lround(sh * scale));
    const int pad_x = (dw - new_w) / 2;
    const int pad_y = (dh - new_h) / 2;

    const float pad_val = static_cast<float>(pad);

    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const int idx = (y * dw + x) * 3;
            const int sx = x - pad_x;
            const int sy = y - pad_y;

            if (sx < 0 || sy < 0 || sx >= new_w || sy >= new_h) {
                dst[idx + 0] = pad_val;
                dst[idx + 1] = pad_val;
                dst[idx + 2] = pad_val;
                continue;
            }

            const float fx = (sx + 0.5f) * sw / new_w - 0.5f;
            const float fy = (sy + 0.5f) * sh / new_h - 0.5f;

            int x0 = static_cast<int>(std::floor(fx));
            int y0 = static_cast<int>(std::floor(fy));
            const float wx = fx - x0;
            const float wy = fy - y0;

            int x1 = std::min(std::max(x0 + 1, 0), sw - 1);
            int y1 = std::min(std::max(y0 + 1, 0), sh - 1);
            x0 = std::min(std::max(x0, 0), sw - 1);
            y0 = std::min(std::max(y0, 0), sh - 1);

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
    }
}

}  // namespace preprocess
