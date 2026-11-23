// ---------------------------------------------------------------------------
// decode_gpu.cpp —— GPU decode 的主机封装（g++ 编译）
//
// 两个教训都在这一个文件里：
//
//  ① **显存一律复用**：每帧 cudaMalloc 要 10–50 µs，会直接毁掉收益。
//
//  ② **主机/设备边界的往返代价 ≈ 50 µs/次，与传输量几乎无关**：
//     早期版本用两次阻塞 cudaMemcpy（count 4 B + dets 1.1 KB），
//     实测 1.1 KB 花了 0.113 ms —— 比纯 kernel（0.027 ms）还贵 4 倍。
//     现在合并成**一次**拷贝（count 和 dets 放在同一块显存里），
//     并且用 **pinned（page-locked）主机内存**，省掉驱动的内部 staging。
// ---------------------------------------------------------------------------
#include "decode_gpu.h"

#include <algorithm>

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "decode_kernel.h"   // detail::launchDecode（纯 CUDA 接口）

namespace yolo {

namespace {

/// 一次回传最多带多少个 det（多于此数的走罕见路径补拷）
constexpr int kMaxCopyOnce = 1024;

/// 复用缓冲
///
/// 布局（同一块设备显存，一次拷完）：
///   d_buf[0]            : count（用 int 的位模式存放）
///   d_buf[1 + i*6 ...]  : 第 i 个 det 的 x1,y1,x2,y2,score,cls
struct DecodeBuffers {
    float* d_buf{nullptr};
    float* h_pinned{nullptr};   // pinned 主机内存，容量 = 1 + kMaxCopyOnce*6
    size_t cap{0};              // 能装多少个 det
    size_t pinned_n{0};         // h_pinned 有多少个 float

    DecodeBuffers() = default;
    DecodeBuffers(const DecodeBuffers&) = delete;             // 持裸指针，禁止拷贝
    DecodeBuffers& operator=(const DecodeBuffers&) = delete;
    ~DecodeBuffers() {
        if (d_buf) cudaFree(d_buf);
        if (h_pinned) cudaFreeHost(h_pinned);
    }
};

DecodeBuffers& buffers() {
    static DecodeBuffers b;    // 单线程假设；多线程要每线程一份
    return b;
}

void ensureCapacity(size_t n) {
    DecodeBuffers& b = buffers();
    if (b.cap >= n) return;

    // 设备缓冲：1 个 count + n 个 det
    if (b.d_buf) cudaFree(b.d_buf);
    CUDA_CHECK(cudaMalloc(&b.d_buf, (1 + n * 6) * sizeof(float)));

    // pinned 主机缓冲必须能装下**最坏情况**（n 个 det），
    // 否则补拷路径会越界写。n=8400 时 = 201 KB，pinned 内存完全承受得起。
    if (b.h_pinned) cudaFreeHost(b.h_pinned);
    b.pinned_n = 1 + n * 6;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&b.h_pinned),
                             b.pinned_n * sizeof(float), cudaHostAllocDefault));
    b.cap = n;
}

/// 把设备上的紧凑结果转成 vector<Detection>（按分数降序，和 CPU 版一致）
std::vector<Detection> fetch(int num_hit) {
    DecodeBuffers& b = buffers();
    std::vector<Detection> out;
    if (num_hit <= 0) return out;

    out.reserve(static_cast<size_t>(num_hit));
    for (int i = 0; i < num_hit; ++i) {
        const float* d = b.h_pinned + 1 + static_cast<size_t>(i) * 6;
        Detection det;
        det.x1 = d[0];
        det.y1 = d[1];
        det.x2 = d[2];
        det.y2 = d[3];
        det.score = d[4];
        det.cls = static_cast<int>(d[5]);
        out.push_back(det);
    }
    // kernel 用 atomicAdd 追加，顺序不定；这里按分数降序，和 CPU 版对齐
    std::sort(out.begin(), out.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
std::vector<Detection> decodeV8Gpu(const void* d_out, int num_classes,
                                   int num_anchors, float conf_thres) {
    if (!d_out || num_classes <= 0 || num_anchors <= 0) return {};

    ensureCapacity(static_cast<size_t>(num_anchors));
    DecodeBuffers& b = buffers();

    // count 清零（和 dets 同一块显存，offset 0）
    CUDA_CHECK(cudaMemsetAsync(b.d_buf, 0, sizeof(int)));

    detail::launchDecode(static_cast<const float*>(d_out), num_classes, num_anchors,
                         conf_thres,
                         b.d_buf + 1,                                   // dets 起始
                         reinterpret_cast<int*>(b.d_buf),                // count
                         static_cast<int>(b.cap));

    // ★ 一次拷贝就拿到 count + 前若干个 det（不在回传前单独查 count，省一次往返）
    const size_t once_n = std::min<size_t>(kMaxCopyOnce, b.cap);
    const size_t n_copy = 1 + once_n * 6;
    CUDA_CHECK(cudaMemcpy(b.h_pinned, b.d_buf, n_copy * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int num_hit = *reinterpret_cast<const int*>(b.h_pinned);
    if (num_hit < 0) num_hit = 0;
    if (num_hit > static_cast<int>(b.cap)) num_hit = static_cast<int>(b.cap);

    // 罕见路径：命中数超过一次回传的容量，再补一次（conf 很低时才会发生）
    if (static_cast<size_t>(num_hit) > once_n) {
        const size_t rest = (static_cast<size_t>(num_hit) - once_n) * 6;
        CUDA_CHECK(cudaMemcpy(b.h_pinned + n_copy,
                              b.d_buf + n_copy, rest * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    return fetch(num_hit);
}

// ---------------------------------------------------------------------------
void decodeV8GpuKernelOnly(const void* d_out, int num_classes,
                           int num_anchors, float conf_thres) {
    if (!d_out || num_classes <= 0 || num_anchors <= 0) return;
    ensureCapacity(static_cast<size_t>(num_anchors));
    DecodeBuffers& b = buffers();
    CUDA_CHECK(cudaMemsetAsync(b.d_buf, 0, sizeof(int)));
    detail::launchDecode(static_cast<const float*>(d_out), num_classes, num_anchors,
                         conf_thres,
                         b.d_buf + 1,
                         reinterpret_cast<int*>(b.d_buf),
                         static_cast<int>(b.cap));
}

/// 供 CUDA Graph 用：只提交 kernel，不做任何回传
void decodeV8GpuEnqueueOnly(const void* d_out, int num_classes,
                           int num_anchors, float conf_thres) {
    decodeV8GpuKernelOnly(d_out, num_classes, num_anchors, conf_thres);
}

}  // namespace yolo
