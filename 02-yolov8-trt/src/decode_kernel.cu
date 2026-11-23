// ---------------------------------------------------------------------------
// decode_kernel.cu —— decode 的 device 代码（纯 CUDA，不含 STL）
//
// 为什么 decode 该上 GPU：
//   CPU 版实测 0.335 ms，而它只是把 2.82 MB 扫描一遍：
//       2.82 MB ÷ 0.335 ms = 8.4 GB/s   ← 单核内存带宽上限
//   同一份数据在 GPU 上是 384 GB/s，理论上 ~7 µs。
//   而且 GPU 版还能省掉把 2.69 MB 输出搬回主机的 D2H（0.417 ms）。
//
// 一个反直觉的点：
//   CPU 上"跨 33.6 KB 步长访问"是灾难（80 条并发内存流），
//   但在 GPU 上**同一个模式是完美合并的**——因为线程是内层维度，
//   相邻线程读相邻地址，一个 warp 的一次 load 正好覆盖 128 字节。
//   所以这里保持"通道在外、anchor 在内"的线程映射，不做 CPU 那套重排。
// ---------------------------------------------------------------------------
#include "decode_kernel.h"

#include "cuda_utils.h"

namespace yolo {
namespace detail {

namespace {

/// 一个线程负责一个 anchor
__global__ void decodeV8Kernel(const float* __restrict__ out,
                               int num_classes, int num_anchors,
                               float conf_thres,
                               float* __restrict__ dets,
                               int* __restrict__ count,
                               int max_dets) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_anchors) return;

    // argmax over classes（通道在前：out[(4+c)*na + i]）
    const float* p = out + static_cast<size_t>(4) * num_anchors + i;
    float best = p[0];
    int best_c = 0;
    for (int c = 1; c < num_classes; ++c) {
        const float s = p[static_cast<size_t>(c) * num_anchors];
        if (s > best) { best = s; best_c = c; }
    }
    if (best < conf_thres) return;

    const float cx = out[0 * num_anchors + i];
    const float cy = out[1 * num_anchors + i];
    const float bw = out[2 * num_anchors + i];
    const float bh = out[3 * num_anchors + i];

    // 原子追加：顺序不确定，但主机侧反正要按分数排序
    const int slot = atomicAdd(count, 1);
    if (slot >= max_dets) return;    // 防御：容量不够就丢弃（实际给的是 na，不会触发）

    float* d = dets + static_cast<size_t>(slot) * 6;
    d[0] = cx - bw * 0.5f;
    d[1] = cy - bh * 0.5f;
    d[2] = cx + bw * 0.5f;
    d[3] = cy + bh * 0.5f;
    d[4] = best;
    d[5] = static_cast<float>(best_c);
}

}  // namespace

void launchDecode(const float* d_out, int num_classes, int num_anchors,
                  float conf_thres, float* d_dets, int* d_count,
                  int max_dets, cudaStream_t stream) {
    if (num_anchors <= 0) return;
    const int block = 256;
    const int grid = (num_anchors + block - 1) / block;
    decodeV8Kernel<<<grid, block, 0, stream>>>(
        d_out, num_classes, num_anchors, conf_thres, d_dets, d_count, max_dets);
    CUDA_CHECK_KERNEL();
}

}  // namespace detail
}  // namespace yolo
