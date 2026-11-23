// ---------------------------------------------------------------------------
// decode_nms_kernel.cu —— decode + NMS 融合（纯 CUDA，不含 STL）
//
// 两个 kernel：
//   ① gatherCandidates：一个线程一个 anchor，argmax + 阈值 → 紧凑候选表
//   ② nmsKernel：单块并行贪心 NMS，最多输出 kMaxDetections 个框
//
// 为什么 NMS 用一个块：
//   贪心过程"挑最高分 → 抑制重叠 → 再挑"是**串行依赖**的，
//   跨块做需要多轮 kernel 启动或复杂的全局同步。
//   而候选只有几十到几百个，**一个块 256 线程扫 1024 个候选只要几百次比较**，
//   单块完全够用，而且省掉了块间同步。
// ---------------------------------------------------------------------------
#include "decode_nms_kernel.h"

#include <math.h>

namespace yolo {
namespace detail {

namespace {

constexpr int kThreadsGather = 256;

/// 两个 xyxy 框的 IoU（和 CPU 版 nms() 完全同一套语义）
__device__ __forceinline__ float iouDev(const float* a, const float* b) {
    const float ix1 = fmaxf(a[0], b[0]);
    const float iy1 = fmaxf(a[1], b[1]);
    const float ix2 = fminf(a[2], b[2]);
    const float iy2 = fminf(a[3], b[3]);
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    const float inter = iw * ih;
    const float ua = (a[2] - a[0]) * (a[3] - a[1]);
    const float ub = (b[2] - b[0]) * (b[3] - b[1]);
    const float uni = ua + ub - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

// ---------------------------------------------------------------------------
// ① 候选收集
// ---------------------------------------------------------------------------
__global__ void gatherKernel(const float* __restrict__ out,
                             int num_classes, int num_anchors,
                             float conf_thres,
                             float* __restrict__ cand,
                             int* __restrict__ count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_anchors) return;

    // 通道在前：out[(4+c)*num_anchors + i]
    const float* p = out + static_cast<size_t>(4) * num_anchors + i;
    float best = p[0];
    int best_c = 0;
    for (int c = 1; c < num_classes; ++c) {
        const float s = p[static_cast<size_t>(c) * num_anchors];
        if (s > best) { best = s; best_c = c; }
    }
    if (best < conf_thres) return;

    const int slot = atomicAdd(count, 1);
    if (slot >= kMaxCandidates) return;   // 超出容量（conf 极低时才会发生）

    const float cx = out[0 * num_anchors + i];
    const float cy = out[1 * num_anchors + i];
    const float bw = out[2 * num_anchors + i];
    const float bh = out[3 * num_anchors + i];

    float* d = cand + static_cast<size_t>(slot) * 6;
    d[0] = cx - bw * 0.5f;
    d[1] = cy - bh * 0.5f;
    d[2] = cx + bw * 0.5f;
    d[3] = cy + bh * 0.5f;
    d[4] = best;
    d[5] = static_cast<float>(best_c);
}

// ---------------------------------------------------------------------------
// ② 贪心 NMS（单块并行）
// ---------------------------------------------------------------------------
__global__ void nmsKernel(float* __restrict__ cand, int n_cand, float iou_thres,
                          float* __restrict__ dets, int* __restrict__ num_det) {
    __shared__ char removed[kMaxCandidates];
    __shared__ float s_score[kNmsBlock];
    __shared__ int s_slot[kNmsBlock];

    const int tid = threadIdx.x;

    // 初始化：超出 n_cand 的槽位当作"已移除"
    for (int i = tid; i < kMaxCandidates; i += blockDim.x) {
        removed[i] = (i < n_cand) ? 0 : 1;
    }
    if (tid == 0) *num_det = 0;
    __syncthreads();

    for (int k = 0; k < kMaxDetections; ++k) {
        // ---- ① 归约出当前最高分的未抑制候选 ----
        float my_score = -1.0f;
        int my_slot = -1;
        for (int i = tid; i < n_cand; i += blockDim.x) {
            if (removed[i]) continue;
            const float s = cand[static_cast<size_t>(i) * 6 + 4];
            if (s > my_score) { my_score = s; my_slot = i; }
        }
        s_score[tid] = my_score;
        s_slot[tid] = my_slot;
        __syncthreads();

        for (int step = kNmsBlock / 2; step > 0; step >>= 1) {
            if (tid < step && s_score[tid + step] > s_score[tid]) {
                s_score[tid] = s_score[tid + step];
                s_slot[tid] = s_slot[tid + step];
            }
            __syncthreads();
        }
        const int pick = s_slot[0];
        if (pick < 0) break;    // 没有候选了

        // ---- ② 写出（线程 0 负责，保证输出顺序 = 分数降序）----
        if (tid == 0) {
            const int idx = *num_det;
            if (idx < kMaxDetections) {
                float* d = dets + static_cast<size_t>(idx) * 6;
                for (int c = 0; c < 6; ++c) d[c] = cand[static_cast<size_t>(pick) * 6 + c];
                *num_det = idx + 1;
            }
        }
        __syncthreads();

        // ---- ③ 抑制：同类且 IoU > 阈值 ----
        const float* pk = cand + static_cast<size_t>(pick) * 6;
        const float pick_cls = pk[5];
        removed[pick] = 1;
        __syncthreads();

        for (int i = tid; i < n_cand; i += blockDim.x) {
            if (removed[i]) continue;
            const float* q = cand + static_cast<size_t>(i) * 6;
            if (q[5] != pick_cls) continue;          // 逐类抑制
            if (iouDev(pk, q) > iou_thres) removed[i] = 1;
        }
        __syncthreads();
    }
}

}  // namespace

// ---------------------------------------------------------------------------
void launchGatherCandidates(const float* d_out, int num_classes, int num_anchors,
                            float conf_thres,
                            float* d_cand, int* d_count,
                            cudaStream_t stream) {
    if (num_anchors <= 0) return;
    const int grid = (num_anchors + kThreadsGather - 1) / kThreadsGather;
    gatherKernel<<<grid, kThreadsGather, 0, stream>>>(
        d_out, num_classes, num_anchors, conf_thres, d_cand, d_count);
}

// ---------------------------------------------------------------------------
void launchNms(const float* d_cand, int n_cand, float iou_thres,
               float* d_dets, int* d_num_det,
               cudaStream_t stream) {
    if (n_cand <= 0) return;
    if (n_cand > kMaxCandidates) n_cand = kMaxCandidates;
    // 需要写 cand 吗？不需要——但 kernel 签名里是非 const（为了通用），这里转换一次
    nmsKernel<<<1, kNmsBlock, 0, stream>>>(
        const_cast<float*>(d_cand), n_cand, iou_thres, d_dets, d_num_det);
}

}  // namespace detail
}  // namespace yolo
