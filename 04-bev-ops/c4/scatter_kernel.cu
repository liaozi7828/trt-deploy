// ---------------------------------------------------------------------------
// scatter_kernel.cu —— 点云 → pillar 的 scatter（纯 CUDA，不含 STL）
//
// 算法（每线程一个点）：
//   ① 用点的 (x,y) 算它落在哪个格子 (iy, ix)
//   ② 查 cell_to_pid[格子]：
//        -1  → 这格还没有 pillar，**原子地**申请一个新编号
//        -2  → 别人正在申请，自旋等它填好
//        >=0 → 已经有 pillar，直接用
//   ③ 在该 pillar 里**原子地**申请一个点槽位
//   ④ 把点的特征拷进去
//
// 两个必须记住的并发铁律（项目②里各踩了一次）：
//   ① 自旋等共享标志位必须用 **volatile 读**，否则编译器可以把读提到循环外 → 死循环
//   ② 发布 pillar 编号前必须**先初始化完 + __threadfence()**，
//      否则别的线程看到一个"半成品"，症状是数据静默出错（当时 69 个格子）
// ---------------------------------------------------------------------------
#include "scatter_kernel.h"

#include <math.h>

// ★★ 编译期防护：这个 kernel 用「原子抢占 + 自旋等待」，
//    在 Volta 之前（compute < 70）的锁步编程模型下**会真死锁**——
//    自旋的线程会卡住同一 warp 里持有 claim 的那个线程。
//    把静默死锁变成编译错误，比让它跑到一半卡住强得多。
//    （本机曾因 CMAKE_CUDA_ARCHITECTURES 默认成 52 而踩过这个坑）
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "scatter_kernel 需要 sm_70+（Volta 及以后的独立线程调度）。请检查 CMAKE_CUDA_ARCHITECTURES。"
#endif

namespace bev {

namespace {

/// "这格还没有 pillar"
constexpr int kNoPillar = -1;
/// "某线程正在为这格申请 pillar 编号"
constexpr int kClaiming = -2;

__global__ void scatterKernel(const float* __restrict__ points, int n_points,
                              int* __restrict__ cell_to_pid,
                              float* __restrict__ pillars,
                              int* __restrict__ coords,
                              int* __restrict__ counts,
                              int* __restrict__ num_pillars,
                              PillarParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    const float* pt = points + static_cast<size_t>(i) * p.point_stride;
    const float x = pt[0];
    const float y = pt[1];

    int ix = static_cast<int>(floorf((x - p.min_x) * p.inv_voxel_x));
    int iy = static_cast<int>(floorf((y - p.min_y) * p.inv_voxel_y));
    if (ix < 0 || ix >= p.grid_x || iy < 0 || iy >= p.grid_y) return;   // 出界丢弃

    const int cell = iy * p.grid_x + ix;

    int pid = atomicCAS(&cell_to_pid[cell], kNoPillar, kClaiming);
    if (pid == kNoPillar) {
        pid = atomicAdd(num_pillars, 1);
        if (pid >= p.max_pillars) {                        // 容量不够，回滚
            atomicExch(&cell_to_pid[cell], kNoPillar);
            return;
        }
        // ★ 顺序：先初始化完 → __threadfence() → 最后才发布 pid
        //   反过来的话，别的线程会看到 pid 并对 counts[pid] 自增，
        //   随后被这里的 counts[pid]=0 覆盖 → 那一个点的计数就丢了。
        coords[pid * 4 + 0] = 0;                           // batch：单 batch 恒 0
        coords[pid * 4 + 1] = 0;                           // z    ：2D pillar 恒 0
        coords[pid * 4 + 2] = iy;                          // y
        coords[pid * 4 + 3] = ix;                          // x
        counts[pid] = 0;
        __threadfence();
        atomicExch(&cell_to_pid[cell], pid);
    } else if (pid == kClaiming) {
        // ★ 必须 volatile 读，否则编译器把读提到循环外 → 死循环
        volatile int* slot = &cell_to_pid[cell];
        while ((pid = *slot) == kClaiming) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
        if (pid < 0) return;
    }

    const int s = atomicAdd(&counts[pid], 1);
    if (s >= p.max_pts_per_pillar) return;                 // 这根 pillar 满了

    float* dst = pillars + ((static_cast<size_t>(pid) * p.max_pts_per_pillar + s)
                            * p.num_features);
    for (int c = 0; c < p.num_features; ++c) dst[c] = pt[c];
}

}  // namespace

void launchScatter(const float* d_points, int n_points,
                   int* d_cell_to_pid,
                   float* d_pillars, int* d_coords, int* d_counts,
                   int* d_num_pillars,
                   const PillarParams& p,
                   cudaStream_t stream) {
    if (n_points <= 0) return;
    const int block = 256;
    const int grid = (n_points + block - 1) / block;
    scatterKernel<<<grid, block, 0, stream>>>(
        d_points, n_points, d_cell_to_pid, d_pillars, d_coords, d_counts,
        d_num_pillars, p);
}

}  // namespace bev
