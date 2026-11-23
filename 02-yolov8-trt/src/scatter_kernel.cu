// ---------------------------------------------------------------------------
// scatter_kernel.cu —— 点云 → pillar 的 scatter kernel（纯 CUDA，不含 STL）
//
// 算法（每个线程处理一个点）：
//   ① 用点的 (x,y) 算它落在哪个格子 (iy, ix)
//   ② 查 cell_to_pid[格子]：
//        -1  → 这个格子还没有 pillar，**原子地**申请一个新编号
//        -2  → 别人正在申请，自旋等它填好
//        >=0 → 已经有 pillar，直接用
//   ③ 在该 pillar 里**原子地**申请一个点槽位
//   ④ 把点的特征拷进去
//
// 两个关键点（也就是 TRT 做不到的两件事）：
//   · 原子分配 pillar 编号 → 输出数量由数据决定
//   · 原子分配点槽位     → 多线程往同一根 pillar 里追加
//
// 代价：**点在同一根 pillar 内的顺序不确定**（取决于线程调度）。
//   所以主机侧对比必须"按集合比"，不能"按顺序比"——见 pillarize.cpp。
// ---------------------------------------------------------------------------
#include "scatter_kernel.h"

#include <math.h>

namespace bev {

namespace {

/// 表示"这个格子还没有 pillar"
constexpr int kNoPillar = -1;
/// 表示"某线程正在为这个格子申请 pillar 编号"
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
    if (ix < 0 || ix >= p.grid_x || iy < 0 || iy >= p.grid_y) return;  // 出界丢弃

    const int cell = iy * p.grid_x + ix;

    // ---- ② 拿到这根 pillar 的编号 ----
    int pid = atomicCAS(&cell_to_pid[cell], kNoPillar, kClaiming);
    if (pid == kNoPillar) {
        // 我是第一个到达这个格子的线程 → 申请编号
        pid = atomicAdd(num_pillars, 1);
        if (pid >= p.max_pillars) {          // 容量不够，回滚并丢弃这个点
            atomicExch(&cell_to_pid[cell], kNoPillar);   // ★ 原子写，保证可见
            return;
        }
        // ★★ 顺序至关重要：**先初始化完，再发布 pid**（safe publication）
        //
        // 反例（我第一版就是这么写的，导致 69 个格子的点丢失）：
        //     atomicExch(&cell_to_pid[cell], pid);   // 先公开
        //     counts[pid] = 0;                       // 后清零
        //   别的线程看到 pid 后立刻 atomicAdd(&counts[pid], 1)，
        //   随后被这行 `counts[pid] = 0` 覆盖掉 → 那个点的计数就没了。
        //
        // 正确顺序：初始化 → __threadfence() → 发布。
        //   这和多线程里"发布一个只构造了一半的对象"是同一个错误。
        // ★ coords 布局沿用 PointPillars / spconv 的约定：(batch, z, y, x)
        //   注意顺序是**反着的**——维度从大到小排在前面。
        //   如果按 (x, y, z) 的顺序写成 (ix, iy, 0, 0)，下游 spconv 会把
        //   batch 读成 y、把 z 读成 x → 静默放到错误的格子（本项目第 5 次 layout 坑）。
        coords[pid * 4 + 0] = 0;             // batch：单 batch，恒 0
        coords[pid * 4 + 1] = 0;             // z    ：2D pillar 无 z 细分，恒 0
        coords[pid * 4 + 2] = iy;            // y（行号）
        coords[pid * 4 + 3] = ix;            // x（列号）
        counts[pid] = 0;
        __threadfence();                     // 保证上面的写对其他线程可见
        atomicExch(&cell_to_pid[cell], pid); // 最后才公开
    } else if (pid == kClaiming) {
        // 别人在申请 → 等它填好。
        //
        // ★★ 这里必须是 **volatile 读**，否则会死循环：
        //    编译器看到循环体内没有任何对 cell_to_pid[cell] 的写，
        //    可以合法地把"读"提到循环外，于是 -2 永远不变 → 无限自旋。
        //    （这是 CUDA 自旋等待最经典的坑，注释留着提醒后来的人）
        volatile int* slot = &cell_to_pid[cell];
        while ((pid = *slot) == kClaiming) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
        if (pid < 0) return;
    }

    // ---- ③ 在这根 pillar 里申请一个点槽位 ----
    const int slot = atomicAdd(&counts[pid], 1);
    if (slot >= p.max_pts_per_pillar) return;   // 这根 pillar 满了

    // ---- ④ 拷贝特征 ----
    float* dst = pillars + ((static_cast<size_t>(pid) * p.max_pts_per_pillar + slot)
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
