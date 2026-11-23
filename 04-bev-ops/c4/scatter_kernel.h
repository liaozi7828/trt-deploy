#pragma once
// ---------------------------------------------------------------------------
// scatter_kernel.h —— 点云 → pillar 的 scatter（纯 CUDA 接口，不含 STL）
//
// 这份 kernel 来自项目②（02-yolov8-trt），那边它已经通过了 5 层验证：
//   · 手算 3 点（人脑按公式算）
//   · 定义级不变量（不需要参考实现：格子唯一、每个界内点都能找到、数量守恒）
//   · NumPy 原语参考（np.unique + np.lexsort，公式化完全不同）
//   · CPU 参考实现
//   · 对比器自检（给对比函数本身做反例测试）
// 结果：5 万点 → 28403 根 pillar，**0 点丢失**，`max|diff| = 0`
//
// 搬到这里做 plugin 的理由（三条同时成立，缺一不可）：
//   ① TRT 没有等价层（不像 grid_sample 有 IGridSampleLayer）
//   ② TRT 也不能用插件库里的东西实现（不像 ScatterElements 有内置 plugin）
//   ③ **输出规模由数据决定**：有几根 pillar 只有算完才知道
//      —— TRT 的 plugin 同样无法表达完全动态的输出形状，
//         所以必须用「固定容量 + count 标量」这个标准解法（EfficientNMS 也这么做）
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

namespace bev {

/// pillar 网格参数
struct PillarParams {
    int grid_x{0};
    int grid_y{0};
    float min_x{0.0f};
    float min_y{0.0f};
    float inv_voxel_x{0.0f};
    float inv_voxel_y{0.0f};
    int max_pillars{0};
    int max_pts_per_pillar{0};
    int num_features{0};
    int point_stride{0};
};

/// 启动 scatter kernel
///
/// @param d_points       设备，[n_points][point_stride]
/// @param d_cell_to_pid  设备，[grid_x*grid_y]，**调用前必须全部填 -1**
///                       （plugin 里用 workspace 承载，enqueue 时 memset 成 0xFF）
/// @param d_pillars      设备，[max_pillars][max_pts][num_features]
/// @param d_coords       设备，[max_pillars][4] = (batch, z, y, x)
/// @param d_counts       设备，[max_pillars]
/// @param d_num_pillars  设备，1 个 int，**调用前必须清零**
/// @return 无（错误由 TRT 层从 stream 上捕）
void launchScatter(const float* d_points, int n_points,
                   int* d_cell_to_pid,
                   float* d_pillars, int* d_coords, int* d_counts,
                   int* d_num_pillars,
                   const PillarParams& p,
                   cudaStream_t stream = nullptr);

}  // namespace bev
