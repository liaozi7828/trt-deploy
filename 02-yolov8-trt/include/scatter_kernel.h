#pragma once
// ---------------------------------------------------------------------------
// scatter_kernel.h —— 点云 → pillar 的 scatter（纯 CUDA 接口，不含 STL）
//
// 为什么这个操作必须写 kernel / plugin：
//   PointPillars / CenterPoint 的第一步是把无序点云"散"进规则的柱状网格。
//   这个过程有两个 TensorRT 做不到的特征：
//     ① **原子分配**：两个点落进同一个格子要合并进同一根 pillar，
//        不同格子要各自拿到新编号 —— 输出数量取决于数据（动态）。
//     ② **输出形状不固定**：有几根 pillar 只有算完才知道。
//   ONNX 里没有对应的算子（ScatterND 语义不同），所以只能写 plugin。
//
// 本文件只声明"启动 kernel"的接口，主机侧封装（显存、CPU 参考、对比测试）
// 在 pillarize.cpp。
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

namespace bev {

/// pillar 网格参数
struct PillarParams {
    int grid_x{0};              // x 方向格子数
    int grid_y{0};              // y 方向格子数
    float min_x{0.0f};          // 网格左下角的世界坐标
    float min_y{0.0f};
    float inv_voxel_x{0.0f};    // 1 / voxel_size（预先取倒数，省除法）
    float inv_voxel_y{0.0f};
    int max_pillars{0};         // 输出容量上限
    int max_pts_per_pillar{0};  // 每根 pillar 最多收几个点
    int num_features{0};        // 每个点拷几个特征（PointPillars 常用 4：x,y,z,intensity）
    int point_stride{0};        // 输入点数组每行几列（可能 > num_features）
};

/// 启动 scatter kernel
///
/// @param d_points       设备，[n_points][point_stride]
/// @param d_cell_to_pid  设备，[grid_x*grid_y]，**调用前必须全部填 -1**
/// @param d_pillars      设备，[max_pillars][max_pts][num_features]
/// @param d_coords       设备，[max_pillars][4] = (batch, z, y, x)  ← spconv 约定
/// @param d_counts       设备，[max_pillars]
/// @param d_num_pillars  设备，1 个 int，**调用前必须清零**
void launchScatter(const float* d_points, int n_points,
                   int* d_cell_to_pid,
                   float* d_pillars, int* d_coords, int* d_counts,
                   int* d_num_pillars,
                   const PillarParams& p,
                   cudaStream_t stream = nullptr);

}  // namespace bev
