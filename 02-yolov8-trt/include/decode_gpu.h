#pragma once
// ---------------------------------------------------------------------------
// decode_gpu.h —— GPU decode 的主机侧接口（std::vector 进出，g++ 编译）
//
// 目的：把 decode 从 CPU 搬到 GPU，**同时省掉 2.69 MB 的 D2H**。
//
//   改前（CPU decode）：
//     GPU 推理 → D2H 2.69 MB（0.417 ms）→ CPU decode（0.335 ms）
//     合计 0.752 ms
//
//   改后（GPU decode）：
//     GPU 推理 → GPU decode kernel（~0.01 ms）→ D2H 1.1 KB（~0.005 ms）
//     合计 ~0.02 ms
//
// 返回值和 CPU 版 decodeV8 语义完全一致（letterbox 空间、xyxy、score、cls），
// 所以后续的 undoLetterbox / NMS 不用改。
// ---------------------------------------------------------------------------
#include <vector>

#include "postprocess.h"

namespace yolo {

/// GPU 版 decode（含 kernel 启动 + 结果回传）
///
/// @param d_out      **设备指针**，引擎输出 [4+nc][num_anchors]，通道在前
/// @param conf_thres 置信度阈值
///
/// 返回按分数降序排列的检出（和 CPU 版同一排序规则，便于逐项对比）。
std::vector<Detection> decodeV8Gpu(const void* d_out, int num_classes,
                                   int num_anchors, float conf_thres);

/// 只跑 kernel、不回传（用于测纯 kernel 耗时，把 D2H 排除掉）
void decodeV8GpuKernelOnly(const void* d_out, int num_classes,
                           int num_anchors, float conf_thres);

/// 只提交 kernel、不回传（CUDA Graph 捕获用；和上面等价，名字表达用途）
void decodeV8GpuEnqueueOnly(const void* d_out, int num_classes,
                            int num_anchors, float conf_thres);

}  // namespace yolo
