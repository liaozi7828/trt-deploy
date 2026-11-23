#pragma once
// ---------------------------------------------------------------------------
// postprocess.h —— YOLOv8 的后处理（P3 W1-W2 的核心）
//
// 后处理链：
//   ① decode      : [1, 84, 8400] 的原始输出 → 候选框列表（letterbox 空间）
//   ② undoLetterbox: 把框坐标从 640×640 缩放+填充空间还原到原图
//   ③ NMS         : 去掉同一个目标上的重复框
//   ④ draw        : 画到图上（验证用）
//
// 为什么不能省第 ②：
//   前处理做了"等比缩放 + 填充"，模型看到的坐标系和原图不一样。
//   不还原的话框会整体偏移+缩小，而且**不会报任何错**。
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <vector>

#include "preprocess.h"   // LetterboxParams

namespace yolo {

struct Detection {
    float x1, y1, x2, y2;   // 原图像素坐标
    float score;
    int cls;
};

/// YOLOv8 输出解码
///
/// 输出布局 [1, 4+nc, num_anchors]（**通道在前**，这是最容易搞反的地方）
///   index(c, i) = c * num_anchors + i
///   c = 0..3   : cx, cy, w, h（已经是 640×640 输入空间的像素值）
///   c = 4..end : 每个类别的分数（导出时已经过 sigmoid）
///
/// @param conf_thres 置信度阈值（YOLO 惯例 0.25）
std::vector<Detection> decodeV8(const float* out, int num_classes, int num_anchors,
                                float conf_thres);

/// 把 letterbox 空间的框还原到原图（前处理 letterbox 的逆运算）
/// 注意 LetterboxParams 在 namespace preprocess 里，必须带限定
void undoLetterbox(std::vector<Detection>& dets,
                   const preprocess::LetterboxParams& lp,
                   int orig_w, int orig_h);

/// 逐类 NMS（只抑制同类框）
/// @param iou_thres 通常 0.45
std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thres);

/// 画框（RGB uint8，HWC 布局，原地修改）
void drawBoxes(unsigned char* img, int w, int h,
               const std::vector<Detection>& dets,
               const char* const* names, int num_names);

// ---- PPM(P6) 读写：无第三方依赖，方便把结果存成图片 ----
bool readPpm(const std::string& path, std::vector<unsigned char>& out, int& w, int& h);
bool writePpm(const std::string& path, const unsigned char* img, int w, int h);

}  // namespace yolo
