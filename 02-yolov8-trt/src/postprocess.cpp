// ---------------------------------------------------------------------------
// postprocess.cpp —— YOLOv8 后处理的 CPU 实现（由 g++ 编译）
//
// P3 W2 要求"先 CPU 后 GPU"。这一版就是 CPU 基准：
//   - 正确性容易验证（可以逐行读）
//   - 之后写 CUDA 版时，用它做对齐基准
// ---------------------------------------------------------------------------
#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace yolo {

// ===========================================================================
// ① decode
// ===========================================================================
std::vector<Detection> decodeV8(const float* out, int num_classes, int num_anchors,
                                float conf_thres) {
    std::vector<Detection> dets;
    dets.reserve(64);
    if (num_classes <= 0 || num_anchors <= 0) return dets;

    // ---- 遍历顺序决定了几十倍的速度差 ----
    //
    // 原实现（anchor 外层 / class 内层）：
    //   for i:  for c:  out[(4+c)*N + i]
    // 相邻两次读取相隔 N*4 = 33.6 KB → 同时在跟 80 条内存流，
    // 超过硬件预取器能跟踪的流数，实测 **0.367 ms**（占端到端 12%）。
    //
    // 现在（class 外层 / anchor 内层）：
    //   for c:  for i:  row[i]        ← 单条顺序流，预取最优，可自动向量化
    //
    // 结果完全一致（同一个 argmax），用 scripts/verify_nms.py 可回归验证。
    static thread_local std::vector<float> best;
    static thread_local std::vector<int> best_c;
    best.resize(static_cast<size_t>(num_anchors));
    best_c.resize(static_cast<size_t>(num_anchors));

    const float* row0 = out + static_cast<size_t>(4) * num_anchors;
    for (int i = 0; i < num_anchors; ++i) {
        best[i] = row0[i];
        best_c[i] = 0;
    }

    for (int c = 1; c < num_classes; ++c) {
        const float* row = out + static_cast<size_t>(4 + c) * num_anchors;
        for (int i = 0; i < num_anchors; ++i) {
            if (row[i] > best[i]) {
                best[i] = row[i];
                best_c[i] = c;
            }
        }
    }

    for (int i = 0; i < num_anchors; ++i) {
        if (best[i] < conf_thres) continue;

        // ★ 通道在前：(c * num_anchors + i)
        const float cx = out[0 * num_anchors + i];
        const float cy = out[1 * num_anchors + i];
        const float bw = out[2 * num_anchors + i];
        const float bh = out[3 * num_anchors + i];

        Detection d;
        d.x1 = cx - bw * 0.5f;
        d.y1 = cy - bh * 0.5f;
        d.x2 = cx + bw * 0.5f;
        d.y2 = cy + bh * 0.5f;
        d.score = best[i];
        d.cls = best_c[i];
        dets.push_back(d);
    }
    return dets;
}

// ===========================================================================
// ② 反 letterbox
// ===========================================================================
void undoLetterbox(std::vector<Detection>& dets,
                   const preprocess::LetterboxParams& lp,
                   int orig_w, int orig_h) {
    const float sx = lp.scale;
    const float sy = lp.scale;

    for (auto& d : dets) {
        // 先减填充，再除缩放
        d.x1 = (d.x1 - lp.pad_x) / sx;
        d.y1 = (d.y1 - lp.pad_y) / sy;
        d.x2 = (d.x2 - lp.pad_x) / sx;
        d.y2 = (d.y2 - lp.pad_y) / sy;

        // clamp 到图像范围（框可能超出画面）
        d.x1 = std::max(0.0f, std::min(d.x1, static_cast<float>(orig_w - 1)));
        d.y1 = std::max(0.0f, std::min(d.y1, static_cast<float>(orig_h - 1)));
        d.x2 = std::max(0.0f, std::min(d.x2, static_cast<float>(orig_w - 1)));
        d.y2 = std::max(0.0f, std::min(d.y2, static_cast<float>(orig_h - 1)));
    }
}

// ===========================================================================
// ③ NMS
// ===========================================================================
namespace {

inline float iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;

    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

}  // namespace

std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thres) {
    // 按分数降序
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<char> removed(dets.size(), 0);
    std::vector<Detection> keep;

    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[j].cls != dets[i].cls) continue;   // 逐类抑制
            if (iou(dets[i], dets[j]) > iou_thres) removed[j] = 1;
        }
    }
    return keep;
}

// ===========================================================================
// ④ 画框
// ===========================================================================
void drawBoxes(unsigned char* img, int w, int h,
               const std::vector<Detection>& dets,
               const char* const* names, int num_names) {
    auto put = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        unsigned char* p = img + (static_cast<size_t>(y) * w + x) * 3;
        p[0] = r;
        p[1] = g;
        p[2] = b;
    };

    for (const auto& d : dets) {
        // 每个类别一个颜色（简单哈希）
        const unsigned char r = static_cast<unsigned char>((d.cls * 67) % 200 + 55);
        const unsigned char g = static_cast<unsigned char>((d.cls * 131) % 200 + 55);
        const unsigned char b = static_cast<unsigned char>((d.cls * 197) % 200 + 55);

        const int x1 = static_cast<int>(d.x1);
        const int y1 = static_cast<int>(d.y1);
        const int x2 = static_cast<int>(d.x2);
        const int y2 = static_cast<int>(d.y2);

        // 2 像素粗的矩形
        for (int t = 0; t < 2; ++t) {
            for (int x = x1; x <= x2; ++x) {
                put(x, y1 + t, r, g, b);
                put(x, y2 - t, r, g, b);
            }
            for (int y = y1; y <= y2; ++y) {
                put(x1 + t, y, r, g, b);
                put(x2 - t, y, r, g, b);
            }
        }
        // 左上角画一个实心小方块，表示"这里有目标"
        const int tag = 6;
        for (int y = y1; y < y1 + tag && y < h; ++y) {
            for (int x = x1; x < x1 + tag && x < w; ++x) put(x, y, r, g, b);
        }
        if (names && d.cls >= 0 && d.cls < num_names) {
            (void)names[d.cls];   // 没字体就不画文字，名字留给 stdout
        }
    }
}

// ===========================================================================
// PPM(P6) 读写
// ===========================================================================
namespace {

/// 跳过 PPM 头里的空白和 # 注释
bool skipSpaceAndComments(std::ifstream& f) {
    while (true) {
        int c = f.peek();
        if (c == '#') {
            std::string line;
            std::getline(f, line);
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            f.get();
        } else {
            return f.good();
        }
    }
}

}  // namespace

bool readPpm(const std::string& path, std::vector<unsigned char>& out, int& w, int& h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[3] = {0, 0, 0};
    f.read(magic, 2);
    if (magic[0] != 'P' || magic[1] != '6') {
        std::fprintf(stderr, "只支持 P6 的 PPM，收到 %s\n", magic);
        return false;
    }

    if (!skipSpaceAndComments(f)) return false;
    f >> w;
    if (!skipSpaceAndComments(f)) return false;
    f >> h;
    if (!skipSpaceAndComments(f)) return false;
    int maxval = 0;
    f >> maxval;
    f.get();   // 吃掉 header 后那个换行

    if (w <= 0 || h <= 0 || maxval != 255) {
        std::fprintf(stderr, "PPM 头不对: %dx%d maxval=%d\n", w, h, maxval);
        return false;
    }

    out.resize(static_cast<size_t>(w) * h * 3);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    return f.gcount() == static_cast<std::streamsize>(out.size());
}

bool writePpm(const std::string& path, const unsigned char* img, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(img), static_cast<std::streamsize>(w) * h * 3);
    return f.good();
}

}  // namespace yolo
