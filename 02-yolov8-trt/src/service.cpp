// ---------------------------------------------------------------------------
// service.cpp —— P5 第一项：把 demo 变成"服务"
//
// 和 main.cpp 的区别：
//   main.cpp    = 实验台：一次推理 + 各种 benchmark 模式
//   service.cpp = 服务：**帧循环**，每帧都记延迟，最后出分位数
//
// 为什么需要它：
//   我们之前的数字全是"中位数"。车端的判据是 **p99 / 最大值**——
//   平均值 0.78 ms 只说明平时够用，p99 12 ms 才说明会不会撞车时晚一米。
//
// 它做什么：
//   ① 加载 engine，捕获 CUDA Graph（一次）
//   ② 进入 N 帧的循环，每帧走完整流水线：
//        preprocess → graph launch → GPU decode → 反 letterbox → NMS
//   ③ 记录**每一帧**的端到端延迟
//   ④ 输出 p50 / p90 / p99 / p999 / max、超预算帧数、直方图
//
// 用法：
//   ./build/bin/yolov8_service --engine models/yolov8n_fp16.engine \
//       --image models/bus.ppm --frames 2000 --cameras 6 --budget-ms 5.5
//
//   # 对照：不走 Graph、decode 留在 CPU
//   ./build/bin/yolov8_service --engine models/yolov8n_fp16.engine \
//       --image models/bus.ppm --frames 2000 --cpu-path
// ---------------------------------------------------------------------------
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_graph.h"
#include "cuda_utils.h"
#include "decode_gpu.h"
#include "postprocess.h"
#include "preprocess.h"
#include "trt_engine.h"

namespace {

constexpr int kSize = 640;   // 模型输入边长

struct Args {
    std::string engine;
    std::string image;
    std::string csv;          // 可选：把每帧延迟写成 CSV
    int frames{2000};
    int cameras{1};           // 模拟几路（路数只影响"按路轮转"的标注）
    int warmup{100};          // 预热帧数（不计入统计）
    float budget_ms{0.0f};    // 单帧预算；0 = 用 33.3/cameras 自动算
    bool cpu_path{false};     // 对照：CPU decode + 直连 enqueue
    bool no_graph{false};     // 关掉 Graph
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("参数缺值: " + k);
            return argv[++i];
        };
        if (k == "--engine")        a.engine = next();
        else if (k == "--image")    a.image = next();
        else if (k == "--csv")      a.csv = next();
        else if (k == "--frames")   a.frames = std::stoi(next());
        else if (k == "--cameras")  a.cameras = std::stoi(next());
        else if (k == "--warmup")   a.warmup = std::stoi(next());
        else if (k == "--budget-ms") a.budget_ms = std::stof(next());
        else if (k == "--cpu-path") a.cpu_path = true;
        else if (k == "--no-graph") a.no_graph = true;
        else if (k == "-h" || k == "--help") {
            std::printf(
                "用法: %s --engine y.engine --image img.ppm [选项]\n"
                "  --frames N      统计多少帧（默认 2000）\n"
                "  --cameras N     模拟几路（默认 1）；用于算单帧预算\n"
                "  --warmup N      预热帧数（默认 100，不计入统计）\n"
                "  --budget-ms X   单帧预算；不填则 = 33.3/N\n"
                "  --cpu-path      对照路径：CPU decode + 直连 enqueue\n"
                "  --no-graph      关掉 CUDA Graph\n"
                "  --csv f         把每帧延迟写进 CSV\n", argv[0]);
            std::exit(0);
        } else {
            std::printf("未知参数: %s\n", k.c_str());
            std::exit(1);
        }
    }
    if (a.engine.empty()) { std::printf("[FAIL] 必须指定 --engine\n"); std::exit(1); }
    if (a.image.empty())  { std::printf("[FAIL] 必须指定 --image\n"); std::exit(1); }
    return a;
}

/// 分位数（线性插值）
float percentile(const std::vector<float>& sorted, float p) {
    if (sorted.empty()) return 0.0f;
    if (sorted.size() == 1) return sorted[0];
    const float pos = (p / 100.0f) * static_cast<float>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const float frac = pos - static_cast<float>(lo);
    return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

/// 直方图（按 0.1 ms 分桶）
void printHistogram(const std::vector<float>& v, float p99) {
    if (v.empty()) return;
    const float lo = *std::min_element(v.begin(), v.end());
    const float hi = *std::max_element(v.begin(), v.end());
    const int nb = 24;
    const float bw = (hi - lo) / static_cast<float>(nb);
    if (bw <= 0.0f) return;

    std::vector<int> bins(static_cast<size_t>(nb), 0);
    for (float x : v) {
        int b = static_cast<int>((x - lo) / bw);
        if (b < 0) b = 0;
        if (b >= nb) b = nb - 1;
        ++bins[static_cast<size_t>(b)];
    }
    const int peak = *std::max_element(bins.begin(), bins.end());

    std::printf("\n  延迟分布（每格 ≈ %d 帧，p99 = %.3f ms 用 | 标出）\n", std::max(1, peak / 40), p99);
    for (int b = 0; b < nb; ++b) {
        const float center = lo + bw * (static_cast<float>(b) + 0.5f);
        const int bar = std::max(1, bins[static_cast<size_t>(b)] * 40 / std::max(peak, 1));
        const bool near_p99 = (center >= p99 - bw * 0.5f) && (center <= p99 + bw * 0.5f);
        std::printf("    %7.3f ms |%-40s| %5d %s\n", center,
                    std::string(static_cast<size_t>(bar), '#').c_str(),
                    bins[static_cast<size_t>(b)], near_p99 ? "<- p99" : "");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        const auto dinfo = cuda_utils::queryDevice();

        std::printf("=========================================================\n");
        std::printf("YOLOv8 推理服务（P5：帧循环 + 延迟分位数）\n");
        std::printf("=========================================================\n");
        std::printf("GPU   : %s (SM %d)\n", dinfo.name, dinfo.sm_count);
        std::printf("引擎  : %s\n", args.engine.c_str());
        std::printf("路径  : %s decode + %s\n",
                    args.cpu_path ? "CPU" : "GPU",
                    (args.no_graph ? "直连 enqueue" : "CUDA Graph"));

        // ---------------- 1. 读图（服务里应换成相机取帧）----------------
        std::vector<unsigned char> host_img;
        int src_w = 0, src_h = 0;
        if (!yolo::readPpm(args.image, host_img, src_w, src_h))
            throw std::runtime_error("读不了 " + args.image + "（只支持 P6 PPM）");
        std::printf("图像  : %s  %dx%d\n", args.image.c_str(), src_w, src_h);

        // ---------------- 2. 引擎 ----------------
        std::vector<std::pair<std::string, std::vector<int64_t>>> opt_shapes{
            {"images", {1, 3, kSize, kSize}}};
        trt::TrtEngine engine(args.engine, opt_shapes, 1, true, true);
        const std::string in_name = engine.inputs().at(0).name;
        engine.setInputShape(in_name, std::vector<int64_t>{1, 3, kSize, kSize});

        const auto& out_info = engine.outputs().at(0);
        const int channels = static_cast<int>(out_info.dims.d[1]);
        const int anchors = static_cast<int>(out_info.dims.d[2]);
        const int num_classes = channels - 4;
        std::printf("输出  : [1, %d, %d]  → %d 个锚点, %d 类\n",
                    channels, anchors, anchors, num_classes);

        // ---------------- 3. 缓冲 ----------------
        const size_t img_bytes = static_cast<size_t>(src_w) * src_h * 3;
        const size_t lb_floats = static_cast<size_t>(kSize) * kSize * 3;
        const size_t in_bytes = static_cast<size_t>(3) * kSize * kSize * sizeof(float);

        unsigned char* d_img = nullptr;
        float* d_lb = nullptr;
        float* d_chw = nullptr;
        float* d_out = nullptr;
        float* d_mean = nullptr;
        float* d_std = nullptr;
        CUDA_CHECK(cudaMalloc(&d_img, img_bytes));
        CUDA_CHECK(cudaMalloc(&d_lb, lb_floats * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_chw, in_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, out_info.bytes));
        CUDA_CHECK(cudaMalloc(&d_mean, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_std, 3 * sizeof(float)));

        const float mean[3] = {0.0f, 0.0f, 0.0f};
        const float stdv[3] = {1.0f, 1.0f, 1.0f};
        CUDA_CHECK(cudaMemcpy(d_mean, mean, sizeof(mean), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_std, stdv, sizeof(stdv), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_img, host_img.data(), img_bytes, cudaMemcpyHostToDevice));

        const preprocess::LetterboxParams lp =
            preprocess::computeLetterbox(src_w, src_h, kSize, kSize);

        // 服务里 CPU 侧输出缓冲也只分配一次（不能每帧 new）
        std::vector<float> raw;
        if (args.cpu_path) raw.resize(out_info.bytes / sizeof(float));

        cuda_utils::warmUpGpu();

        // ---------------- 4. 单帧流水线（可被 Graph 捕获）----------------
        auto stage_pre = [&] {
            preprocess::letterbox(d_img, src_w, src_h, d_lb, kSize, kSize, 114);
            preprocess::hwcToChwNormalize(d_lb, d_chw, kSize, kSize, 3, d_mean, d_std);
        };
        // 先把输入填好，再捕获（保持代码意图清晰）
        stage_pre();
        CUDA_CHECK(cudaDeviceSynchronize());

        // 热机一次
        engine.inferDevice(d_chw, d_out);
        CUDA_CHECK(cudaStreamSynchronize(engine.stream()));

        yolo::PipelineGraph graph;
        if (!args.no_graph) {
            if (graph.capture(engine.stream(), [&] { engine.enqueueDevice(d_chw, d_out); })) {
                std::printf("Graph : ✅ 已捕获\n");
            } else {
                std::printf("Graph : ❌ 捕获失败（%s）→ 回退直连\n", graph.err());
            }
        }

        auto stage_infer = [&] {
            if (graph.valid()) {
                graph.launch(engine.stream());
                CUDA_CHECK(cudaStreamSynchronize(engine.stream()));
            } else {
                engine.inferDevice(d_chw, d_out);
            }
        };

        // decode + 后处理：返回这一帧检出的框数
        auto stage_post = [&]() -> size_t {
            std::vector<yolo::Detection> dets;
            if (args.cpu_path) {
                CUDA_CHECK(cudaMemcpy(raw.data(), d_out, out_info.bytes,
                                      cudaMemcpyDeviceToHost));
                dets = yolo::decodeV8(raw.data(), num_classes, anchors, 0.25f);
            } else {
                dets = yolo::decodeV8Gpu(d_out, num_classes, anchors, 0.25f);
            }
            yolo::undoLetterbox(dets, lp, src_w, src_h);
            auto kept = yolo::nms(dets, 0.45f);
            return kept.size();
        };

        // ---------------- 5. 帧循环 ----------------
        const float budget = (args.budget_ms > 0.0f)
                                 ? args.budget_ms
                                 : 33.3f / static_cast<float>(std::max(1, args.cameras));

        const int total = args.warmup + args.frames;
        std::printf("\n开始：%d 帧预热 + %d 帧统计，单帧预算 %.2f ms（%d 路 × 30fps）\n",
                    args.warmup, args.frames, budget, args.cameras);

        std::vector<float> lat;
        lat.reserve(static_cast<size_t>(args.frames));
        std::vector<float> all_ms;
        all_ms.reserve(static_cast<size_t>(total));

        size_t det_sum = 0;

        for (int f = 0; f < total; ++f) {
            const auto t0 = std::chrono::steady_clock::now();

            stage_pre();
            stage_infer();
            const size_t n_det = stage_post();

            const auto t1 = std::chrono::steady_clock::now();
            const float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

            all_ms.push_back(ms);
            if (f >= args.warmup) {
                lat.push_back(ms);
                det_sum += n_det;
            }
        }

        if (lat.empty()) { std::printf("[FAIL] 统计帧数为 0\n"); return 1; }

        // ---------------- 6. 统计 ----------------
        std::vector<float> sorted = lat;
        std::sort(sorted.begin(), sorted.end());

        const float p50 = percentile(sorted, 50.0f);
        const float p90 = percentile(sorted, 90.0f);
        const float p99 = percentile(sorted, 99.0f);
        const float p999 = percentile(sorted, 99.9f);
        const float vmax = sorted.back();
        const float vmin = sorted.front();

        double sum = 0.0;
        for (float x : lat) sum += x;
        const double avg = sum / static_cast<double>(lat.size());

        size_t miss = 0;
        for (float x : lat) if (x > budget) ++miss;

        std::printf("\n---------------------------------------------------------\n");
        std::printf("延迟分位数（%zu 帧）\n", lat.size());
        std::printf("  min    : %8.3f ms\n", vmin);
        std::printf("  p50    : %8.3f ms\n", p50);
        std::printf("  p90    : %8.3f ms\n", p90);
        std::printf("  p99    : %8.3f ms\n", p99);
        std::printf("  p99.9  : %8.3f ms\n", p999);
        std::printf("  max    : %8.3f ms\n", vmax);
        std::printf("  mean   : %8.3f ms\n", avg);
        std::printf("---------------------------------------------------------\n");
        std::printf("单帧预算 %.2f ms  →  超预算 %zu / %zu 帧（%.2f%%）\n",
                    budget, miss, lat.size(), 100.0 * static_cast<double>(miss)
                                                / static_cast<double>(lat.size()));
        std::printf("平均每帧检出 %.1f 个框\n",
                    static_cast<double>(det_sum) / static_cast<double>(lat.size()));
        std::printf("---------------------------------------------------------\n");

        std::printf("\n  怎么读：\n");
        std::printf("    · p50 好看不代表能上车——**p99 / max 才是判据**\n");
        std::printf("    · p99 / p50 > 2 说明有抖动源（降频、调度、显存争用）\n");
        std::printf("    · 超预算帧数 > 0 表示这个配置在这台机器上不满足实时性\n");
        if (miss > 0) {
            std::printf("    ⚠️ 有 %zu 帧超预算，最大超 %.2f ms\n",
                        miss, vmax - budget);
        } else {
            std::printf("    ✅ 全部帧都在预算内（但注意：这台机器比 Orin 快约 5 倍）\n");
        }

        printHistogram(lat, p99);

        // ---------------- 7. 可选：导出 CSV ----------------
        if (!args.csv.empty()) {
            std::ofstream f(args.csv);
            f << "frame,latency_ms,over_budget\n";
            for (size_t i = 0; i < lat.size(); ++i) {
                f << i << ',' << lat[i] << ',' << (lat[i] > budget ? 1 : 0) << '\n';
            }
            std::printf("\n每帧延迟已导出: %s（%zu 行）\n", args.csv.c_str(), lat.size());
        }

        CUDA_CHECK(cudaFree(d_img));
        CUDA_CHECK(cudaFree(d_lb));
        CUDA_CHECK(cudaFree(d_chw));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_mean));
        CUDA_CHECK(cudaFree(d_std));
        return 0;

    } catch (const std::exception& e) {
        std::printf("\n[EXCEPTION] %s\n", e.what());
        return 1;
    }
}
