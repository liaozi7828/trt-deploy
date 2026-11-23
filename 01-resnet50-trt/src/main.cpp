// ---------------------------------------------------------------------------
// main.cpp —— 项目① 的完整流水线 + 4 项优化的 A/B 对比
//
//   [合成图像] → [GPU 前处理] → [TRT 推理] → [argmax] → [优化前后对比表]
//
// 本版本相比"朴素版"加了 4 项优化，每项都印出前后对比：
//   ① pinned memory        —— 主机侧页锁定，DMA 才能真异步
//   ② 向量化 normalize     —— 一个线程写 float4（16 字节）而不是 float（4 字节）
//   ③ letterbox            —— 等比缩放 + 居中填充（不拉伸，检测模型标准做法）
//   ④ 零拷贝推理           —— 输入/输出全程留在显存，省掉两趟 PCIe 往返
//
// 用法：
//   ./resnet50_demo --engine models/resnet50.engine --batch 1 --iters 50
//   ./resnet50_demo --onnx models/resnet50.onnx --save-engine models/resnet50.engine
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>     // --dump-input / --dump-output 用
#include <numeric>
#include <random>
#include <stdexcept>   // std::runtime_error（不能依赖别的头"顺便"带进来）
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "preprocess.h"
#include "trt_engine.h"

namespace {

struct Args {
    std::string model;          // .onnx 或 .engine
    std::string save_engine;    // 非空则把编译结果存盘
    std::string dump_input;     // 非空则把喂给引擎的输入写成 bin
    std::string dump_output;    // 非空则把引擎输出写成 bin
    std::string dump_raw;       // 非空则把**原始图像**写成 bin（验证前处理用）
    bool allow_tf32{true};      // --no-tf32 关掉，用于严格 FP32 精度验收
    int batch{1};
    int iters{50};
    int height{224};
    int width{224};
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("参数缺值: " + k);
            return argv[++i];
        };
        if (k == "--onnx" || k == "--engine") a.model = next();
        else if (k == "--save-engine")     a.save_engine = next();
        else if (k == "--dump-input")      a.dump_input = next();
        else if (k == "--dump-output")     a.dump_output = next();
        else if (k == "--dump-raw")        a.dump_raw = next();
        else if (k == "--batch")           a.batch = std::stoi(next());
        else if (k == "--iters")           a.iters = std::stoi(next());
        else if (k == "--size")            { a.height = a.width = std::stoi(next()); }
        else if (k == "--no-tf32")         a.allow_tf32 = false;
        else if (k == "-h" || k == "--help") {
            std::printf("用法: %s --onnx model.onnx [--batch N] [--iters N] [--size N]\n"
                        "          [--save-engine out.engine] [--no-tf32]\n"
                        "          [--dump-input in.bin] [--dump-output out.bin]\n"
                        "          [--dump-raw raw.bin]\n"
                        "  --no-tf32  : 关闭 TF32，得到严格 FP32（精度验收用）\n"
                        "  --dump-raw : 导出原始图像（uint8 值存成 float），\n"
                        "               用于 compare_preprocess.py 和 torchvision 对齐\n", argv[0]);
            std::exit(0);
        } else {
            std::printf("未知参数: %s\n", k.c_str());
            std::exit(1);
        }
    }
    if (a.model.empty()) {
        std::printf("[FAIL] 必须指定 --onnx 或 --engine\n");
        std::exit(1);
    }
    return a;
}

/// 造一张假的"图像"：真实项目里这里是读 JPEG + 解码
std::vector<unsigned char> makeFakeImage(int h, int w, unsigned seed = 42) {
    std::vector<unsigned char> img(static_cast<size_t>(h) * w * 3);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : img) v = static_cast<unsigned char>(dist(rng));
    return img;
}

/// 两个数组的最大绝对差（不用 <cmath>，省一个头）
float maxAbsDiff(const float* a, const float* b, size_t n) {
    float md = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > md) md = d;
    }
    return md;
}

/// 差异的完整统计——只看 max 是不够的
///
/// 判据用 **max + mean 双条件**，不要用"有多少元素不同"：
///   - padding 区 / 常数区天然相同，会拉低差异比例（本 kernel 有 25% 是常数 114）
///   - FMA 只在"舍入恰好不同"时才产生差异，很多结果是逐位相同的
///   所以"只有 10% 元素不同"完全正常，不能当作异常信号。
///
///   max 大 + mean 小  → 个别点偏离很远 → 逻辑错误（比如某条边越界）
///   max 小 + mean 小  → 处处差一丁点   → 舍入
struct DiffStats {
    float max_abs{0.0f};
    double mean_abs{0.0};
    size_t n_diff{0};      // 仅供参考，不参与判定
    size_t n{0};
};

DiffStats diffStats(const float* a, const float* b, size_t n) {
    DiffStats s;
    s.n = n;
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > s.max_abs) s.max_abs = d;
        if (d != 0.0f) ++s.n_diff;
        acc += static_cast<double>(d);
    }
    s.mean_abs = (n > 0) ? acc / static_cast<double>(n) : 0.0;
    return s;
}

/// 把张量写成自描述二进制文件，供 compare_precision.py 读取
///
/// 格式（小端）：
///   int32  ndim
///   int32  dims[ndim]
///   float32 data[prod(dims)]
///
/// 带头上信息的好处：Python 侧不用再传 --batch/--size，文件自己说明形状。
void writeTensorBin(const std::string& path,
                    const std::vector<float>& data,
                    const std::vector<int32_t>& dims) {
    size_t expect = 1;
    for (int32_t d : dims) expect *= static_cast<size_t>(d);
    if (expect != data.size()) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "导出 %s 失败：形状乘积 %zu 与数据长度 %zu 不符",
                      path.c_str(), expect, data.size());
        throw std::runtime_error(msg);
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("写不了文件: " + path);

    const int32_t nd = static_cast<int32_t>(dims.size());
    f.write(reinterpret_cast<const char*>(&nd), sizeof(nd));
    f.write(reinterpret_cast<const char*>(dims.data()),
            static_cast<std::streamsize>(sizeof(int32_t) * nd));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(sizeof(float) * data.size()));

    std::printf("  已导出 %-28s (%zu 个 float，形状 ", path.c_str(), data.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        std::printf(i + 1 < dims.size() ? "%d×" : "%d", dims[i]);
    }
    std::printf(")\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        const auto info = cuda_utils::queryDevice();

        std::printf("=========================================================\n");
        std::printf("ResNet50 TensorRT 部署 demo（含 4 项优化 A/B 对比）\n");
        std::printf("=========================================================\n");
        std::printf("GPU          : %s  (SM %d, sm_%d%d)\n",
                    info.name, info.sm_count, info.major, info.minor);
        std::printf("理论带宽     : %.0f GB/s\n", info.peak_bw_gbs);
        std::printf("模型         : %s\n", args.model.c_str());
        std::printf("batch        : %d\n", args.batch);
        std::printf("输入尺寸     : %d×%d\n\n", args.height, args.width);

        // ---------------- 1. 构建引擎 ----------------
        std::printf("[1/4] 构建/加载引擎\n");
        std::vector<std::pair<std::string, std::vector<int64_t>>> opt_shapes = {
            {"input", {args.batch, 3, args.height, args.width}},
        };
        trt::TrtEngine engine(args.model, opt_shapes, /*max_batch=*/8,
                              /*use_fp16=*/true, /*allow_tf32=*/args.allow_tf32);
        std::printf("%s", engine.describe().c_str());
        if (!args.save_engine.empty()) engine.save(args.save_engine);

        // ★ 动态 shape：必须先告诉引擎"这次跑多大"。
        //   不调这一步，getTensorShape 会返回 -1，后面推输出形状就全错。
        engine.setInputShape("input", std::vector<int64_t>{
            args.batch, 3, args.height, args.width});
        std::printf("已设置输入形状: batch=%d, 3x%dx%d\n",
                    args.batch, args.height, args.width);

        // ===================================================================
        // 2. 前处理（GPU）—— 三条路径 A/B/C
        // ===================================================================
        std::printf("\n[2/4] 前处理（GPU）\n");

        const int src_h = 480, src_w = 640;                 // 假设原图 640×480
        const size_t img_bytes = static_cast<size_t>(src_h) * src_w * 3;
        const size_t one_img_floats =
            static_cast<size_t>(args.height) * args.width * 3;
        const size_t resized_bytes = one_img_floats * sizeof(float);
        const size_t input_bytes =
            static_cast<size_t>(args.batch) * 3 * args.height * args.width * sizeof(float);

        unsigned char* d_img      = nullptr;
        float* d_resized          = nullptr;   // resize 输出（HWC）
        float* d_lb_hwc           = nullptr;   // letterbox 输出（HWC）
        float* d_input_naive      = nullptr;   // 朴素 normalize 输出（CHW）
        float* d_input_vec        = nullptr;   // 向量化 normalize 输出（CHW）
        float* d_input_lb         = nullptr;   // letterbox + normalize（CHW）
        float* d_mean             = nullptr;
        float* d_std              = nullptr;
        float* d_output           = nullptr;   // 零拷贝路径的输出（显存）
        unsigned char* h_img      = nullptr;   // pinned host 内存

        // ---- 优化 ①：pinned memory ----
        // pageable 内存做 cudaMemcpyAsync 会先拷到内部缓冲区，带宽打对折。
        // cudaMallocHost 分配的是页锁定内存，DMA 引擎能直接访问。
        CUDA_CHECK(cudaMallocHost(&h_img, img_bytes));
        {
            const auto tmp = makeFakeImage(src_h, src_w);
            std::memcpy(h_img, tmp.data(), img_bytes);
        }

        // 导出原始图像（uint8 值存成 float，反正是 0–255，float 精确表示）
        if (!args.dump_raw.empty()) {
            std::vector<float> raw(img_bytes);
            for (size_t i = 0; i < img_bytes; ++i) {
                raw[i] = static_cast<float>(h_img[i]);
            }
            writeTensorBin(args.dump_raw, raw, {src_h, src_w, 3});
        }

        CUDA_CHECK(cudaMalloc(&d_img, img_bytes));
        CUDA_CHECK(cudaMalloc(&d_resized, resized_bytes * args.batch));
        CUDA_CHECK(cudaMalloc(&d_lb_hwc, resized_bytes * args.batch));
        CUDA_CHECK(cudaMalloc(&d_input_naive, input_bytes));
        CUDA_CHECK(cudaMalloc(&d_input_vec, input_bytes));
        CUDA_CHECK(cudaMalloc(&d_input_lb, input_bytes));

        // ★ mean/std 必须在显存里！
        //   传主机栈数组进去，GPU 会访问非法地址 → illegal memory access
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stdv[3] = {0.229f, 0.224f, 0.225f};
        CUDA_CHECK(cudaMalloc(&d_mean, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_std, 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_std, stdv, 3 * sizeof(float), cudaMemcpyHostToDevice));

        // ★ 先预热 GPU：笔记本 GPU 时钟在 1.4–2.1 GHz 之间跳，
        //   不预热的话同一份代码两次测量能差 40% 以上（A/B 结论直接反号）。
        std::printf("  预热 GPU（顶满时钟）...\n");
        cuda_utils::warmUpGpu();

        // H2D 单独计时（pinned 的收益就体现在这里）
        const float h2d_ms = cuda_utils::benchmark([&] {
            CUDA_CHECK(cudaMemcpyAsync(d_img, h_img, img_bytes,
                                       cudaMemcpyHostToDevice, nullptr));
        }, 20, 200);
        std::printf("  H2D（pinned %.0f KB）        : %.4f ms\n",
                    img_bytes / 1024.0, h2d_ms);

        // ---- 路径 A：resize + 朴素 normalize ----
        auto run_naive = [&] {
            for (int b = 0; b < args.batch; ++b) {
                preprocess::resizeBilinear(
                    d_img, src_w, src_h,
                    d_resized + static_cast<size_t>(b) * one_img_floats,
                    args.width, args.height);
            }
            for (int b = 0; b < args.batch; ++b) {
                preprocess::hwcToChwNormalize(
                    d_resized + static_cast<size_t>(b) * one_img_floats,
                    d_input_naive + static_cast<size_t>(b) * 3 * args.height * args.width,
                    args.height, args.width, 3, d_mean, d_std);
            }
        };

        // ---- 路径 B：resize + 向量化 normalize（优化 ②）----
        auto run_vec4 = [&] {
            for (int b = 0; b < args.batch; ++b) {
                preprocess::resizeBilinear(
                    d_img, src_w, src_h,
                    d_resized + static_cast<size_t>(b) * one_img_floats,
                    args.width, args.height);
            }
            for (int b = 0; b < args.batch; ++b) {
                preprocess::hwcToChwNormalizeVec4(
                    d_resized + static_cast<size_t>(b) * one_img_floats,
                    d_input_vec + static_cast<size_t>(b) * 3 * args.height * args.width,
                    args.height, args.width, 3, d_mean, d_std);
            }
        };

        // ---- 路径 C：letterbox + 向量化 normalize（优化 ③）----
        auto run_letterbox = [&] {
            for (int b = 0; b < args.batch; ++b) {
                preprocess::letterbox(
                    d_img, src_w, src_h,
                    d_lb_hwc + static_cast<size_t>(b) * one_img_floats,
                    args.width, args.height, /*pad=*/114);
            }
            for (int b = 0; b < args.batch; ++b) {
                preprocess::hwcToChwNormalizeVec4(
                    d_lb_hwc + static_cast<size_t>(b) * one_img_floats,
                    d_input_lb + static_cast<size_t>(b) * 3 * args.height * args.width,
                    args.height, args.width, 3, d_mean, d_std);
            }
        };

        // 前处理只有 ~24 µs，必须：
        //   ① 预热顶满时钟  ② 跑足够多次  ③ 取中位数  ④ **交错测量 A/B**
        // 前三条做了还是不稳，因为先测完 A 再测 B 时时钟已经漂了。
        // 所以用 benchmarkPair 交错跑，让两者承受同样的漂移。
        float ms_naive = 0.0f;
        float ms_vec4 = 0.0f;
        cuda_utils::benchmarkPair(run_naive, run_vec4, 50, 500, 9, ms_naive, ms_vec4);
        const float ms_lb = cuda_utils::benchmark(run_letterbox, 50, 500);

        std::printf("  A  resize + 朴素 normalize   : %.4f ms\n", ms_naive);
        std::printf("  B  resize + 向量化 normalize : %.4f ms   (%+.1f%%)\n",
                    ms_vec4, 100.0f * (ms_vec4 - ms_naive) / ms_naive);
        std::printf("  C  letterbox（等比+填充）    : %.4f ms\n", ms_lb);

        // ---- 校验 B：向量化版必须和朴素版逐元素一致 ----
        {
            std::vector<float> a(input_bytes / sizeof(float));
            std::vector<float> b(input_bytes / sizeof(float));
            CUDA_CHECK(cudaMemcpy(a.data(), d_input_naive, input_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), d_input_vec, input_bytes, cudaMemcpyDeviceToHost));
            const float md = maxAbsDiff(a.data(), b.data(), a.size());
            std::printf("  校验 A vs B                  : max|diff| = %.3e  %s\n",
                        md, md == 0.0f ? "完全一致 OK" : "有差异，不要用 B！");
        }

        // ---- 校验 C：letterbox GPU vs CPU ----
        {
            std::vector<unsigned char> h_src(h_img, h_img + img_bytes);
            std::vector<float> ref(one_img_floats);
            std::vector<float> got(one_img_floats);
            preprocess::letterboxCpu(h_src.data(), src_w, src_h, ref.data(),
                                     args.width, args.height, 114);
            CUDA_CHECK(cudaMemcpy(got.data(), d_lb_hwc,
                                  one_img_floats * sizeof(float), cudaMemcpyDeviceToHost));
            const DiffStats s = diffStats(ref.data(), got.data(), ref.size());

            // 判据：max + mean 双条件（不用"差异元素比例"，理由见 diffStats 注释）
            //   值域 0–255 时 1 ulp ≈ 1.5e-5 ~ 3e-5，所以 max < 1e-4 已经覆盖 3 ulp
            const bool small_max  = (s.max_abs < 1e-4f);
            const bool small_mean = (s.mean_abs < 1e-5);
            const bool ok = small_max && small_mean;

            std::printf("  校验 letterbox GPU vs CPU\n");
            std::printf("      max|diff| = %.3e    mean|diff| = %.3e\n",
                        s.max_abs, s.mean_abs);
            std::printf("      有差异元素: %zu / %zu (%.1f%%)  [仅供参考]\n",
                        s.n_diff, s.n, 100.0 * static_cast<double>(s.n_diff) / s.n);
            std::printf("      → %s\n",
                        ok ? "OK：max 和 mean 都在 ulp 级，符合 FMA 舍入"
                           : "FAIL：max 或 mean 偏大，去查逻辑");
        }

        // ===================================================================
        // 3. 推理 —— 两条路径 A/B（优化 ④ 零拷贝）
        // ===================================================================
        std::printf("\n[3/4] TensorRT 推理\n");

        // 路径 1（旧）：host vector 进、host vector 出 → 含两趟 PCIe 往返
        std::vector<float> host_input(input_bytes / sizeof(float));
        CUDA_CHECK(cudaMemcpy(host_input.data(), d_input_vec, input_bytes,
                              cudaMemcpyDeviceToHost));
        std::vector<float> output;

        // 两条推理路径也交错测（2 ms 量级噪声小，但保持一致更稳妥）
        const size_t out_bytes = engine.outputBytes();
        if (out_bytes == 0) throw std::runtime_error("outputBytes() 为 0，形状没设对");
        CUDA_CHECK(cudaMalloc(&d_output, out_bytes));

        float ms_infer_copy = 0.0f;
        float ms_infer_zero = 0.0f;
        cuda_utils::benchmarkPair(
            [&] { engine.infer(host_input, output); },
            [&] { engine.inferDevice(d_input_vec, d_output); },
            20, args.iters, 9, ms_infer_copy, ms_infer_zero);

        // 把零拷贝路径的结果取回来做后处理
        output.resize(out_bytes / sizeof(float));
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, out_bytes,
                              cudaMemcpyDeviceToHost));

        // ---- 导出输入/输出，供 compare_precision.py 做精度对齐 ----
        // 导出的输入就是**真正喂给引擎的那份数据**（已归一化、CHW 布局），
        // 所以 Python 侧可以直接 model(input) 而不用复现 C++ 的随机数。
        if (!args.dump_input.empty()) {
            writeTensorBin(args.dump_input, host_input,
                           {args.batch, 3, args.height, args.width});
        }
        if (!args.dump_output.empty()) {
            writeTensorBin(args.dump_output, output,
                           {args.batch, static_cast<int32_t>(output.size() / args.batch)});
        }

        std::printf("  输出元素数          : %zu\n", output.size());
        std::printf("  路径1 infer()       : %.3f ms   (含 H2D + D2H)\n", ms_infer_copy);
        std::printf("  路径2 inferDevice() : %.3f ms   (零拷贝)\n", ms_infer_zero);
        std::printf("  省下                : %.3f ms   (%.1f%%)\n",
                    ms_infer_copy - ms_infer_zero,
                    100.0f * (ms_infer_copy - ms_infer_zero) / ms_infer_copy);

        // ===================================================================
        // 4. 后处理 + 优化前后对比
        // ===================================================================
        std::printf("\n[4/4] 后处理 + 汇总\n");
        const int classes = static_cast<int>(output.size() / args.batch);
        std::vector<std::pair<float, int>> top;
        for (int c = 0; c < std::min(classes, 1000); ++c) top.emplace_back(output[c], c);
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::printf("  top-5 (第 0 张图；输入是随机噪声，所以类别无意义):\n");
        for (int i = 0; i < 5 && i < static_cast<int>(top.size()); ++i) {
            std::printf("    #%d  class %-4d  logit %.4f\n",
                        i + 1, top[i].second, top[i].first);
        }

        const float total_before = ms_naive + ms_infer_copy;
        const float total_after  = ms_vec4 + ms_infer_zero;

        std::printf("\n=========================================================\n");
        std::printf("优化前后对比\n");
        std::printf("=========================================================\n");
        std::printf("                        优化前      优化后\n");
        std::printf("  preprocess          %7.4f ms  %7.4f ms\n", ms_naive, ms_vec4);
        std::printf("  infer               %7.4f ms  %7.4f ms\n", ms_infer_copy, ms_infer_zero);
        std::printf("  -----------------------------------------------------\n");
        std::printf("  合计                %7.4f ms  %7.4f ms\n", total_before, total_after);
        std::printf("  提速                %7.1f %%\n",
                    100.0f * (total_before - total_after) / total_before);
        std::printf("  吞吐(优化后)        %7.1f images/s\n",
                    args.batch * 1000.0f / total_after);
        std::printf("=========================================================\n");
        std::printf("下一步可以做：\n");
        std::printf("  1. polygraphy 转 FP16 → 预计再快 1.5–2 倍\n");
        std::printf("  2. CUDA Graph 录制 → 省掉 ~150 次 launch\n");
        std::printf("  3. 精度对齐（TRT vs PyTorch，max|diff| < 1e-4）\n");

        // ---------------- 释放 ----------------
        CUDA_CHECK(cudaFreeHost(h_img));
        CUDA_CHECK(cudaFree(d_img));
        CUDA_CHECK(cudaFree(d_resized));
        CUDA_CHECK(cudaFree(d_lb_hwc));
        CUDA_CHECK(cudaFree(d_input_naive));
        CUDA_CHECK(cudaFree(d_input_vec));
        CUDA_CHECK(cudaFree(d_input_lb));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_mean));
        CUDA_CHECK(cudaFree(d_std));
        return 0;

    } catch (const std::exception& e) {
        std::printf("\n[EXCEPTION] %s\n", e.what());
        return 1;
    }
}
