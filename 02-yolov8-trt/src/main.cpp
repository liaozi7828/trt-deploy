// ---------------------------------------------------------------------------
// main.cpp —— P3 项目② W1：YOLOv8 + TensorRT 跑通（含 decode）
//
//   [读图] → [letterbox 到 640×640] → [/255 + HWC→CHW] → [TRT 推理]
//          → [decode] → [反 letterbox] → [NMS] → [画框存 PPM]
//
// 验收（计划 P3 W1）：**能出框、能画到图上**
//
// 用法：
//   ./build/bin/yolov8_demo --engine models/yolov8n.engine --image dog.ppm
//   ./build/bin/yolov8_demo --onnx models/yolov8n.onnx --image dog.ppm \
//                            --save-engine models/yolov8n.engine
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

#include "cuda_graph.h"
#include "cuda_utils.h"
#include "decode_gpu.h"
#include "multi_stream.h"
#include "nms_gpu.h"
#include "postprocess.h"
#include "preprocess.h"
#include "trt_engine.h"

namespace {

/// 主机侧计时（毫秒），用于 CPU NMS 等不走 CUDA event 的测量
template <typename Fn>
double timeMsHost(Fn&& f, int iters) {
    if (iters < 1) iters = 1;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

/// 从一个样本序列算出：中位数、相对 σ(%)、95% 置信区间半宽(%)
///
/// 为什么不用极差：n 个样本的极差 ≈ 3.3σ（n=12 时），
/// 会把 σ=7% 说成 23%。极差是"分布宽度的最坏估计"，不是误差棒。
struct Stat {
    float med{0.0f};        // 中位数
    float sigma_pct{0.0f};  // 相对标准差
    float ci_pct{0.0f};     // 1.96·σ/√n，这才是"准到多少"
    int n{0};
};

Stat summarize(std::vector<float> v) {
    Stat s;
    s.n = static_cast<int>(v.size());
    if (s.n == 0) return s;
    std::sort(v.begin(), v.end());
    s.med = v[s.n / 2];
    if (s.n < 2) return s;

    double mean = 0.0;
    for (float x : v) mean += x;
    mean /= static_cast<double>(s.n);

    double var = 0.0;
    for (float x : v) var += (x - mean) * (x - mean);
    var /= static_cast<double>(s.n - 1);

    const double sd = std::sqrt(var);
    if (mean > 1e-12) {
        s.sigma_pct = static_cast<float>(100.0 * sd / mean);
        s.ci_pct = static_cast<float>(100.0 * sd / mean
                                      / std::sqrt(static_cast<double>(s.n)) * 1.96);
    }
    return s;
}

struct Args {
    std::string model;         // .onnx 或 .engine
    std::string save_engine;
    std::string image;         // 输入 PPM；空则用合成图
    std::string out_ppm{"out.ppm"};
    std::string dump_output;   // 导出原始输出，供和 ultralytics 对齐
    std::string dump_input;    // 导出前处理结果（交给 PyTorch 复算，隔离前处理差异）
    std::string dump_dets;     // 导出最终框（交给 verify_nms.py 和 ultralytics 的 NMS 比）
    std::string input_name{"images"};   // ultralytics 导出的 ONNX 输入名
    bool allow_tf32{true};     // --no-tf32 关掉，用于和 PyTorch FP32 严格对比
    bool use_gpu_nms{false};   // --gpu-nms 用 GPU 版 NMS
    bool bench_nms{false};     // --bench-nms 扫描候选数，找 CPU/GPU 交叉点
    int streams{1};            // --streams N 多流并发测试（W7）
    std::string model2;        // --engine2 第二个引擎（可重复多次，N 路交错）
    std::vector<std::string> extra_models;   // 除主引擎外的所有对比引擎
    bool null_test{false};     // --null-test 把同一个引擎对象测两次（零假设检验）
    bool gpu_decode{false};    // --gpu-decode 用 GPU 做 decode（省掉 2.69 MB 的 D2H）
    bool cuda_graph{false};    // --cuda-graph 用 CUDA Graph 重放推理（省 launch 开销）
    float conf{0.25f};
    float iou{0.45f};
    int iters{1};
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
        else if (k == "--save-engine")  a.save_engine = next();
        else if (k == "--image")        a.image = next();
        else if (k == "--out")          a.out_ppm = next();
        else if (k == "--dump-output")  a.dump_output = next();
        else if (k == "--dump-input")   a.dump_input = next();
        else if (k == "--dump-dets")    a.dump_dets = next();
        else if (k == "--input-name")   a.input_name = next();
        else if (k == "--no-tf32")      a.allow_tf32 = false;
        else if (k == "--gpu-nms")      a.use_gpu_nms = true;
        else if (k == "--bench-nms")    a.bench_nms = true;
        else if (k == "--streams")      a.streams = std::stoi(next());
        else if (k == "--engine2")    { a.model2 = next(); a.extra_models.push_back(a.model2); }
        else if (k == "--null-test")  a.null_test = true;
        else if (k == "--gpu-decode") a.gpu_decode = true;
        else if (k == "--cuda-graph") a.cuda_graph = true;
        else if (k == "--conf")         a.conf = std::stof(next());
        else if (k == "--iou")          a.iou = std::stof(next());
        else if (k == "--iters")        a.iters = std::stoi(next());
        else if (k == "-h" || k == "--help") {
            std::printf(
                "用法: %s --onnx y.onnx|--engine y.engine [选项]\n"
                "  --image img.ppm    输入图（P6），不填则用合成图\n"
                "  --out out.ppm      画框图输出，默认 out.ppm\n"
                "  --conf 0.25        置信度阈值\n"
                "  --iou  0.45        NMS 的 IoU 阈值\n"
                "  --dump-output f    导出原始输出（和 ultralytics 对齐用）\n"
                "  --dump-input f     导出前处理结果（喂给 PyTorch 复算，隔离前处理差异）\n"
                "  --dump-dets f      导出最终框（和 ultralytics 的 NMS 逐框对比用）\n"
                "  --input-name n     ONNX 输入名，默认 images\n"
                "  --no-tf32          关闭 TF32（和 PyTorch FP32 严格对比时用）\n"
                "  --gpu-nms          用 GPU 版 NMS（默认 CPU 版）\n"
                "  --bench-nms        扫描不同候选数，找 CPU/GPU NMS 的交叉点\n"
                "  --streams N        多流并发测试（W7），N=2..8\n"
                "  --engine2 f        对比引擎，可重复多次做 N 路交错（消除时钟漂移）\n"
                "  --null-test        把同一个引擎对象测两次（零假设检验，测方法残余误差）\n"
                "  --gpu-decode       在 GPU 上做 decode，只回传框（省 2.69 MB 的 D2H）\n"
                "  --cuda-graph       用 CUDA Graph 重放推理（省上百次 kernel launch）\n"
                "  --iters N          推理重复次数（默认 1）\n", argv[0]);
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

/// COCO 80 类（YOLOv8 默认训练集）
const char* const kCocoNames[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"};

/// 自描述二进制导出（和 p2 的 writeTensorBin 同格式，供 Python 读）
void writeTensorBin(const std::string& path, const std::vector<float>& data,
                    const std::vector<int32_t>& dims) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("写不了文件: " + path);
    const int32_t nd = static_cast<int32_t>(dims.size());
    f.write(reinterpret_cast<const char*>(&nd), 4);
    f.write(reinterpret_cast<const char*>(dims.data()), 4 * nd);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * 4));
    std::printf("  已导出 %s (%zu float)\n", path.c_str(), data.size());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        const auto info = cuda_utils::queryDevice();

        std::printf("=========================================================\n");
        std::printf("YOLOv8 + TensorRT（P3 W1：含 decode）\n");
        std::printf("=========================================================\n");
        std::printf("GPU   : %s (SM %d, sm_%d%d)\n",
                    info.name, info.sm_count, info.major, info.minor);
        std::printf("模型  : %s\n", args.model.c_str());

        // ---------------- 1. 读图 ----------------
        std::vector<unsigned char> host_img;
        int src_w = 0, src_h = 0;
        if (!args.image.empty()) {
            if (!yolo::readPpm(args.image, host_img, src_w, src_h)) {
                std::printf("[FAIL] 读不了 %s（只支持 P6 的 PPM）\n", args.image.c_str());
                return 1;
            }
            std::printf("图像  : %s  %dx%d\n", args.image.c_str(), src_w, src_h);
        } else {
            // 没有图就造一张：画两个亮矩形，方便肉眼确认框画在哪
            src_w = 640;
            src_h = 480;
            host_img.assign(static_cast<size_t>(src_w) * src_h * 3, 32);
            auto fill = [&](int x0, int y0, int w, int h,
                            unsigned char r, unsigned char g, unsigned char b) {
                for (int y = y0; y < y0 + h && y < src_h; ++y)
                    for (int x = x0; x < x0 + w && x < src_w; ++x) {
                        unsigned char* p = host_img.data()
                                           + (static_cast<size_t>(y) * src_w + x) * 3;
                        p[0] = r; p[1] = g; p[2] = b;
                    }
            };
            fill(80, 120, 160, 200, 200, 60, 60);
            fill(380, 200, 180, 140, 60, 160, 220);
            std::printf("图像  : 合成图 %dx%d（没有真实图时用于验证流水线）\n", src_w, src_h);
        }

        // ---------------- 2. 引擎 ----------------
        const int kSize = 640;   // YOLOv8 默认输入
        std::vector<std::pair<std::string, std::vector<int64_t>>> opt_shapes = {
            {args.input_name, {1, 3, kSize, kSize}},
        };
        trt::TrtEngine engine(args.model, opt_shapes, /*max_batch=*/1,
                              /*use_fp16=*/true, /*allow_tf32=*/args.allow_tf32);
        std::printf("%s", engine.describe().c_str());
        if (!args.save_engine.empty()) engine.save(args.save_engine);

        // 输入名可能是 images / input / input.0，取引擎报出来的
        const std::string in_name = engine.inputs().at(0).name;
        engine.setInputShape(in_name, std::vector<int64_t>{1, 3, kSize, kSize});
        std::printf("输入  : %s = [1, 3, %d, %d]\n", in_name.c_str(), kSize, kSize);

        const auto& out_info = engine.outputs().at(0);
        const int channels = static_cast<int>(out_info.dims.d[1]);
        const int anchors = static_cast<int>(out_info.dims.d[2]);
        const int num_classes = channels - 4;

        // ---- 输出布局自动识别 ----
        //
        //   [1, 84, 8400]：原始输出，需要自己 decode（通道在前）
        //   [1, N, 6]    ：引擎内部已做 NMS（如 EfficientNMS plugin），
        //                  每行 = x1,y1,x2,y2,score,cls，直接就是最终框
        //
        // 判别条件用第三维 == 6：我们的原始输出第三维是 8400，不会误判。
        const bool e2e_layout = (out_info.dims.nbDims == 3 &&
                                 out_info.dims.d[2] == 6 &&
                                 out_info.dims.d[1] <= 2000);

        if (e2e_layout) {
            std::printf("输出  : %s = [1, %d, 6]  → **引擎已内置 NMS**，直接是最终框\n",
                        out_info.name.c_str(), channels);
            std::printf("        布局：每行 x1,y1,x2,y2,score,cls（坐标在 640 letterbox 空间）\n");
        } else {
            std::printf("输出  : %s = [1, %d, %d]  → %d 个候选框, %d 类\n",
                        out_info.name.c_str(), channels, anchors, anchors, num_classes);
        }

        // ---------------- 3. 前处理（GPU） ----------------
        const size_t img_bytes = static_cast<size_t>(src_w) * src_h * 3;
        const size_t lb_floats = static_cast<size_t>(kSize) * kSize * 3;
        const size_t in_bytes = static_cast<size_t>(3) * kSize * kSize * sizeof(float);

        unsigned char* d_img = nullptr;
        float* d_lb = nullptr;      // letterbox 后（HWC, 0..255）
        float* d_chw = nullptr;     // 归一化后（CHW, 0..1）
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_img, img_bytes));
        CUDA_CHECK(cudaMalloc(&d_lb, lb_floats * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_chw, in_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, out_info.bytes));

        const preprocess::LetterboxParams lp =
            preprocess::computeLetterbox(src_w, src_h, kSize, kSize);
        std::printf("letterbox: scale=%.6f  new=%dx%d  pad=(%d,%d)\n",
                    lp.scale, lp.new_w, lp.new_h, lp.pad_x, lp.pad_y);

        // YOLO 只要 /255，不要 ImageNet 归一化 → mean=0, std=1
        float* d_mean = nullptr;
        float* d_std = nullptr;
        const float mean[3] = {0.0f, 0.0f, 0.0f};
        const float stdv[3] = {1.0f, 1.0f, 1.0f};
        CUDA_CHECK(cudaMalloc(&d_mean, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_std, 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_std, stdv, 3 * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_img, host_img.data(), img_bytes, cudaMemcpyHostToDevice));

        cuda_utils::warmUpGpu();

        auto run_pre = [&] {
            preprocess::letterbox(d_img, src_w, src_h, d_lb, kSize, kSize, 114);
            // (v/255 - 0)/1 → 正好是 v/255
            preprocess::hwcToChwNormalize(d_lb, d_chw, kSize, kSize, 3, d_mean, d_std);
        };

        // 先真实跑一次前处理，把 d_chw 填上。
        //
        // 为什么放在捕获之前：CUDA Graph 的**捕获 ≠ 执行**——
        //   cudaStreamBeginCapture 之后提交的操作只会被"记录"到图里
        //   （kernel 名、参数、指针、依赖关系），GPU 不会真的跑。
        //   所以捕获时 d_chw 里有没有数据，对图本身没有影响；
        //   图里存的是**指针地址**，真正读内容是 cudaGraphLaunch 的时候。
        //
        //   但"拿未初始化的显存取捕获"读起来就是错的，容易让人以为有 bug。
        //   所以这里先填好，让代码的意图和实际行为一致。
        run_pre();
        CUDA_CHECK(cudaDeviceSynchronize());

        // ===============================================================
        // CUDA Graph 捕获 —— 必须在**主推理之前**做，
        // 因为下面的主流水线要真的走 Graph 路径，否则延迟拆解里的
        // "infer" 还是直连的数，所谓"端到端优化"就只是算出来的。
        //
        // 三条硬约束（写错就捕获失败）：
        //  ① 非默认流：TrtEngine::stream() 是 cudaStreamCreate 建的 ✔
        //  ② 捕获期间不能同步：用 enqueueDevice()（只提交）
        //  ③ 图绑定 d_chw / d_out 的地址与形状 ✔
        // ===============================================================
        yolo::PipelineGraph pipe_graph;
        if (args.cuda_graph) {
            engine.inferDevice(d_chw, d_out);          // 热机（tactic 绑定在第一次完成）
            CUDA_CHECK(cudaStreamSynchronize(engine.stream()));

            // 先试最严格的 ThreadLocal；含 plugin 的引擎通常会失败，
            // 再退一步试 Relaxed（仍然不允许同步，但允许更多 API）。
            auto try_capture = [&](cudaStreamCaptureMode m) {
                return pipe_graph.capture(engine.stream(), [&] {
                    engine.enqueueDevice(d_chw, d_out);
                }, m);
            };

            bool ok = try_capture(cudaStreamCaptureModeThreadLocal);
            if (!ok) {
                std::printf("CUDA Graph : ThreadLocal 模式失败（%s），改试 Relaxed\n",
                            pipe_graph.err());
                ok = try_capture(cudaStreamCaptureModeRelaxed);
            }

            if (ok) {
                std::printf("CUDA Graph : ✅ 已捕获，主流水线走 Graph 重放\n");
            } else {
                std::printf("CUDA Graph : ❌ 捕获失败 → 回退直连\n");
                std::printf("             原因：%s\n", pipe_graph.err());
                std::printf("             含义：这个引擎不能被 CUDA Graph 捕获，"
                            "所以它拿不到那 1.4 ms 的收益。\n");
            }
        }

        // 主推理入口：只有捕获成功才用 Graph
        auto run_infer = [&] {
            if (pipe_graph.valid()) {
                pipe_graph.launch(engine.stream());
                CUDA_CHECK(cudaStreamSynchronize(engine.stream()));
            } else {
                engine.inferDevice(d_chw, d_out);
            }
        };

        const float pre_ms = cuda_utils::benchmark(run_pre, 20, 100, 3);
        std::printf("前处理 : %.3f ms\n", pre_ms);

        // ---------------- 4. 推理 ----------------
        const float infer_ms = cuda_utils::benchmark(
            run_infer, 10, std::max(1, args.iters), 3);
        std::printf("推理   : %.3f ms%s\n", infer_ms,
                    pipe_graph.valid() ? "（Graph 重放）" : "");

        // ---------------------------------------------------------------
        // 输出回传 + decode：两条路径
        //
        //   CPU decode（默认）：D2H 2.69 MB（0.417 ms）→ CPU 扫 2.82 MB（0.335 ms）
        //   GPU decode（--gpu-decode）：kernel 在设备上挑出 46 个框 →
        //                              只回传 46×6×4 = 1.1 KB
        //
        // 这两项合计 0.75 ms，占端到端 27%，而它们本质是"把 2.7 MB 摸两遍"。
        // ---------------------------------------------------------------
        std::vector<float> raw;
        float d2h_ms = 0.0f;

        if (!args.gpu_decode && !e2e_layout) {
            raw.resize(out_info.bytes / sizeof(float));
            d2h_ms = cuda_utils::benchmark([&] {
                CUDA_CHECK(cudaMemcpy(raw.data(), d_out, out_info.bytes,
                                      cudaMemcpyDeviceToHost));
            }, 3, 10, 3);
            std::printf("D2H    : %.3f ms（%zu 字节 = %.2f MB）\n",
                        d2h_ms, out_info.bytes,
                        static_cast<double>(out_info.bytes) / 1048576.0);
        } else if (e2e_layout) {
            std::printf("D2H    : 稍后回传 %.2f KB（E2E 布局：[1,%d,6]）\n",
                        static_cast<double>(out_info.bytes) / 1024.0, channels);
        } else {
            std::printf("D2H    : 跳过 2.69 MB 回传（GPU decode 只回传 ~1.1 KB）\n");
        }

        if (!args.dump_input.empty()) {
            std::vector<float> chw(in_bytes / sizeof(float));
            CUDA_CHECK(cudaMemcpy(chw.data(), d_chw, in_bytes, cudaMemcpyDeviceToHost));
            writeTensorBin(args.dump_input, chw, {1, 3, kSize, kSize});
        }

        if (!args.dump_output.empty()) {
            // dump 需要完整 raw，即使走 GPU decode 也补一次（不计入延迟）
            if (raw.empty()) {
                raw.resize(out_info.bytes / sizeof(float));
                CUDA_CHECK(cudaMemcpy(raw.data(), d_out, out_info.bytes,
                                      cudaMemcpyDeviceToHost));
            }
            writeTensorBin(args.dump_output, raw, {1, channels, anchors});
        }

        // ---------------- 5. 后处理 ----------------
        std::vector<yolo::Detection> dets;
        float dec_ms = 0.0f;        // decode 总耗时（含回传）
        float deck_ms = 0.0f;       // 仅 kernel（GPU 路径才有意义）

        if (e2e_layout) {
            // ===== E2E 路径：引擎已 NMS，只需回传 [1,N,6] 并解析 =====
            //
            // 对比一下回传量：
            //   原始布局 [1,84,8400] → 2.69 MB
            //   GPU decode 后         → 1.1 KB（46 个通过阈值的框）
            //   E2E 布局 [1,300,6]    → 7.2 KB（固定 300 行，含零填充）
            std::printf("\n[后处理] E2E 布局：引擎直出最终框，无需 decode / NMS\n");

            const int n_rows = channels;   // [1, N, 6] 的 N
            raw.resize(out_info.bytes / sizeof(float));
            d2h_ms = cuda_utils::benchmark([&] {
                CUDA_CHECK(cudaMemcpy(raw.data(), d_out, out_info.bytes,
                                      cudaMemcpyDeviceToHost));
            }, 3, 10, 3);

            auto parse_e2e = [&] {
                std::vector<yolo::Detection> v;
                v.reserve(32);
                for (int i = 0; i < n_rows; ++i) {
                    const float* r = raw.data() + static_cast<size_t>(i) * 6;
                    if (r[4] < args.conf) continue;          // 零填充行分数为 0，自动跳过
                    yolo::Detection d;
                    d.x1 = r[0]; d.y1 = r[1]; d.x2 = r[2]; d.y2 = r[3];
                    d.score = r[4]; d.cls = static_cast<int>(r[5]);
                    v.push_back(d);
                }
                return v;
            };
            dec_ms = static_cast<float>(timeMsHost([&] {
                auto v = parse_e2e();
                (void)v;
            }, 20));
            dets = parse_e2e();
            std::printf("  回传 %.2f KB → 解析出 %zu 个框\n",
                        static_cast<double>(out_info.bytes) / 1024.0, dets.size());
        } else if (args.gpu_decode) {
            // 纯 kernel（不含回传），用来把"计算"和"搬运"分开
            deck_ms = cuda_utils::benchmark([&] {
                yolo::decodeV8GpuKernelOnly(d_out, num_classes, anchors, args.conf);
            }, 5, 20, 3);

            // 含回传 1.1 KB
            dec_ms = cuda_utils::benchmark([&] {
                dets = yolo::decodeV8Gpu(d_out, num_classes, anchors, args.conf);
            }, 5, 20, 3);

            // ---- 同一次运行内的对照（关键：不能跨运行减）----
            //
            // 之前我写死 2.819 ms 当基线去减，那是**跨运行对比**：
            // 两次运行的 infer 能差 0.6 ms（热漂移），减出来的"省 0.557 ms"毫无意义。
            // 正确做法：只比"要优化的那两个组件"（D2H + decode），而且都在本进程内测。
            raw.resize(out_info.bytes / sizeof(float));
            const float cpu_d2h = cuda_utils::benchmark([&] {
                CUDA_CHECK(cudaMemcpy(raw.data(), d_out, out_info.bytes,
                                      cudaMemcpyDeviceToHost));
            }, 3, 10, 3);
            const float cpu_dec = static_cast<float>(timeMsHost([&] {
                auto d = yolo::decodeV8(raw.data(), num_classes, anchors, args.conf);
                yolo::undoLetterbox(d, lp, src_w, src_h);
            }, 20));

            // 校验：和 CPU 版逐个比
            auto cpu_dets = yolo::decodeV8(raw.data(), num_classes, anchors, args.conf);
            std::sort(cpu_dets.begin(), cpu_dets.end(),
                      [](const yolo::Detection& a, const yolo::Detection& b) {
                          return a.score > b.score;
                      });
            bool same = (cpu_dets.size() == dets.size());
            float max_cd = 0.0f, max_sd = 0.0f;
            if (same) {
                for (size_t i = 0; i < dets.size(); ++i) {
                    max_cd = std::max({max_cd, std::fabs(cpu_dets[i].x1 - dets[i].x1),
                                       std::fabs(cpu_dets[i].y1 - dets[i].y1),
                                       std::fabs(cpu_dets[i].x2 - dets[i].x2),
                                       std::fabs(cpu_dets[i].y2 - dets[i].y2)});
                    max_sd = std::max(max_sd, std::fabs(cpu_dets[i].score - dets[i].score));
                    if (cpu_dets[i].cls != dets[i].cls) same = false;
                }
            }
            std::printf("\n[GPU decode] 同一次运行内的对照（只比要优化的两个组件）\n");
            std::printf("  CPU 路径: D2H %.3f + decode %.3f = %.3f ms\n",
                        cpu_d2h, cpu_dec, cpu_d2h + cpu_dec);
            std::printf("  GPU 路径: D2H ~0.000 + decode %.3f = %.3f ms"
                        "（其中纯 kernel %.3f）\n", dec_ms, dec_ms, deck_ms);
            const float saved = (cpu_d2h + cpu_dec) - dec_ms;
            std::printf("  省 %.3f ms（%.0f%%）\n",
                        saved, 100.0f * saved / std::max(cpu_d2h + cpu_dec, 1e-6f));
            std::printf("  校验: %s（%zu vs %zu 个，坐标差 %.4f px，分数差 %.6f）\n",
                        same ? "✅ 与 CPU 版一致" : "❌ 不一致",
                        dets.size(), cpu_dets.size(), max_cd, max_sd);
            std::printf("  注：回传已合并成**一次** cudaMemcpy + pinned 主机内存，\n");
            std::printf("      所以 1.1 KB 只要 %.3f ms；早期两次拷贝时要 0.113 ms。\n",
                        dec_ms - deck_ms);
        } else {
            dec_ms = static_cast<float>(timeMsHost([&] {
                auto d = yolo::decodeV8(raw.data(), num_classes, anchors, args.conf);
                yolo::undoLetterbox(d, lp, src_w, src_h);
            }, 20));
            dets = yolo::decodeV8(raw.data(), num_classes, anchors, args.conf);
        }

        std::printf("\n[后处理]\n");
        std::printf("  decode 后      : %zu 个候选（conf > %.2f）\n", dets.size(), args.conf);

        yolo::undoLetterbox(dets, lp, src_w, src_h);

        // ---- W2：CPU NMS vs GPU NMS 的 A/B ----
        std::vector<yolo::Detection> kept;
        float nms_ms = 0.0f;      // 提到外层，供"延迟拆解"使用
        {
            const int rep = 20;
            // CPU 版（nms 会原地排序，所以每次用副本）
            double cpu_ms = timeMsHost([&] {
                auto c = dets;
                yolo::nms(c, args.iou);
            }, rep);
            nms_ms = static_cast<float>(cpu_ms);
            auto cpu_dets = dets;
            auto cpu_kept = yolo::nms(cpu_dets, args.iou);

            // GPU 版
            double gpu_ms = timeMsHost([&] {
                auto g = dets;
                yolo::nmsGpu(g, args.iou);
            }, rep);
            auto gpu_dets = dets;
            auto gpu_kept = yolo::nmsGpu(gpu_dets, args.iou);

            // 一致性校验：GPU 版必须和 CPU 版给出**完全相同**的框
            bool same = (cpu_kept.size() == gpu_kept.size());
            if (same) {
                for (size_t i = 0; i < cpu_kept.size(); ++i) {
                    if (cpu_kept[i].cls != gpu_kept[i].cls ||
                        cpu_kept[i].score != gpu_kept[i].score ||
                        cpu_kept[i].x1 != gpu_kept[i].x1 ||
                        cpu_kept[i].y2 != gpu_kept[i].y2) {
                        same = false;
                        break;
                    }
                }
            }

            std::printf("  NMS 后         : %zu 个框（iou > %.2f 的被抑制）\n",
                        cpu_kept.size(), args.iou);
            std::printf("      CPU NMS    : %8.4f ms\n", cpu_ms);
            std::printf("      GPU NMS    : %8.4f ms   (%+.1f%%)\n", gpu_ms,
                        100.0 * (gpu_ms - cpu_ms) / (cpu_ms > 1e-9 ? cpu_ms : 1.0));
            std::printf("      结果一致   : %s\n", same ? "✅ 完全相同" : "❌ 不一致！");

            kept = args.use_gpu_nms ? gpu_kept : cpu_kept;

            // ---- 交叉点扫描：候选数越多，GPU 越值 ----
            if (args.bench_nms) {
                std::printf("\n[交叉点扫描] 不同 conf 阈值下的候选数与耗时\n");
                std::printf("  %-8s %-8s %-12s %-12s %-14s %-8s\n",
                            "conf", "候选数", "CPU(ms)", "GPU总(ms)", "GPU内核(ms)", "谁快");
                for (float c : {0.9f, 0.7f, 0.5f, 0.3f, 0.25f, 0.1f, 0.05f, 0.01f, 0.001f}) {
                    auto d = yolo::decodeV8(raw.data(), num_classes, anchors, c);
                    if (d.empty()) continue;
                    yolo::undoLetterbox(d, lp, src_w, src_h);
                    const int n = static_cast<int>(d.size());
                    const int rep = (n > 1000) ? 3 : 20;

                    double c_ms = timeMsHost([&] {
                        auto cc = d;
                        yolo::nms(cc, args.iou);
                    }, rep);
                    double g_ms = timeMsHost([&] {
                        auto gg = d;
                        yolo::nmsGpu(gg, args.iou);
                    }, rep);
                    // 只算 kernel：不含传输、不含贪心 —— 用来判断"慢在计算还是慢在搬运"
                    double k_ms = yolo::benchIouMatrixGpu(d, args.iou, 5, rep);

                    std::printf("  %-8.3f %-8d %-12.4f %-12.4f %-14.4f %-8s\n",
                                c, n, c_ms, g_ms, k_ms, (g_ms < c_ms ? "GPU" : "CPU"));
                }
                std::printf("\n  读法：\n");
                std::printf("    GPU内核 ≪ GPU总  → 瓶颈在\"传输 + 贪心\"，不在计算\n");
                std::printf("    GPU总 < CPU      → 这个规模下 GPU 值得用\n");
                std::printf("    n 越大 GPU总 涨得越快（n² 下载），说明设计不可扩展\n");
            }
        }

        std::printf("\n检测结果（按分数降序，最多显示 20 个）:\n");
        const int show = std::min<int>(20, static_cast<int>(kept.size()));
        for (int i = 0; i < show; ++i) {
            const auto& d = kept[i];
            const char* name = (d.cls >= 0 && d.cls < 80) ? kCocoNames[d.cls] : "?";
            std::printf("  #%-2d %-14s %.3f  [%.1f, %.1f, %.1f, %.1f]\n",
                        i + 1, name, d.score, d.x1, d.y1, d.x2, d.y2);
        }
        if (kept.empty()) {
            std::printf("  （没有框——如果是合成图或噪声图，这是正常的；\n");
            std::printf("    真实图片跑不出来说明前处理或 decode 有问题）\n");
        }

        // ---- 导出最终框，供和 ultralytics 的 NMS 对比（verify_nms.py 用）----
        //
        // 为什么要把框落盘：
        //   CPU/GPU NMS 的 A/B 只能证明"两个实现一致"（自证），
        //   证明"语义对不对"必须和独立实现比 —— 即 ultralytics.non_max_suppression。
        //   落盘后 Python 就能逐框比对，不用靠肉眼读屏。
        if (!args.dump_dets.empty()) {
            std::ofstream f(args.dump_dets);
            if (!f) {
                std::printf("[WARN] 写不了 %s\n", args.dump_dets.c_str());
            } else {
                // 第一行写 letterbox 参数，Python 侧才能把两边映射到同一坐标系
                f << "# letterbox scale=" << lp.scale
                  << " pad_x=" << lp.pad_x
                  << " pad_y=" << lp.pad_y
                  << " src_w=" << src_w
                  << " src_h=" << src_h
                  << " conf=" << args.conf
                  << " iou=" << args.iou << "\n";
                f << "# cls score x1 y1 x2 y2\n";
                for (const auto& d : kept) {
                    f << d.cls << ' ' << d.score << ' '
                      << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << '\n';
                }
                std::printf("\n最终框已导出: %s（%zu 个）\n",
                            args.dump_dets.c_str(), kept.size());
            }
        }

        // ---------------- 6. 画框 ----------------
        yolo::drawBoxes(host_img.data(), src_w, src_h, kept, kCocoNames, 80);
        if (yolo::writePpm(args.out_ppm, host_img.data(), src_w, src_h)) {
            std::printf("\n画框图已保存: %s\n", args.out_ppm.c_str());
            std::printf("转 PNG 看: python -c \"from PIL import Image;"
                        "Image.open('%s').save('%s.png')\"\n",
                        args.out_ppm.c_str(), args.out_ppm.c_str());
        }

        std::printf("\n---------------------------------------------------------\n");
        std::printf("延迟拆解（单张，端到端）%s\n",
                    args.gpu_decode ? "  【GPU decode】" : "");
        std::printf("  preprocess : %7.3f ms   读图后的 letterbox + /255\n", pre_ms);
        std::printf("  infer      : %7.3f ms   引擎 inferDevice\n", infer_ms);
        if (args.gpu_decode) {
            std::printf("  decode     : %7.3f ms   GPU kernel（其中纯 kernel %.3f）\n",
                        dec_ms, deck_ms);
            std::printf("  D2H        : %7.3f ms   只回传 ~1.1 KB（原 0.417 ms）\n", 0.0);
        } else {
            std::printf("  D2H        : %7.3f ms   把 %.2f MB 输出搬回主机\n",
                        d2h_ms, static_cast<double>(out_info.bytes) / 1048576.0);
            std::printf("  decode     : %7.3f ms   decodeV8 + 反 letterbox（CPU）\n", dec_ms);
        }
        std::printf("  NMS        : %7.3f ms   CPU 逐类贪心\n", nms_ms);
        std::printf("  --------------------------------\n");
        const float total_ms = pre_ms + infer_ms + d2h_ms + dec_ms + nms_ms;
        std::printf("  合计       : %7.3f ms   ← 真实端到端\n", total_ms);
        if (args.gpu_decode) {
            std::printf("  （省下的 0.6+ ms 请看上面「同一次运行内的对照」，不要跨运行减）\n");
        }
        std::printf("---------------------------------------------------------\n");

        // ===============================================================
        // W7：多流并发 —— 吞吐 vs 延迟
        //
        // ⚠️ 早期版本是"先测串行 60 张，再测并发 60 张"，
        //    那是**串行测量**，前后两段撞上不同的 GPU 时钟状态，
        //    三次运行得出 1.25× / 1.48× / 1.10× —— 全是噪声。
        //
        // 现在改成和 §5.1 同一套方法：**交错 + 轮转 + 逐轮比值 + CI**。
        // ===============================================================
        if (args.streams > 1) {
            std::printf("\n[W7 多流并发] %d 路（交错测量）\n", args.streams);
            yolo::MultiStreamRunner runner(engine.rawEngine(), args.streams,
                                           in_name, out_info.name,
                                           in_bytes, out_info.bytes);

            constexpr int kRoundsS = 6;
            constexpr int kBatch = 128;    // 块 ≈ 128×2.1 = 270 ms，必须远大于 DVFS 反应时间
            const int total = kRoundsS * kBatch * 2;

            // 预热
            for (int i = 0; i < kBatch; ++i) engine.inferDevice(d_chw, d_out);
            for (int i = 0; i < args.streams; ++i) {
                runner.submit(i, d_chw, runner.outputBuffer(i));
            }
            runner.syncAll();

            std::vector<float> t_serial(kRoundsS, 0.0f), t_multi(kRoundsS, 0.0f);
            for (int r = 0; r < kRoundsS; ++r) {
                for (int k = 0; k < 2; ++k) {
                    const bool serial = ((r + k) % 2 == 0);   // 轮转，消除位置偏差
                    const auto t0 = std::chrono::steady_clock::now();
                    if (serial) {
                        for (int i = 0; i < kBatch; ++i) engine.inferDevice(d_chw, d_out);
                    } else {
                        for (int i = 0; i < kBatch; ++i) {
                            const int s = i % args.streams;
                            runner.submit(s, d_chw, runner.outputBuffer(s));
                        }
                        runner.syncAll();
                    }
                    const auto t1 = std::chrono::steady_clock::now();
                    const float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    (serial ? t_serial : t_multi)[r] = ms;
                }
            }

            // 逐轮比值（多流耗时 / 串行耗时），比值 <1 表示多流更快
            std::vector<float> rr;
            rr.reserve(kRoundsS);
            for (int r = 0; r < kRoundsS; ++r) {
                if (t_serial[r] > 1e-6f) rr.push_back(t_multi[r] / t_serial[r]);
            }
            const Stat ratio_st = summarize(rr);
            const Stat ser_st = summarize(t_serial);
            const Stat mul_st = summarize(t_multi);

            const double tp_serial = kBatch * 1000.0 / ser_st.med;
            const double tp_multi = kBatch * 1000.0 / mul_st.med;

            std::printf("  串行（1 路）  : %8.2f ms / %d 张  →  %7.1f img/s\n",
                        ser_st.med, kBatch, tp_serial);
            std::printf("  并发（%d 路）  : %8.2f ms / %d 张  →  %7.1f img/s\n",
                        args.streams, mul_st.med, kBatch, tp_multi);
            std::printf("  多流/串行 比值: %.3f ± %.1f%%（95%%CI）  σ=%.1f%%\n",
                        ratio_st.med, ratio_st.ci_pct, ratio_st.sigma_pct);
            if (ratio_st.med > 1e-6f) {
                std::printf("  吞吐提升      : %.2f×\n", 1.0 / ratio_st.med);
            }
            std::printf("\n  判读：\n");
            if (ratio_st.ci_pct > 5.0f) {
                std::printf("    ⚠️ CI 偏大（±%.1f%%）→ 这个结论不可信，"
                            "只能定性说\"多流%s\"\n",
                            ratio_st.ci_pct, ratio_st.med < 1.0f ? "有收益" : "无收益");
            } else if (std::fabs(1.0f - ratio_st.med) * 100.0f < ratio_st.ci_pct) {
                std::printf("    比值与 1.000 的差 < CI → **多流没有可测出的收益**\n");
            } else if (ratio_st.med < 1.0f) {
                std::printf("    多流确实更快，且超过噪声 → 加流有效\n");
            } else {
                std::printf("    多流更慢，且超过噪声 → 加流有害（抢带宽/SM）\n");
            }
            std::printf("    单张延迟会变差是正常的：N 路抢同一批 SM。\n");
            std::printf("    共 %d 张（%d 轮 × %d 张 × 2 种方式）\n", total, kRoundsS, kBatch);
        }

        // ===============================================================
        // N 路交错对比（消除 GPU 时钟漂移）
        //
        // 为什么不能"先测 A 再测 B 再测 C"：
        //   笔记本 GPU 主频随温度/功耗在 1.4–2.1 GHz 之间漂移，
        //   串行测量时后半段必然更热 → 系统性偏向先测的那个。
        //   （P2 里实测过：同一份代码两次得出 +46% / −35%）
        //
        // 为什么不能"两两比完再拼成表"：
        //   A<B 和 B<C 是在不同的运行里测的，拼起来不保证自洽。
        //   所以这里**一次加载全部引擎，每轮依次测一遍**，
        //   所有引擎共享同一段温度曲线，比值才可横向比较。
        //
        // 附带一个自检：如果传同一个引擎两次，比值应该是 1.000×。
        //   偏离多少，就是这套测量方法的残余误差。
        // ===============================================================
        if (!args.extra_models.empty() || args.null_test) {
            // 引擎列表：显式保存指针，这样 --null-test 可以把同一个对象塞两次
            std::vector<trt::TrtEngine*> engines{&engine};
            std::vector<std::string> labels{args.model};

            std::vector<std::unique_ptr<trt::TrtEngine>> holder;   // 保持生命周期
            for (const auto& m : args.extra_models) {
                // ⚠️ 从 .engine 加载时 use_fp16/allow_tf32 无效：
                //    TRT 11 是 strong typing，精度已经烧在文件里。
                holder.emplace_back(new trt::TrtEngine(m, opt_shapes, 1, true, args.allow_tf32));
                const std::string nm = holder.back()->inputs().at(0).name;
                holder.back()->setInputShape(nm, std::vector<int64_t>{1, 3, kSize, kSize});
                engines.push_back(holder.back().get());
                labels.push_back(m);
            }
            if (args.null_test) {
                // 真正的零假设：**同一个引擎对象**测两次。
                // 不重新加载，所以没有"分配地址不同"这个混杂因素。
                engines.push_back(&engine);
                labels.push_back("(null test: 同一个引擎对象)");
            }

            const int n = static_cast<int>(engines.size());
            std::printf("\n[N 路交错对比] %d 项\n", n);
            for (int i = 0; i < n; ++i) {
                std::printf("  [%d] %s\n", i, labels[i].c_str());
            }

            // 公平性检查：输出缓冲要够大
            size_t need = out_info.bytes;
            for (auto* e : engines) need = std::max(need, e->outputs().at(0).bytes);
            float* d_cmp = d_out;
            float* d_big = nullptr;
            if (need > out_info.bytes) {
                CUDA_CHECK(cudaMalloc(&d_big, need));
                d_cmp = d_big;
                std::printf("  （输出缓冲按最大者分配：%zu 字节）\n", need);
            }

            auto runOne = [&](int i) { engines[i]->inferDevice(d_chw, d_cmp); };

            // 预热
            for (int i = 0; i < n; ++i) runOne(i);
            CUDA_CHECK(cudaDeviceSynchronize());

            // N 路交错：12 轮，每轮把所有引擎各测一遍
            //
            // 三个关键细节（少一个噪声就会涨到 20%+）：
            //
            //  ① 轮内**轮转起点**：第 r 轮从引擎 (r mod n) 开始。
            //     否则引擎 0 永远在"轮首"、引擎 n-1 永远在"轮尾"，
            //     而轮内温度在爬 → 位置变成系统性偏差。
            //
            //  ② **一轮只同步一次**（不是每个引擎同步一次）。
            //     每次 cudaEventSynchronize 都会让 GPU 短暂空闲，
            //     DVFS 检测到空闲就降频 → 下一个引擎在更低的频率上测。
            //     4 个引擎 × 9 轮 = 36 次空闲，是 20–30% 抖动的元凶。
            //
            //  ③ 每次切换引擎后先跑 1 次**不计时**的（暖 ICache / TLB / 显存池），
            //     再开始计时，避免"冷启动"混进样本。
            // 块长度是这套方法最关键的一个参数。
            //
            // 实测教训：
            //   kIters=50（块 ≈115 ms）→ 逐轮比值 σ = 12%，CI ±6.9%
            //   原因：DVFS 的反应时间尺度也是 ~100 ms，
            //         块太短，每一块撞上的是"某个瞬间的时钟状态"，
            //         逐块之间时钟不同 → 比值抖得厉害。
            //   拉长块（内部平均掉更多时钟状态）比"多测几轮"有效得多：
            //         σ 大致按 1/√(块长) 下降。
            //
            // 代价是总时长：n 个引擎 × (iters+1) 次 × 单次耗时 × 轮数。
            constexpr int kRounds = 8;
            constexpr int kItersPerRound = 200;
            std::printf("  （每块 %d 次 × %d 轮 × %d 引擎 ≈ 预计 %.0f 秒；块拉长是为了压 σ）\n",
                        kItersPerRound, kRounds, n,
                        static_cast<double>(n) * (kItersPerRound + 1) * 2.5
                            * kRounds / 1000.0);

            std::vector<std::vector<float>> samples(n);
            std::vector<cudaEvent_t> ev0(n), ev1(n);
            for (int i = 0; i < n; ++i) {
                CUDA_CHECK(cudaEventCreate(&ev0[i]));
                CUDA_CHECK(cudaEventCreate(&ev1[i]));
            }

            for (int r = 0; r < kRounds; ++r) {
                for (int k = 0; k < n; ++k) {
                    const int i = (r + k) % n;      // ① 轮转起点
                    runOne(i);                       // ③ 预热一次，不计时
                    CUDA_CHECK(cudaEventRecord(ev0[i]));
                    for (int t = 0; t < kItersPerRound; ++t) runOne(i);
                    CUDA_CHECK(cudaEventRecord(ev1[i]));
                }
                // ② 一轮只同步一次：这中间 GPU 一直是满的
                CUDA_CHECK(cudaStreamSynchronize(nullptr));

                for (int i = 0; i < n; ++i) {
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0[i], ev1[i]));
                    samples[i].push_back(ms / static_cast<float>(kItersPerRound));
                }
            }
            for (int i = 0; i < n; ++i) {
                CUDA_CHECK(cudaEventDestroy(ev0[i]));
                CUDA_CHECK(cudaEventDestroy(ev1[i]));
            }

            // 每个引擎取中位数
            std::vector<float> med(n);
            for (int i = 0; i < n; ++i) {
                auto v = samples[i];
                std::sort(v.begin(), v.end());
                med[i] = v[v.size() / 2];
            }

            // 按快慢排序输出
            std::vector<int> order(n);
            for (int i = 0; i < n; ++i) order[i] = i;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return med[a] < med[b]; });

            // ---- 误差棒：逐轮比值 + σ + 95% 置信区间 ----
            //
            // 三步递进，每一步都踩过坑：
            //  ① 用"单引擎样本极差"当误差棒 → 错：里面全是共模温漂（实测 25–35%）
            //  ② 用"逐轮比值的极差"      → 半错：极差是分布宽度的最坏估计，
            //                              12 个样本的极差 ≈ 3.3σ，会把 σ=7% 说成 23%
            //  ③ 正确做法：报告 σ，再报**中位数的标准误差** SE = σ/√n
            //              SE 才是"这个比值准到多少"的答案
            std::vector<float> ratio_med(n, 1.0f), ratio_sigma(n, 0.0f), ratio_se(n, 0.0f);
            {
                const auto& base = samples[0];
                for (int i = 0; i < n; ++i) {
                    std::vector<float> rr;
                    rr.reserve(kRounds);
                    for (int r = 0; r < kRounds; ++r) {
                        if (base[r] > 1e-6f) rr.push_back(samples[i][r] / base[r]);
                    }
                    if (rr.size() < 2) continue;

                    double mean = 0.0;
                    for (float v : rr) mean += v;
                    mean /= static_cast<double>(rr.size());

                    double var = 0.0;
                    for (float v : rr) var += (v - mean) * (v - mean);
                    var /= static_cast<double>(rr.size() - 1);

                    const double sd = std::sqrt(var);
                    std::sort(rr.begin(), rr.end());
                    ratio_med[i] = rr[rr.size() / 2];
                    ratio_sigma[i] = (mean > 1e-9)
                                         ? static_cast<float>(100.0 * sd / mean) : 0.0f;
                    ratio_se[i] = (mean > 1e-9)
                                      ? static_cast<float>(100.0 * sd / mean
                                                           / std::sqrt(static_cast<double>(rr.size())))
                                      : 0.0f;
                }
            }

            std::printf("\n  %-30s %9s %9s %8s %10s\n",
                        "引擎", "推理(ms)", "比值", "σ", "95%CI");
            std::printf("  %s\n", std::string(74, '-').c_str());

            float worst_ci = 0.0f;
            for (int k = 0; k < n; ++k) {
                const int i = order[k];
                if (i != 0) worst_ci = std::max(worst_ci, 1.96f * ratio_se[i]);
                std::printf("  %-30s %9.4f %8.3f× %7.1f%% %9.1f%%\n",
                            labels[i].c_str(), med[i], ratio_med[i],
                            ratio_sigma[i], 1.96f * ratio_se[i]);
            }
            std::printf("\n  全部引擎共享同一段温度曲线（每轮轮转起点），温漂被配对约掉。\n");
            std::printf("  σ            = 逐轮比值（t_i/t_0）的标准差，反映单块抖动\n");
            std::printf("  95%%CI        = 1.96·σ/√n，这才是「比值准到多少」的答案\n");
            std::printf("  → 本次测量：比值的不确定度 ≤ ±%.1f%%\n", worst_ci);
            std::printf("  → 两个引擎的比值差小于这个数时，不能断定谁更快。\n");
            if (worst_ci > 5.0f) {
                std::printf("  ⚠️ ±%.1f%% 偏大：只能分辨 >%.0f%% 的差异。"
                            "想更准就加大 kItersPerRound。\n",
                            worst_ci, worst_ci * 2);
            }
            if (args.null_test) {
                std::printf("\n  ★ 本次含 null test（同一个引擎对象测两次）。\n");
                std::printf("    那一项的比值应该在 1.000× 附近；偏离多少就是方法的残余误差。\n");
            }

            if (d_big) CUDA_CHECK(cudaFree(d_big));
        }

        // ===============================================================
        // W8：显存账本
        //
        // 车端的显存预算是硬的（Orin 整机共享几十 GB，还要跑好几个模型）。
        // 所以必须能说清"每一 MB 花在哪"，而不是只知道"engine 多大"。
        // ===============================================================
        {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            auto mb = [](size_t b) { return static_cast<double>(b) / 1024.0 / 1024.0; };

            const double b_img = static_cast<double>(src_w) * src_h * 3;
            const double b_lb = static_cast<double>(kSize) * kSize * 3;
            const double b_chw = static_cast<double>(in_bytes);
            const double b_out = static_cast<double>(out_info.bytes);
            const double b_ms = 3.0 * sizeof(float) * 2;
            const double own = b_img + b_lb + b_chw + b_out + b_ms;

            // 模型文件大小（只有 .engine 才等于显存里的权重占用）
            //
            // ⚠️ 必须把 --engine2 加载的所有引擎都算进来：
            //    做 N 路对比时进程里同时驻留多个引擎，只算主引擎会严重低估。
            auto fileMb = [](const std::string& p) -> double {
                std::ifstream f(p, std::ios::binary | std::ios::ate);
                return f ? static_cast<double>(f.tellg()) / 1048576.0 : 0.0;
            };
            auto isEngine = [](const std::string& p) {
                return p.size() > 7 && p.compare(p.size() - 7, 7, ".engine") == 0;
            };

            const double engine_mb = fileMb(args.model);
            double extra_mb = 0.0;
            int extra_cnt = 0;
            for (const auto& m : args.extra_models) {
                if (isEngine(m)) { extra_mb += fileMb(m); ++extra_cnt; }
            }
            const bool all_engines = isEngine(args.model) && extra_cnt == static_cast<int>(args.extra_models.size());

            std::printf("\n[W8] 显存账本\n");
            std::printf("  %-26s %10.2f MB   源图缓冲 %dx%d\n",
                        "d_img（原图）", b_img / 1048576.0, src_w, src_h);
            std::printf("  %-26s %10.2f MB   letterbox %dx%d\n",
                        "d_lb（放大图）", b_lb / 1048576.0, kSize, kSize);
            std::printf("  %-26s %10.2f MB   输入张量 CHW\n",
                        "d_chw（模型输入）", b_chw / 1048576.0);
            std::printf("  %-26s %10.2f MB   输出张量\n",
                        "d_out（模型输出）", b_out / 1048576.0);
            std::printf("  %-26s %10.2f MB\n", "mean/std（2 个小数组）", b_ms / 1048576.0);
            std::printf("  %-26s %10.2f MB\n", "我们自己占的合计", own / 1048576.0);
            if (all_engines) {
                std::printf("  %-26s %10.2f MB   ← 权重 + workspace（主引擎）\n",
                            "TensorRT engine", engine_mb);
                if (extra_cnt > 0) {
                    std::printf("  %-26s %10.2f MB   %d 个对比引擎（--engine2）\n",
                                "  对比引擎合计", extra_mb, extra_cnt);
                }
                // 每个 engine 至少一个 context，scratch 是每 context 一份
                const double n_ctx = 1.0 + extra_cnt;
                std::printf("  %-26s %10.2f MB   %.0f 个 context（含对比引擎）\n",
                            "context scratch 另计", 14.13 * n_ctx, n_ctx);
                std::printf("  %-26s %10.2f MB\n", "本进程总计（估）",
                            own / 1048576.0 + engine_mb + extra_mb + 14.13 * n_ctx);
                std::printf("  %-26s %10s   （trtexec --verbose 可测得）\n",
                            "", "↑ 14.13 MB/个");
            } else {
                std::printf("  %-26s %10.2f MB   （ONNX，engine 在内存里）\n",
                            "模型文件", engine_mb);
            }
            std::printf("  %-26s %10.2f / %.2f MB   （整卡）\n",
                        "空闲 / 总显存", mb(free_b), mb(total_b));
            if (args.streams > 1) {
                const double per = static_cast<double>(in_bytes) + out_info.bytes;
                std::printf("  %-26s %10.2f MB   %d 路 × (输入+输出)\n",
                            "多流额外占用", per * args.streams / 1048576.0, args.streams);
            }
            std::printf("\n  读法：\n");
            std::printf("    · 我们自己的 buffer 只有几 MB，大头是 engine\n");
            std::printf("    · engine 里除权重外还有 workspace（编译期 tactic 需要）\n");
            std::printf("    · 想压 workspace：config->setMemoryPoolLimit(kWORKSPACE, N)\n");
            std::printf("      代价是 TRT 只能挑需要 ≤N 的 tactic，可能变慢\n");
        }

        // ===============================================================
        // CUDA Graph（--cuda-graph）
        //
        // 目标：把 TRT 每次 enqueueV3 内部提交的上百个 kernel 变成"一次提交"。
        // 我们量化过 launch 开销：融合后仍有 100+ kernel × ~2 µs ≈ 0.1–0.2 ms。
        //
        // 关键约束（写错就捕获失败）：
        //  ① 必须在**非默认流**上捕获 —— TrtEngine 的 stream() 是 cudaStreamCreate 建的 ✔
        //  ② 捕获期间**不能有任何同步** —— 所以用 enqueueDevice() 而不是 inferDevice()
        //  ③ 图绑定指针和形状 —— d_chw / d_out 一直不变 ✔
        // ===============================================================
        if (args.cuda_graph) {
            std::printf("\n[CUDA Graph] 单独对比（主流水线已经走 Graph 了）\n");

            if (!pipe_graph.valid()) {
                std::printf("  ⚠️ 图未捕获成功，跳过对比\n");
            } else {
                constexpr int kRoundsG = 8;
                constexpr int kItersG = 200;
                std::vector<float> t_direct(kRoundsG, 0.0f), t_graph(kRoundsG, 0.0f);

                auto runDirect = [&] {
                    engine.enqueueDevice(d_chw, d_out);
                    CUDA_CHECK(cudaStreamSynchronize(engine.stream()));
                };
                auto runGraph = [&] {
                    pipe_graph.launch(engine.stream());
                    CUDA_CHECK(cudaStreamSynchronize(engine.stream()));
                };

                runDirect();   // 预热
                runGraph();

                // ---- ★ 必须验证输出，否则"快"可能只是"没算完" ----
                //
                // 陷阱：如果捕获的图是空的，d_out 会保留上一次的值，
                //       直接比较两边会"相等"。所以先把它填成 NaN（0xFF）
                //       再各跑一次——图要是没干活，d_out 还是 NaN，一眼就露。
                {
                    std::vector<float> h_ref(out_info.bytes / sizeof(float));
                    std::vector<float> h_graph(out_info.bytes / sizeof(float));

                    CUDA_CHECK(cudaMemset(d_out, 0xFF, out_info.bytes));   // NaN 位模式
                    runDirect();
                    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_out, out_info.bytes,
                                          cudaMemcpyDeviceToHost));

                    CUDA_CHECK(cudaMemset(d_out, 0xFF, out_info.bytes));
                    runGraph();
                    CUDA_CHECK(cudaMemcpy(h_graph.data(), d_out, out_info.bytes,
                                          cudaMemcpyDeviceToHost));

                    float max_abs = 0.0f;
                    size_t nan_cnt = 0;
                    for (size_t i = 0; i < h_ref.size(); ++i) {
                        if (std::isnan(h_graph[i])) ++nan_cnt;
                        max_abs = std::max(max_abs, std::fabs(h_ref[i] - h_graph[i]));
                    }
                    std::printf("\n  输出校验（两边都先把 d_out 填成 NaN 再跑）\n");
                    std::printf("    直连 vs Graph 的 max|abs| = %.3e\n", max_abs);
                    std::printf("    Graph 输出里残留的 NaN   = %zu / %zu\n",
                                nan_cnt, h_ref.size());
                    if (nan_cnt > 0) {
                        std::printf("    ❌ 图**没有执行完整计算**（输出还是 NaN）\n");
                        std::printf("       → 下面那个比值是假的，不能当真\n");
                    } else if (max_abs < 1e-3f) {
                        std::printf("    ✅ 输出一致 → Graph 确实做了同样的计算\n");
                    } else {
                        std::printf("    ⚠️ 输出有差异（%.3e）→ Graph 可能少跑了部分层\n",
                                    max_abs);
                    }
                }

                for (int r = 0; r < kRoundsG; ++r) {
                    for (int k = 0; k < 2; ++k) {
                        const bool use_graph = ((r + k) % 2 == 0);   // 轮转起点
                        const auto t0 = std::chrono::steady_clock::now();
                        if (use_graph) {
                            for (int t = 0; t < kItersG; ++t) runGraph();
                        } else {
                            for (int t = 0; t < kItersG; ++t) runDirect();
                        }
                        const auto t1 = std::chrono::steady_clock::now();
                        const float ms =
                            std::chrono::duration<float, std::milli>(t1 - t0).count()
                            / static_cast<float>(kItersG);
                        (use_graph ? t_graph : t_direct)[r] = ms;
                    }
                }

                std::vector<float> rr;
                rr.reserve(kRoundsG);
                for (int r = 0; r < kRoundsG; ++r) {
                    if (t_direct[r] > 1e-6f) rr.push_back(t_graph[r] / t_direct[r]);
                }
                const Stat st = summarize(rr);
                const Stat d_st = summarize(t_direct);
                const Stat g_st = summarize(t_graph);

                std::printf("  直连 enqueue   : %8.4f ms\n", d_st.med);
                std::printf("  Graph 重放     : %8.4f ms\n", g_st.med);
                std::printf("  比值 graph/直连: %.3f ± %.1f%%（95%%CI）\n", st.med, st.ci_pct);
                if (std::fabs(1.0f - st.med) * 100.0f < st.ci_pct) {
                    std::printf("  → 差异小于 CI，**Graph 没有可测出的收益**\n");
                } else if (st.med < 1.0f) {
                    std::printf("  → Graph 更快，省 %.4f ms（%.1f%%）\n",
                                d_st.med - g_st.med,
                                100.0f * (d_st.med - g_st.med) / std::max(d_st.med, 1e-6f));
                } else {
                    std::printf("  → Graph 更慢（重放本身有固定开销）\n");
                }
                std::printf("  注：图绑定 d_chw/d_out 的地址与形状；换 batch/尺寸必须重新捕获。\n");
            }
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
