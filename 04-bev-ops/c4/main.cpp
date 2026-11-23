// ---------------------------------------------------------------------------
// main.cpp —— C4 的 driver：把 plugin 塞进 engine，和"直接调 kernel"对拍
//
// 这个 driver 只做一件有说服力的事：
//   ① 直接调 P3 那个已验证的 kernel → **参照答案**
//   ② 用 plugin 建一个只有这一层的 engine → 跑 → 和 ① 比
//   ③ 顺带测两者的延迟差（in-engine 省掉了一次 launch + host 往返）
//
// 为什么参照答案是"直接调 kernel"而不是 PyTorch：
//   因为 plugin 里跑的**就是那个 kernel**。比的是"包成 plugin 之后语义有没有变"，
//   而不是"kernel 算得对不对"（那件事在项目②里用 5 层验证做过了）。
// ---------------------------------------------------------------------------
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

#include "scatter_kernel.h"

using namespace nvinfer1;

namespace {

constexpr char const* kNAME = "PillarScatter";
constexpr char const* kVER = "1";
constexpr char const* kNS = "";

class Logger : public ILogger {
public:
    void log(Severity s, char const* msg) noexcept override {
        if (s == Severity::kERROR || s == Severity::kWARNING)
            std::printf("[TRT] %s\n", msg);
    }
} gLogger;

void check(cudaError_t e, char const* what) {
    if (e != cudaSuccess) {
        std::printf("[FAIL] %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

double nowMs() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch()).count();
}

}  // namespace

int main() {
    // ---------------- 参数（和项目②的测试一致）----------------
    bev::PillarParams p;
    p.grid_x = 200;
    p.grid_y = 200;
    p.min_x = -50.0f;
    p.min_y = -50.0f;
    p.inv_voxel_x = 1.0f / 0.5f;
    p.inv_voxel_y = 1.0f / 0.5f;
    p.max_pillars = 60000;
    p.max_pts_per_pillar = 32;
    p.num_features = 4;
    p.point_stride = 4;

    const int N = 50000;
    const size_t cells = static_cast<size_t>(p.grid_x) * p.grid_y;
    const size_t pillar_elems = static_cast<size_t>(p.max_pillars)
                                * p.max_pts_per_pillar * p.num_features;

    std::printf("=========================================================\n");
    std::printf("C4：点云 scatter 的 IPluginV3\n");
    std::printf("=========================================================\n");
    std::printf("点数 %d  网格 %dx%d  容量 %d 根 / 每根 %d 点\n",
                N, p.grid_x, p.grid_y, p.max_pillars, p.max_pts_per_pillar);

    // ---------------- 输入 ----------------
    std::vector<float> h_pts(static_cast<size_t>(N) * 4);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(-50.0f, 50.0f), uz(-2.0f, 2.0f),
        ui(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_pts[i * 4 + 0] = ux(rng);
        h_pts[i * 4 + 1] = ux(rng);
        h_pts[i * 4 + 2] = uz(rng);
        h_pts[i * 4 + 3] = ui(rng);
    }

    float* d_pts = nullptr;
    check(cudaMalloc(&d_pts, h_pts.size() * sizeof(float)), "malloc pts");
    check(cudaMemcpy(d_pts, h_pts.data(), h_pts.size() * sizeof(float),
                     cudaMemcpyHostToDevice), "cp pts");

    // ---------------- ① 直接调 kernel（参照答案）----------------
    int* d_cell = nullptr;
    float* d_pillars_ref = nullptr;
    int* d_coords_ref = nullptr;
    int* d_counts_ref = nullptr;
    int* d_num_ref = nullptr;
    check(cudaMalloc(&d_cell, cells * sizeof(int)), "malloc cell");
    check(cudaMalloc(&d_pillars_ref, pillar_elems * sizeof(float)), "malloc pil");
    check(cudaMalloc(&d_coords_ref, static_cast<size_t>(p.max_pillars) * 4 * sizeof(int)), "malloc c");
    check(cudaMalloc(&d_counts_ref, static_cast<size_t>(p.max_pillars) * sizeof(int)), "malloc n");
    check(cudaMalloc(&d_num_ref, sizeof(int)), "malloc np");

    auto run_kernel = [&] {
        check(cudaMemset(d_cell, 0xFF, cells * sizeof(int)), "memset cell");
        check(cudaMemset(d_num_ref, 0, sizeof(int)), "memset np");
        check(cudaMemset(d_counts_ref, 0, static_cast<size_t>(p.max_pillars) * sizeof(int)), "memset cnt");
        bev::launchScatter(d_pts, N, d_cell, d_pillars_ref, d_coords_ref,
                           d_counts_ref, d_num_ref, p, nullptr);
        check(cudaDeviceSynchronize(), "sync kernel");
    };
    run_kernel();
    int num_ref = 0;
    check(cudaMemcpy(&num_ref, d_num_ref, sizeof(int), cudaMemcpyDeviceToHost), "cp np");

    std::vector<float> h_pil_ref(pillar_elems);
    std::vector<int> h_coord_ref(static_cast<size_t>(p.max_pillars) * 4);
    std::vector<int> h_cnt_ref(static_cast<size_t>(p.max_pillars));
    check(cudaMemcpy(h_pil_ref.data(), d_pillars_ref, pillar_elems * sizeof(float),
                     cudaMemcpyDeviceToHost), "cp pil");
    check(cudaMemcpy(h_coord_ref.data(), d_coords_ref,
                     h_coord_ref.size() * sizeof(int), cudaMemcpyDeviceToHost), "cp coord");
    check(cudaMemcpy(h_cnt_ref.data(), d_counts_ref,
                     h_cnt_ref.size() * sizeof(int), cudaMemcpyDeviceToHost), "cp cnt");
    std::printf("\n[①] 直接调 kernel: %d 根 pillar\n", num_ref);

    // ---------------- ② 用 plugin 建 engine ----------------
    std::printf("\n[②] 用 plugin 建 engine\n");
    auto* reg = getPluginRegistry();
    auto* ci = reg->getCreator(kNAME, kVER, kNS);
    if (ci == nullptr) {
        std::printf("[FAIL] 插件注册表里找不到 %s —— REGISTER_TENSORRT_PLUGIN 没生效\n", kNAME);
        return 1;
    }
    auto* creator = dynamic_cast<IPluginCreatorV3One*>(ci);
    if (creator == nullptr) {
        std::printf("[FAIL] creator 不是 IPluginCreatorV3One\n");
        return 1;
    }
    std::printf("      creator 找到了\n");

    int32_t i_vals[6] = {p.grid_x, p.grid_y, p.max_pillars,
                         p.max_pts_per_pillar, p.num_features, p.point_stride};
    float f_vals[4] = {p.min_x, p.min_y, p.inv_voxel_x, p.inv_voxel_y};
    char const* i_names[6] = {"grid_x", "grid_y", "max_pillars",
                              "max_pts_per_pillar", "num_features", "point_stride"};
    char const* f_names[4] = {"min_x", "min_y", "inv_voxel_x", "inv_voxel_y"};
    std::vector<PluginField> fields;
    for (int i = 0; i < 6; ++i)
        fields.push_back(PluginField{i_names[i], &i_vals[i], PluginFieldType::kINT32, 1});
    for (int i = 0; i < 4; ++i)
        fields.push_back(PluginField{f_names[i], &f_vals[i], PluginFieldType::kFLOAT32, 1});
    PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};

    auto* builder = createInferBuilder(gLogger);
    // TRT 10+ 已移除 NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    // （explicit batch 变成唯一模式）→ 传 0
    auto* network = builder->createNetworkV2(0U);
    auto* input = network->addInput("points", DataType::kFLOAT,
                                    Dims2(N, p.point_stride));
    if (input == nullptr) { std::printf("[FAIL] addInput\n"); return 1; }

    auto* plugin = creator->createPlugin(kNAME, &fc, TensorRTPhase::kBUILD);
    if (plugin == nullptr) { std::printf("[FAIL] createPlugin\n"); return 1; }

    ITensor* ins[1] = {input};
    // ★ addPluginV3 第 5 个参数是 IPluginV3& （引用），不是指针
    auto* layer = network->addPluginV3(ins, 1, nullptr, 0, *plugin);
    if (layer == nullptr) {
        std::printf("[FAIL] addPluginV3（看上面的 TRT 报错）\n");
        return 1;
    }
    for (int o = 0; o < layer->getNbOutputs(); ++o) {
        char const* nm = (o == 0) ? "pillars" : (o == 1) ? "coords"
                        : (o == 2) ? "counts" : "num_pillars";
        layer->getOutput(o)->setName(nm);
        network->markOutput(*layer->getOutput(o));
    }
    std::printf("      network: %d 层, %d 输出\n",
                network->getNbLayers(), network->getNbOutputs());

    auto* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ull << 30);
    // ★ config 也是 IBuilderConfig& （引用）
    auto* ser = builder->buildSerializedNetwork(*network, *config);
    if (ser == nullptr) { std::printf("[FAIL] 建引擎失败\n"); return 1; }
    std::printf("      engine 建好了: %.1f KB\n", ser->size() / 1024.0);

    auto* runtime = createInferRuntime(gLogger);
    auto* engine = runtime->deserializeCudaEngine(ser->data(), ser->size());
    if (engine == nullptr) { std::printf("[FAIL] 反序列化\n"); return 1; }
    auto* ctx = engine->createExecutionContext();
    // ★ TRT 10+ 移除了 IExecutionContext::getWorkspaceSize() / setWorkspace()：
    //   context 内部按需分配；插件通过 getWorkspaceSize() 声明的需求由 TRT 自动满足，
    //   并在 enqueue() 的 workspace 参数里传进来。应用侧不做 workspace 管理。
    //   （这和 destroy() 被删、参数从指针改引用是同一个设计方向：把所有权收回去）

    // ---------------- 跑 plugin 路径 ----------------
    float* d_pillars_pl = nullptr;
    int* d_coords_pl = nullptr;
    int* d_counts_pl = nullptr;
    int* d_num_pl = nullptr;
    check(cudaMalloc(&d_pillars_pl, pillar_elems * sizeof(float)), "m");
    check(cudaMalloc(&d_coords_pl, static_cast<size_t>(p.max_pillars) * 4 * sizeof(int)), "m");
    check(cudaMalloc(&d_counts_pl, static_cast<size_t>(p.max_pillars) * sizeof(int)), "m");
    check(cudaMalloc(&d_num_pl, sizeof(int)), "m");

    ctx->setTensorAddress("points", d_pts);
    ctx->setTensorAddress("pillars", d_pillars_pl);
    ctx->setTensorAddress("coords", d_coords_pl);
    ctx->setTensorAddress("counts", d_counts_pl);
    ctx->setTensorAddress("num_pillars", d_num_pl);

    auto run_plugin = [&] {
        if (!ctx->enqueueV3(0)) { std::printf("[FAIL] enqueueV3\n"); std::exit(1); }
        check(cudaStreamSynchronize(0), "sync plugin");
    };
    run_plugin();

    int num_pl = 0;
    check(cudaMemcpy(&num_pl, d_num_pl, sizeof(int), cudaMemcpyDeviceToHost), "cp");
    std::vector<float> h_pil_pl(pillar_elems);
    std::vector<int> h_coord_pl(static_cast<size_t>(p.max_pillars) * 4);
    std::vector<int> h_cnt_pl(static_cast<size_t>(p.max_pillars));
    check(cudaMemcpy(h_pil_pl.data(), d_pillars_pl, pillar_elems * sizeof(float),
                     cudaMemcpyDeviceToHost), "cp");
    check(cudaMemcpy(h_coord_pl.data(), d_coords_pl,
                     h_coord_pl.size() * sizeof(int), cudaMemcpyDeviceToHost), "cp");
    check(cudaMemcpy(h_cnt_pl.data(), d_counts_pl,
                     h_cnt_pl.size() * sizeof(int), cudaMemcpyDeviceToHost), "cp");

    // ---------------- ③ 对拍 ----------------
    std::printf("\n[③] plugin 输出 vs 直接调 kernel\n");
    std::printf("  pillar 数: plugin %d  直接 %d  %s\n", num_pl, num_ref,
                (num_pl == num_ref) ? "✅" : "❌");

    // pillar 内的顺序由 atomicAdd 决定，**不保证一致** → 必须按集合比。
    // 这里用"按 (格子→点集合)"的规范化比较：把每个 pillar 的点排序后比。
    auto canon = [&](int np, const std::vector<float>& pil,
                     const std::vector<int>& coord, const std::vector<int>& cnt) {
        std::vector<std::tuple<int, int, std::vector<std::array<float, 4>>>> out;
        for (int i = 0; i < np; ++i) {
            std::vector<std::array<float, 4>> pts;
            const int s = std::min(cnt[i], p.max_pts_per_pillar);
            for (int k = 0; k < s; ++k) {
                const float* src = pil.data()
                    + (static_cast<size_t>(i) * p.max_pts_per_pillar + k) * p.num_features;
                pts.push_back({src[0], src[1], src[2], src[3]});
            }
            std::sort(pts.begin(), pts.end());
            out.emplace_back(coord[i * 4 + 2], coord[i * 4 + 3], std::move(pts));
        }
        std::sort(out.begin(), out.end(),
                  [](auto const& a, auto const& b) {
                      return std::tie(std::get<0>(a), std::get<1>(a))
                           < std::tie(std::get<0>(b), std::get<1>(b));
                  });
        return out;
    };
    auto ca = canon(num_pl, h_pil_pl, h_coord_pl, h_cnt_pl);
    auto cb = canon(num_ref, h_pil_ref, h_coord_ref, h_cnt_ref);
    bool same = (ca.size() == cb.size());
    long long total_pts = 0;
    if (same) {
        for (size_t i = 0; i < ca.size(); ++i) {
            if (std::get<0>(ca[i]) != std::get<0>(cb[i]) ||
                std::get<1>(ca[i]) != std::get<1>(cb[i]) ||
                std::get<2>(ca[i]) != std::get<2>(cb[i])) { same = false; break; }
            total_pts += static_cast<long long>(std::get<2>(ca[i]).size());
        }
    }
    std::printf("  按集合比（排序后逐元素）: %s，共 %lld 个点\n",
                same ? "✅ 完全一致" : "❌ 不一致", total_pts);

    // ---------------- ④ 延迟 ----------------
    const int iters = 50;
    for (int i = 0; i < 5; ++i) { run_kernel(); run_plugin(); }
    double t0 = nowMs();
    for (int i = 0; i < iters; ++i) run_kernel();
    const double t_k = (nowMs() - t0) / iters;
    t0 = nowMs();
    for (int i = 0; i < iters; ++i) run_plugin();
    const double t_p = (nowMs() - t0) / iters;
    std::printf("\n[④] 延迟（%d 次，块末一次 sync）\n", iters);
    std::printf("  直接调 kernel : %7.3f ms\n", t_k);
    std::printf("  plugin in engine: %7.3f ms   （差 %+.3f ms）\n", t_p, t_p - t_k);

    std::printf("\n%s\n", (same && num_pl == num_ref)
                               ? "结果：✅ 通过" : "结果：❌ 失败");

    check(cudaFree(d_pts), "free");
    check(cudaFree(d_cell), "free"); check(cudaFree(d_pillars_ref), "free");
    check(cudaFree(d_coords_ref), "free"); check(cudaFree(d_counts_ref), "free");
    check(cudaFree(d_num_ref), "free"); check(cudaFree(d_pillars_pl), "free");
    check(cudaFree(d_coords_pl), "free"); check(cudaFree(d_counts_pl), "free");
    check(cudaFree(d_num_pl), "free");
    delete ctx; delete engine; delete ser; delete runtime;
    delete config; delete network; delete builder;
    return 0;
}
