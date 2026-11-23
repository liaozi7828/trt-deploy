// ---------------------------------------------------------------------------
// decode_nms_test.cpp —— 融合 decode+NMS kernel 的自检
//
// 验证链（这是关键）：
//   我们的 CPU 版 decodeV8 + nms   ←  已经被 verify_nms.py 证明和 ultralytics 逐位一致
//              ↑ 比
//   新的 GPU 融合 kernel
//   所以：kernel 和 CPU 版一致  ⇒  kernel 和 ultralytics 一致
//
// 比法：按 IoU 匹配（不按顺序），因为 NMS 遇到"分数相同的近重复候选"时，
//       保留哪个取决于 tie-break，两边可能不同——那是等优解，不是错误。
//
// 用法：
//   ./build/bin/decode_nms_test --raw dump/out.bin [--conf 0.25] [--iou 0.45] [--bench]
// ---------------------------------------------------------------------------
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "decode_nms_kernel.h"
#include "postprocess.h"

namespace {

/// 读 main.cpp 的 writeTensorBin：int32 ndim + ndim×int32 dims + float32
bool readTensorBin(const std::string& path, std::vector<float>& data,
                   std::vector<int>& dims) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int ndim = 0;
    f.read(reinterpret_cast<char*>(&ndim), 4);
    if (ndim <= 0 || ndim > 8) return false;
    dims.resize(static_cast<size_t>(ndim));
    f.read(reinterpret_cast<char*>(dims.data()), 4 * ndim);
    size_t n = 1;
    for (int d : dims) n *= static_cast<size_t>(d);
    data.resize(n);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 4));
    return static_cast<bool>(f);
}

float iouXYXY(const yolo::Detection& a, const yolo::Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0.0f;
    const float inter = iw * ih;
    const float ua = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float ub = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / std::max(ua + ub - inter, 1e-9f);
}

struct MatchStat {
    int matched{0};
    int only_ref{0};
    int only_test{0};
    float min_iou{1e9f};
    float max_coord{0.0f};
};

/// 按类别 + IoU 贪心匹配（ref = 参考，test = 待测）
MatchStat matchDetections(const std::vector<yolo::Detection>& ref,
                          const std::vector<yolo::Detection>& test) {
    MatchStat s;
    std::vector<char> used(test.size(), 0);
    std::vector<char> ref_used(ref.size(), 0);

    for (size_t i = 0; i < ref.size(); ++i) {
        int best_j = -1;
        float best_u = 0.5f;
        for (size_t j = 0; j < test.size(); ++j) {
            if (used[j]) continue;
            if (ref[i].cls != test[j].cls) continue;
            const float u = iouXYXY(ref[i], test[j]);
            if (u > best_u) { best_u = u; best_j = static_cast<int>(j); }
        }
        if (best_j >= 0) {
            used[static_cast<size_t>(best_j)] = 1;
            ref_used[i] = 1;
            ++s.matched;
            s.min_iou = std::min(s.min_iou, best_u);
            const auto& b = test[static_cast<size_t>(best_j)];
            s.max_coord = std::max({s.max_coord, std::fabs(ref[i].x1 - b.x1),
                                    std::fabs(ref[i].y1 - b.y1),
                                    std::fabs(ref[i].x2 - b.x2),
                                    std::fabs(ref[i].y2 - b.y2)});
        }
    }
    for (size_t i = 0; i < ref.size(); ++i) if (!ref_used[i]) ++s.only_ref;
    for (size_t j = 0; j < test.size(); ++j) if (!used[j]) ++s.only_test;
    return s;
}

/// 跑一遍完整 GPU 融合路径
std::vector<yolo::Detection> runGpu(const float* d_out, int nc, int na,
                                    float conf, float iou,
                                    float* d_cand, int* d_cand_cnt,
                                    float* d_dets, int* d_num_det,
                                    bool* overflow) {
    cudaMemset(d_cand_cnt, 0, sizeof(int));
    yolo::detail::launchGatherCandidates(d_out, nc, na, conf, d_cand, d_cand_cnt);

    int n_cand = 0;
    cudaMemcpy(&n_cand, d_cand_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    if (overflow) *overflow = (n_cand > yolo::detail::kMaxCandidates);
    if (n_cand > yolo::detail::kMaxCandidates) n_cand = yolo::detail::kMaxCandidates;

    std::vector<yolo::Detection> out;
    if (n_cand <= 0) return out;

    cudaMemset(d_num_det, 0, sizeof(int));
    yolo::detail::launchNms(d_cand, n_cand, iou, d_dets, d_num_det);
    cudaDeviceSynchronize();

    int nd = 0;
    cudaMemcpy(&nd, d_num_det, sizeof(int), cudaMemcpyDeviceToHost);
    const int cap = yolo::detail::kMaxDetections;
    if (nd > cap) nd = cap;
    if (nd <= 0) return out;

    std::vector<float> h(static_cast<size_t>(nd) * 6);
    cudaMemcpy(h.data(), d_dets, h.size() * sizeof(float), cudaMemcpyDeviceToHost);

    out.reserve(static_cast<size_t>(nd));
    for (int i = 0; i < nd; ++i) {
        const float* d = h.data() + static_cast<size_t>(i) * 6;
        yolo::Detection det;
        det.x1 = d[0]; det.y1 = d[1]; det.x2 = d[2]; det.y2 = d[3];
        det.score = d[4];
        det.cls = static_cast<int>(d[5]);
        out.push_back(det);
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string raw_path = "dump/out.bin";
    float conf = 0.25f, iou = 0.45f;
    bool bench = false;

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--raw" && i + 1 < argc) raw_path = argv[++i];
        else if (k == "--conf" && i + 1 < argc) conf = std::stof(argv[++i]);
        else if (k == "--iou" && i + 1 < argc) iou = std::stof(argv[++i]);
        else if (k == "--bench") bench = true;
        else if (k == "-h" || k == "--help") {
            std::printf("用法: %s [--raw path] [--conf x] [--iou y] [--bench]\n", argv[0]);
            return 0;
        }
    }

    std::vector<float> raw;
    std::vector<int> dims;
    if (!readTensorBin(raw_path, raw, dims)) {
        std::printf("[FAIL] 读不了 %s\n", raw_path.c_str());
        std::printf("  先跑: ./build/bin/yolov8_demo --engine models/yolov8n_fp16.engine \\\n");
        std::printf("            --image models/bus.ppm --dump-output dump/out.bin\n");
        return 1;
    }
    if (dims.size() != 3) {
        std::printf("[FAIL] 期望 3 维张量，得到 %zu 维\n", dims.size());
        return 1;
    }
    const int nc = dims[1] - 4;
    const int na = dims[2];

    std::printf("=========================================================\n");
    std::printf("decode + NMS 融合 kernel 自检\n");
    std::printf("=========================================================\n");
    std::printf("输入     : %s  [%d, %d, %d]\n", raw_path.c_str(), dims[0], dims[1], dims[2]);
    std::printf("类别/锚点: %d 类, %d 锚点\n", nc, na);
    std::printf("阈值     : conf > %.2f, IoU > %.2f\n", conf, iou);

    // ---------------- ① 参考：CPU 版（已被 verify_nms.py 证明和 ultralytics 一致）----------------
    auto ref = yolo::decodeV8(raw.data(), nc, na, conf);
    std::printf("\n[参考] CPU decode 后 %zu 个候选\n", ref.size());
    ref = yolo::nms(ref, iou);
    std::printf("[参考] CPU NMS 后 %zu 个框\n", ref.size());

    // ---------------- ② GPU 融合路径 ----------------
    float* d_out = nullptr;
    float* d_cand = nullptr;
    int* d_cand_cnt = nullptr;
    float* d_dets = nullptr;
    int* d_num_det = nullptr;
    cudaMalloc(&d_out, raw.size() * sizeof(float));
    cudaMalloc(&d_cand, static_cast<size_t>(yolo::detail::kMaxCandidates) * 6 * sizeof(float));
    cudaMalloc(&d_cand_cnt, sizeof(int));
    cudaMalloc(&d_dets, static_cast<size_t>(yolo::detail::kMaxDetections) * 6 * sizeof(float));
    cudaMalloc(&d_num_det, sizeof(int));
    cudaMemcpy(d_out, raw.data(), raw.size() * sizeof(float), cudaMemcpyHostToDevice);

    bool overflow = false;
    auto got = runGpu(d_out, nc, na, conf, iou, d_cand, d_cand_cnt,
                      d_dets, d_num_det, &overflow);
    std::printf("[GPU]  融合 kernel 后 %zu 个框\n", got.size());
    if (overflow) std::printf("  ⚠️ 候选数超过 %d，被截断\n",
                              yolo::detail::kMaxCandidates);

    // ---------------- ③ 对比 ----------------
    const MatchStat st = matchDetections(ref, got);
    std::printf("\n[对比] 按 IoU 匹配（忽略顺序）\n");
    std::printf("  匹配上  : %d 对\n", st.matched);
    std::printf("  只有参考: %d 个（GPU 漏检）\n", st.only_ref);
    std::printf("  只有待测: %d 个（GPU 多检）\n", st.only_test);
    if (st.matched > 0) {
        std::printf("  最小 IoU: %.4f\n", st.min_iou);
        std::printf("  最大坐标差: %.4f px\n", st.max_coord);
    }
    const bool ok = (st.only_ref == 0 && st.only_test == 0 && st.min_iou >= 0.95f);
    std::printf("  判定（无漏检 + 无多检 + 最小 IoU ≥ 0.95）: %s\n",
                ok ? "✅ 通过" : "❌ 不通过");

    // ---------------- ④ 耗时 ----------------
    if (bench) {
        // CPU 路径
        std::vector<float> cpu_ms;
        for (int r = 0; r < 5; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i) {
                auto d = yolo::decodeV8(raw.data(), nc, na, conf);
                auto k = yolo::nms(d, iou);
                (void)k;
            }
            const auto t1 = std::chrono::steady_clock::now();
            cpu_ms.push_back(std::chrono::duration<float, std::milli>(t1 - t0).count() / 50.0f);
        }
        std::sort(cpu_ms.begin(), cpu_ms.end());

        // GPU 路径（只算 kernel，不含回传）
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::vector<float> gpu_ms;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0);
            for (int i = 0; i < 50; ++i) {
                cudaMemsetAsync(d_cand_cnt, 0, sizeof(int));
                yolo::detail::launchGatherCandidates(d_out, nc, na, conf,
                                                     d_cand, d_cand_cnt);
                yolo::detail::launchNms(d_cand, 64, iou, d_dets, d_num_det);
            }
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            gpu_ms.push_back(ms / 50.0f);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        std::sort(gpu_ms.begin(), gpu_ms.end());

        std::printf("\n[耗时] 中位数（5 组 × 50 次）\n");
        std::printf("  CPU decode+NMS : %.4f ms\n", cpu_ms[cpu_ms.size() / 2]);
        std::printf("  GPU 融合 kernel: %.4f ms（只算 kernel，不含回传）\n",
                    gpu_ms[gpu_ms.size() / 2]);
    }

    cudaFree(d_out);
    cudaFree(d_cand);
    cudaFree(d_cand_cnt);
    cudaFree(d_dets);
    cudaFree(d_num_det);

    std::printf("\n%s\n", ok ? "结果：✅ 通过" : "结果：❌ 失败");
    return ok ? 0 : 1;
}
