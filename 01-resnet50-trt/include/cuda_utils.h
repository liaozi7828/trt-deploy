#pragma once
// ---------------------------------------------------------------------------
// cuda_utils.h —— CUDA 错误检查 + 计时 + 设备信息
//
// ⚠️ 这个头文件会被 .cu 文件 include，而 .cu 编译时带着 -U_GNU_SOURCE
//    （为了绕开 glibc 2.41 与 CUDA 12.8 的 cospi 声明冲突）。
//    带 -U_GNU_SOURCE 时，<string> / <stdexcept> 这类头会因 <cwchar>
//    缺声明而编译失败。
//    所以本文件**只用 C 风格头**：cstdio / cstdlib；不用 std::string、不抛异常。
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 错误检查：打印后直接退出（.cu 里不能用异常）
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s  ->  %s\n",            \
                         __FILE__, __LINE__, #call, cudaGetErrorString(_e));    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                     \
    do {                                                                        \
        cudaError_t _e = cudaGetLastError();                                    \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d  kernel launch failed: %s\n", \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace cuda_utils {

struct DeviceInfo {
    const char* name;      // 用 const char* 而不是 std::string，见文件头说明
    int sm_count;
    int major;
    int minor;
    double peak_bw_gbs;
};

inline DeviceInfo queryDevice() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    DeviceInfo info;
    info.name = prop.name;
    info.sm_count = prop.multiProcessorCount;
    info.major = prop.major;
    info.minor = prop.minor;
    info.peak_bw_gbs =
        2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) / 1.0e9;
    return info;
}

/// 基于 cudaEvent 的计时器
class Timer {
public:
    Timer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~Timer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

/// 让 GPU 时钟升到稳态。笔记本 GPU 在 1.4–2.1 GHz 之间跳，
/// 不预热的话第一次测量会偏慢/偏快，A/B 对比直接失真。
inline void warmUpGpu(int rounds = 200) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // 申请一小块显存反复拷贝，制造持续负载把 clocks 顶上去
    size_t n = static_cast<size_t>(prop.multiProcessorCount) * 1024 * 1024;
    void* a = nullptr;
    void* b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, n));
    CUDA_CHECK(cudaMalloc(&b, n));
    cudaStream_t s = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s));
    for (int i = 0; i < rounds; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(a, b, n, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(b, a, n, cudaMemcpyDeviceToDevice, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
    cudaFree(a);
    cudaFree(b);
}

/// 测量耗时：warmup + iters × repeats 轮，取**中位数**
///
/// 为什么不用平均值：
///   - 笔记本 GPU 随时可能降频，个别轮次会明显偏慢
///   - 一次"慢轮"就能把平均值拉偏 10% 以上
///   - 中位数对离群值不敏感
///
/// @param repeats 重复轮数（1–16），每轮内部跑 iters 次后取一次总时间
template <typename Fn>
inline float benchmark(Fn&& fn, int warmup = 10, int iters = 50, int repeats = 5) {
    if (repeats < 1) repeats = 1;
    if (repeats > 16) repeats = 16;

    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    float samples[16];
    for (int r = 0; r < repeats; ++r) {
        Timer t;
        t.start();
        for (int i = 0; i < iters; ++i) fn();
        samples[r] = t.stop() / static_cast<float>(iters);
    }

    // 插入排序（最多 16 个元素，且不用 STL —— 这个头会被 .cu 包含）
    for (int i = 1; i < repeats; ++i) {
        float key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > key) {
            samples[j + 1] = samples[j];
            --j;
        }
        samples[j + 1] = key;
    }
    return samples[repeats / 2];
}

/// 交错测量 A 和 B，取各自中位数
///
/// 为什么需要它：
///   先完整测 A、再完整测 B，中间隔了几十毫秒 —— 笔记本 GPU 的时钟
///   可能已经变了，于是"后测的那个"天然吃亏，结论随机反号。
///   交错执行（A,B,A,B,...）能让两者承受同样的时钟漂移，只留下真实差异。
template <typename Fa, typename Fb>
inline void benchmarkPair(Fa&& fa, Fb&& fb,
                          int warmup, int iters, int repeats,
                          float& out_a, float& out_b) {
    if (repeats < 1) repeats = 1;
    if (repeats > 16) repeats = 16;

    for (int i = 0; i < warmup; ++i) { fa(); fb(); }
    CUDA_CHECK(cudaDeviceSynchronize());

    float sa[16];
    float sb[16];
    for (int r = 0; r < repeats; ++r) {
        {
            Timer t;
            t.start();
            for (int i = 0; i < iters; ++i) fa();
            sa[r] = t.stop() / static_cast<float>(iters);
        }
        {
            Timer t;
            t.start();
            for (int i = 0; i < iters; ++i) fb();
            sb[r] = t.stop() / static_cast<float>(iters);
        }
    }

    // 各自排序取中位数
    for (int i = 1; i < repeats; ++i) {
        float k = sa[i];
        int j = i - 1;
        while (j >= 0 && sa[j] > k) { sa[j + 1] = sa[j]; --j; }
        sa[j + 1] = k;
    }
    for (int i = 1; i < repeats; ++i) {
        float k = sb[i];
        int j = i - 1;
        while (j >= 0 && sb[j] > k) { sb[j + 1] = sb[j]; --j; }
        sb[j + 1] = k;
    }
    out_a = sa[repeats / 2];
    out_b = sb[repeats / 2];
}

}  // namespace cuda_utils
