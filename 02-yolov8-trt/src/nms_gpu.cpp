// ---------------------------------------------------------------------------
// nms_gpu.cpp —— GPU NMS 的**主机侧封装**（std::vector 进出，g++ 编译）
//
// 分层原因（和 preprocess.cu / preprocess_cpu.cpp 同理）：
//   nms_kernel.cu  ← 纯 CUDA，带 -U_GNU_SOURCE，不能碰 STL
//   nms_gpu.cpp    ← 这个文件，用 <algorithm>/<vector> 处理业务逻辑
//
// 职责划分：
//   GPU: IoU 矩阵（n² 个独立计算，可以并行）
//   CPU: 排序 + 贪心选择（有严格前后依赖，串行）
// ---------------------------------------------------------------------------
#include "nms_gpu.h"

#include <algorithm>
#include <cstring>

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "nms_kernel.h"   // detail::launchOverlap（纯 CUDA 接口）

namespace yolo {

namespace {

/// 把 detections 打包成 SoA（扁平 float 数组 + 类别数组）
void packBoxes(const std::vector<Detection>& dets,
               std::vector<float>& boxes, std::vector<int>& cls) {
    const int n = static_cast<int>(dets.size());
    boxes.resize(static_cast<size_t>(n) * 4);
    cls.resize(n);
    for (int i = 0; i < n; ++i) {
        boxes[i * 4 + 0] = dets[i].x1;
        boxes[i * 4 + 1] = dets[i].y1;
        boxes[i * 4 + 2] = dets[i].x2;
        boxes[i * 4 + 3] = dets[i].y2;
        cls[i] = dets[i].cls;
    }
}

/// 上传 boxes/cls，返回设备指针（调用方负责 cudaFree）
///
/// ⚠️ 这个结构体持有裸指针 + 有析构函数，所以**必须禁止拷贝**：
///    一旦被按值返回/传递，拷贝出来的副本析构时会 double free。
///    这里用"输出参数"的方式填充，从根本上避免拷贝。
struct DevicePack {
    float* boxes{nullptr};
    int* cls{nullptr};
    int n{0};

    DevicePack() = default;
    DevicePack(const DevicePack&) = delete;             // 禁止拷贝
    DevicePack& operator=(const DevicePack&) = delete;
    ~DevicePack() {
        if (boxes) cudaFree(boxes);
        if (cls) cudaFree(cls);
    }
};

// ---------------------------------------------------------------------------
// 复用显存：cudaMalloc/cudaFree 每次 10–50 µs，绝不能每帧都做
//
// 第一次调用按需扩容，之后一直复用。这是"流水线里不许运行时分配"的基本功
// （车端还会进一步换成自建显存池，见 P3 W8 的 IGpuAllocator）。
// ---------------------------------------------------------------------------
struct NmsBuffers {
    float* boxes{nullptr};
    int* cls{nullptr};
    unsigned char* overlap{nullptr};
    size_t cap_boxes{0};     // 能装多少个框
    size_t cap_overlap{0};   // 能装多少个 overlap 元素
};

NmsBuffers& nmsBuffers() {
    static NmsBuffers b;     // 进程级复用；单线程假设（多线程要每线程一份）
    return b;
}

/// 确保 buffer 够大（够就不动）
void ensureCapacity(size_t n) {
    NmsBuffers& b = nmsBuffers();

    if (b.cap_boxes < n) {
        if (b.boxes) cudaFree(b.boxes);
        if (b.cls) cudaFree(b.cls);
        CUDA_CHECK(cudaMalloc(&b.boxes, n * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b.cls, n * sizeof(int)));
        b.cap_boxes = n;
    }
    if (b.cap_overlap < n * n) {
        if (b.overlap) cudaFree(b.overlap);
        CUDA_CHECK(cudaMalloc(&b.overlap, n * n));
        b.cap_overlap = n * n;
    }
}

void uploadBoxes(const std::vector<Detection>& dets, DevicePack& d) {
    d.n = static_cast<int>(dets.size());
    std::vector<float> hb;
    std::vector<int> hc;
    packBoxes(dets, hb, hc);

    CUDA_CHECK(cudaMalloc(&d.boxes, hb.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.cls, hc.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d.boxes, hb.data(), hb.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.cls, hc.data(), hc.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
}

}  // namespace

// ---------------------------------------------------------------------------
std::vector<Detection> nmsGpu(std::vector<Detection>& dets, float iou_thres) {
    const int n = static_cast<int>(dets.size());
    if (n == 0) return {};

    // ① 排序留 CPU：O(n log n) 远小于 O(n²)，而且 GPU 排序要写代码
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    if (n == 1) return dets;

    // ② 上传（复用缓存 buffer，不再每帧 cudaMalloc）
    ensureCapacity(static_cast<size_t>(n));
    NmsBuffers& b = nmsBuffers();

    std::vector<float> hb;
    std::vector<int> hc;
    packBoxes(dets, hb, hc);
    CUDA_CHECK(cudaMemcpyAsync(b.boxes, hb.data(), hb.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(b.cls, hc.data(), hc.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    // ③ GPU 算 IoU 矩阵（只算，不传）
    detail::launchOverlap(b.boxes, b.cls, b.overlap, n, iou_thres);

    // ④ 下载矩阵
    //
    // ⚠️ 已知局限：这里把 n² 字节整块搬回主机。
    //     n=46   →   2 KB  ≈ 0 µs
    //     n=637  → 405 KB  ≈ 30 µs
    //     n=8400 →  70 MB  ≈ 5 ms   ← 比推理还慢，不可用
    //
    //    正确做法是在 GPU 上完成贪心（多轮 kernel / bitmask），只回传结果。
    //    那是 TensorRT EfficientNMS plugin 的思路，属于 P5 写 plugin 的内容。
    //    本实现的价值是"证明这个设计不可扩展"——见 README 的实验记录。
    static thread_local std::vector<unsigned char> h_overlap;
    const size_t ovl_bytes = static_cast<size_t>(n) * n;
    h_overlap.resize(ovl_bytes);
    CUDA_CHECK(cudaMemcpy(h_overlap.data(), b.overlap, ovl_bytes,
                          cudaMemcpyDeviceToHost));

    // ④ 贪心选择（查表，不再算 IoU）
    std::vector<char> removed(n, 0);
    std::vector<Detection> keep;
    keep.reserve(n);

    for (int i = 0; i < n; ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        const unsigned char* row = h_overlap.data() + static_cast<size_t>(i) * n;
        for (int j = i + 1; j < n; ++j) {
            if (!removed[j] && row[j]) removed[j] = 1;
        }
    }
    return keep;
}

// ---------------------------------------------------------------------------
float benchIouMatrixGpu(const std::vector<Detection>& dets, float iou_thres,
                        int warmup, int iters) {
    const int n = static_cast<int>(dets.size());
    if (n < 2) return 0.0f;

    // 先按分数排序（和 nmsGpu 内部一致）
    std::vector<Detection> sorted = dets;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    DevicePack dp;
    uploadBoxes(sorted, dp);

    unsigned char* d_overlap = nullptr;
    const size_t ovl_bytes = static_cast<size_t>(n) * n;
    CUDA_CHECK(cudaMalloc(&d_overlap, ovl_bytes));

    const float ms = cuda_utils::benchmark([&] {
        detail::launchOverlap(dp.boxes, dp.cls, d_overlap, n, iou_thres);
    }, warmup, iters, 3);

    CUDA_CHECK(cudaFree(d_overlap));
    return ms;
}

}  // namespace yolo
