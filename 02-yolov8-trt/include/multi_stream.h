#pragma once
// ---------------------------------------------------------------------------
// multi_stream.h —— P3 W7：多流并发
//
// 核心认知（这是"吞吐"和"延迟"的分水岭）：
//
//   延迟（latency）  = 一次推理要多久       ← 单流优化：降精度、融合、CUDA Graph
//   吞吐（throughput）= 每秒能处理多少张     ← 多流优化：并发执行
//
//   两者不是一回事：
//     4 路并发时，单次延迟可能**变差**（互相抢 SMs），
//     但吞吐通常能涨 1.5–3 倍（GPU 不再有空闲的 SM）。
//
// 正确姿势：**一个 ICudaEngine + N 个 IExecutionContext + N 个 cudaStream**
//   · engine 共享 → 权重只有一份（13.5 MB），不会 ×N
//   · context 独立 → 每个有自己的 tensor 地址、形状、中间状态
//   · stream 独立 → 各自的 H2D / kernel / D2H 可以重叠
//
// ⚠️ 一个 context 不能被多线程同时 enqueue（P1 W5 的验收点）。
//    多流 ≠ 多线程；这里全部在单线程里提交，靠 stream 重叠。
// ---------------------------------------------------------------------------
#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

namespace yolo {

class MultiStreamRunner {
public:
    /// @param engine      共享的引擎（由 TrtEngine 持有，这里不 delete）
    /// @param num_streams 并发路数（通常 2–8）
    /// @param in_name / out_name  张量名（从 TrtEngine::inputs()/outputs() 取）
    /// @param in_bytes / out_bytes 单个样本的字节数
    MultiStreamRunner(nvinfer1::ICudaEngine* engine, int num_streams,
                      const std::string& in_name, const std::string& out_name,
                      size_t in_bytes, size_t out_bytes);

    ~MultiStreamRunner();

    MultiStreamRunner(const MultiStreamRunner&) = delete;
    MultiStreamRunner& operator=(const MultiStreamRunner&) = delete;

    int streams() const { return num_streams_; }

    /// 在第 i 路提交一次推理（异步，不等待）
    /// @param d_input 设备指针（所有流可以共用同一份输入）
    /// @param d_output 设备指针（**必须每路不同**，否则结果互相覆盖）
    void submit(int i, const void* d_input, void* d_output);

    /// 等所有流完成
    void syncAll();

    /// 查询第 i 路的输出缓冲（供调用方读取结果）
    void* outputBuffer(int i) const;

    /// 第 i 路的输入缓冲（如果调用方想自己填数据）
    void* inputBuffer(int i) const;

    /// 把 host 数据拷到第 i 路的输入缓冲（异步）
    void uploadInput(int i, const void* host_data);

private:
    struct Slot {
        nvinfer1::IExecutionContext* ctx{nullptr};
        cudaStream_t stream{nullptr};
        void* d_in{nullptr};
        void* d_out{nullptr};
    };

    nvinfer1::ICudaEngine* engine_{nullptr};
    std::string in_name_;
    std::string out_name_;
    size_t in_bytes_{0};
    size_t out_bytes_{0};
    int num_streams_{0};
    std::vector<Slot> slots_;
};

}  // namespace yolo
