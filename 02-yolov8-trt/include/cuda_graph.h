#pragma once
// ---------------------------------------------------------------------------
// cuda_graph.h —— CUDA Graph：把"每次提交上百个 kernel"变成"提交一次"
//
// 为什么对 YOLOv8 有效：
//   TRT 融合后仍有 100+ 个 kernel，每个 kernel 的 launch 约 2 µs
//   （CPU 组装参数 + 驱动校验 + GPU 分配 block）。
//   => 固定开销 0.1–0.2 ms，占端到端 5–8%。
//   CUDA Graph 把这些 kernel 的依赖关系和参数**录下来**，
//   重放时只有一次提交，GPU 自己按图执行。
//
// 为什么必须在**非默认流**上捕获：
//   cudaStreamBeginCapture 不支持 legacy default stream。
//   TrtEngine 内部用 cudaStreamCreate 建的流，正好满足。
//
// 为什么必须"不同步"地捕获：
//   捕获期间任何 cudaStreamSynchronize / cudaEventSynchronize 都会让
//   EndCapture 失败。所以 TrtEngine 里专门加了 enqueueDevice()。
//
// ⚠️ 图绑定**指针和形状**：
//   捕获时用的 d_input / d_output 必须一直是同一块显存。
//   换 batch、换图片尺寸、换缓冲地址 => 必须重新捕获。
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include <functional>

namespace yolo {

class PipelineGraph {
public:
    PipelineGraph() = default;
    ~PipelineGraph();

    PipelineGraph(const PipelineGraph&) = delete;
    PipelineGraph& operator=(const PipelineGraph&) = delete;

    /// 捕获 fn 里提交到 stream 上的所有 kernel/操作
    /// @return 成功返回 true；失败时 err() 给出原因
    /// 捕获 fn 里提交到 stream 上的所有 kernel/操作
    ///
    /// @param mode 捕获模式。`ThreadLocal` 最严格；某些 plugin 在 enqueue 期间
    ///             会分配/释放显存（cudaErrorStreamCaptureUnsupported），
    ///             这时可以试 `Relaxed`——但它仍然不允许同步。
    /// @return 成功 true；失败 false 且 err() 说明原因，**不会抛异常**
    bool capture(cudaStream_t stream, const std::function<void()>& fn,
                 cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal);

    /// 重放（异步，调用方负责同步）
    bool launch(cudaStream_t stream);

    bool valid() const { return exec_ != nullptr; }
    const char* err() const { return err_; }

private:
    void release();

    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t exec_{nullptr};
    const char* err_{""};
};

}  // namespace yolo
