// ---------------------------------------------------------------------------
// cuda_graph.cpp —— CUDA Graph 的捕获/重放（g++ 编译）
// ---------------------------------------------------------------------------
#include "cuda_graph.h"

#include <cstdio>

namespace yolo {

PipelineGraph::~PipelineGraph() { release(); }

void PipelineGraph::release() {
    if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
}

bool PipelineGraph::capture(cudaStream_t stream, const std::function<void()>& fn,
                            cudaStreamCaptureMode mode) {
    release();
    err_ = "";

    cudaError_t st = cudaStreamBeginCapture(stream, mode);
    if (st != cudaSuccess) {
        err_ = "cudaStreamBeginCapture 失败（该流可能正在被占用）";
        cudaGetLastError();
        return false;
    }

    // ★ fn 里可能抛异常：含 plugin 的引擎在捕获期间可能报
    //   cudaErrorStreamCaptureUnsupported（因为它内部要分配/释放显存）。
    //   必须在这里接住，否则异常会穿过 capture() 把程序带崩，
    //   而且流会停在"捕获中"的非法状态。
    bool fn_ok = true;
    try {
        fn();   // 只能"提交"，不能同步
    } catch (...) {
        fn_ok = false;
    }

    cudaGraph_t g = nullptr;
    st = cudaStreamEndCapture(stream, &g);

    if (!fn_ok || st != cudaSuccess || g == nullptr) {
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();     // 清错误标志，避免污染后续调用
        err_ = fn_ok
                   ? "EndCapture 失败（捕获期间出现了同步，或图为空）"
                   : "捕获期间提交失败：引擎内部可能分配/释放显存"
                     "（含 plugin 的引擎会报 cudaErrorStreamCaptureUnsupported）";
        return false;
    }

    graph_ = g;
    st = cudaGraphInstantiate(&exec_, graph_, 0);
    if (st != cudaSuccess) {
        err_ = "cudaGraphInstantiate 失败";
        cudaGetLastError();
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        exec_ = nullptr;
        return false;
    }
    return true;
}

bool PipelineGraph::launch(cudaStream_t stream) {
    if (!exec_) { err_ = "图还没捕获成功"; return false; }
    const cudaError_t st = cudaGraphLaunch(exec_, stream);
    if (st != cudaSuccess) {
        err_ = "cudaGraphLaunch 失败";
        cudaGetLastError();
        return false;
    }
    return true;
}

}  // namespace yolo
