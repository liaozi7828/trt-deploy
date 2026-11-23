// ---------------------------------------------------------------------------
// multi_stream.cpp —— P3 W7：多流并发（g++ 编译）
//
// 一个 engine + N 个 context + N 个 stream + N 组 buffer。
// 提交时只在 host 侧循环，不创建线程——靠 cudaStream 让 kernel 重叠。
// ---------------------------------------------------------------------------
#include "multi_stream.h"

#include <stdexcept>

#include "cuda_utils.h"

namespace yolo {

MultiStreamRunner::MultiStreamRunner(nvinfer1::ICudaEngine* engine, int num_streams,
                                     const std::string& in_name,
                                     const std::string& out_name,
                                     size_t in_bytes, size_t out_bytes)
    : engine_(engine),
      in_name_(in_name),
      out_name_(out_name),
      in_bytes_(in_bytes),
      out_bytes_(out_bytes),
      num_streams_(num_streams) {
    if (!engine_) throw std::runtime_error("MultiStreamRunner: engine 为空");
    if (num_streams_ < 1) num_streams_ = 1;

    slots_.resize(static_cast<size_t>(num_streams_));

    for (int i = 0; i < num_streams_; ++i) {
        Slot& s = slots_[i];

        // 每个 context 独立的执行状态（tensor 地址、形状）
        s.ctx = engine_->createExecutionContext();
        if (!s.ctx) throw std::runtime_error("createExecutionContext 失败（第 " +
                                            std::to_string(i) + " 路）");

        // 每个 stream 独立的执行队列
        CUDA_CHECK(cudaStreamCreate(&s.stream));

        // 每个 slot 独立的输入/输出缓冲
        //   ★ 输出必须独立：否则后一路的结果会覆盖前一路
        CUDA_CHECK(cudaMalloc(&s.d_in, in_bytes_));
        CUDA_CHECK(cudaMalloc(&s.d_out, out_bytes_));
    }
}

MultiStreamRunner::~MultiStreamRunner() {
    for (auto& s : slots_) {
        if (s.stream) cudaStreamDestroy(s.stream);
        if (s.d_in) cudaFree(s.d_in);
        if (s.d_out) cudaFree(s.d_out);
        if (s.ctx) delete s.ctx;
    }
    // engine_ 不归这里管（由 TrtEngine 持有）
}

void MultiStreamRunner::submit(int i, const void* d_input, void* d_output) {
    if (i < 0 || i >= num_streams_) throw std::runtime_error("submit: 路数越界");
    Slot& s = slots_[i];

    if (!s.ctx->setTensorAddress(in_name_.c_str(),
                                 const_cast<void*>(d_input)))
        throw std::runtime_error("setTensorAddress(input) 失败");
    if (!s.ctx->setTensorAddress(out_name_.c_str(), d_output))
        throw std::runtime_error("setTensorAddress(output) 失败");

    if (!s.ctx->enqueueV3(s.stream))
        throw std::runtime_error("enqueueV3 失败（第 " + std::to_string(i) + " 路）");
}

void MultiStreamRunner::syncAll() {
    for (auto& s : slots_) {
        if (s.stream) CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }
}

void* MultiStreamRunner::outputBuffer(int i) const {
    if (i < 0 || i >= num_streams_) return nullptr;
    return slots_[i].d_out;
}

void* MultiStreamRunner::inputBuffer(int i) const {
    if (i < 0 || i >= num_streams_) return nullptr;
    return slots_[i].d_in;
}

void MultiStreamRunner::uploadInput(int i, const void* host_data) {
    if (i < 0 || i >= num_streams_) throw std::runtime_error("uploadInput: 越界");
    Slot& s = slots_[i];
    CUDA_CHECK(cudaMemcpyAsync(s.d_in, host_data, in_bytes_,
                               cudaMemcpyHostToDevice, s.stream));
}

}  // namespace yolo
