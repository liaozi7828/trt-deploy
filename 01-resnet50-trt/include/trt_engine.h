#pragma once
// ---------------------------------------------------------------------------
// trt_engine.h —— TensorRT 引擎的 RAII 封装（项目① 的核心交付物）
//
// 计划要求的方法：
//   build / load / serialize / infer / setInputShape
//
// 设计原则：
//   1. RAII：构造即资源就绪，析构自动释放，不写 close()
//   2. 禁止拷贝（引擎不能被复制），允许移动
//   3. 错误用异常抛（CUDA / TRT 都是 C 风格返回码，封装成异常更好用）
//   4. 所有 resource 都在析构里按「后创建先释放」顺序销毁
// ---------------------------------------------------------------------------
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

namespace trt {

/// 一个张量的描述
struct TensorInfo {
    std::string name;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    bool is_input{false};
    size_t bytes{0};   // dims 相乘 × dtype 大小
};

/// 引擎封装
class TrtEngine {
public:
    /// 从一个 .onnx 或 .engine 文件构造（根据后缀自动判断）
    /// @param opt_shapes  形如 {{"input", {1,3,224,224}}}；动态 shape 必需
    /// @param allow_tf32  是否允许 TF32。NVIDIA 从 Ampere 起**默认开启** TF32：
    ///                    FP32 卷积累加用 10 位尾数（和 FP16 一样），速度快但
    ///                    相对误差 ~5e-4。做精度验收（< 1e-4）必须设为 false。
    explicit TrtEngine(const std::string& path,
                       const std::vector<std::pair<std::string, std::vector<int64_t>>>& opt_shapes = {},
                       int max_batch = 8,
                       bool use_fp16 = true,
                       bool allow_tf32 = true);
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;
    TrtEngine(TrtEngine&&) noexcept = default;
    TrtEngine& operator=(TrtEngine&&) noexcept = default;

    /// 把当前引擎序列化到文件（.engine）
    void save(const std::string& engine_path) const;

    /// 设置输入形状（动态 shape 每次换形状都要调）
    void setInputShape(const std::string& name, const std::vector<int64_t>& shape);

    /// 查询某个张量的形状（输出形状要 setInputShape 之后才有效）
    std::vector<int64_t> shapeOf(const std::string& name) const;

    /// 推理一次。input/output 都是**宿主内存**，内部负责 H2D / D2H。
    /// 适合"数据本来就在 CPU 上"的场景（比如从文件读图后直接推理）。
    void infer(const std::vector<float>& input, std::vector<float>& output);

    /// ★ 零拷贝推理：输入已在显存、输出也留在显存。
    ///
    /// 相比 infer() 省掉两次多余拷贝（前处理已经写进显存，没必要 D2H 再 H2D）：
    ///   batch=1 约省 5%，batch=8 约省 10%
    ///
    /// @param d_input   设备指针，字节数 = inputBytes()
    /// @param d_output  设备指针，字节数 = outputBytes()
    /// 调用前必须已 setInputShape()；调用返回后结果已经在 d_output 里（已同步）。
    void inferDevice(const void* d_input, void* d_output);

    /// 只提交、**不同步**（CUDA Graph 捕获专用）
    ///
    /// 为什么单独开一个：`cudaStreamBeginCapture` 期间**不允许任何同步**，
    /// 而 inferDevice 末尾有 cudaStreamSynchronize —— 那会让捕获直接失败。
    ///
    /// ⚠️ 本函数返回时结果还没算完，调用方自己负责同步。
    void enqueueDevice(const void* d_input, void* d_output);

    /// 引擎内部使用的 stream（非默认流）。
    /// CUDA Graph 必须捕获在非默认流上，所以要把这个流暴露出来。
    cudaStream_t stream() const { return stream_; }

    /// 当前输入/输出占多少字节（setInputShape() 之后才准）
    size_t inputBytes() const;
    size_t outputBytes() const;

    const std::vector<TensorInfo>& inputs()  const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }

    /// 暴露底层 engine（**只读**），用于自建多个 IExecutionContext。
    ///
    /// 用途：多流并发（P3 W7）。
    /// 一个 ICudaEngine 可以创建多个 context，每个 context 有独立的
    /// tensor 地址和形状，但共享同一份权重——这是多流并发的正确姿势。
    ///
    /// ⚠️ 调用方不得 delete 返回的指针；它归本对象所有。
    nvinfer1::ICudaEngine* rawEngine() const { return engine_.get(); }

    /// 引擎元信息（用于 README 的架构图/表格）
    std::string describe() const;

private:
    void parseOnnx(const std::string& onnx_path,
                   const std::vector<std::pair<std::string, std::vector<int64_t>>>& opt_shapes,
                   int max_batch, bool use_fp16, bool allow_tf32);
    void loadEngine(const std::string& engine_path);
    void buildRuntimeObjects();
    void allocBuffers();

    // ---- TRT 对象（用 unique_ptr + 自定义 deleter，保证任何路径都能释放）----
    // TRT 10 用 p->destroy()；TRT 11 移除了 destroy()，改用标准 delete。
    struct TrtDeleter {
        template <typename T>
        void operator()(T* p) const { delete p; }
    };
    template <typename T>
    using TrtPtr = std::unique_ptr<T, TrtDeleter>;

    TrtPtr<nvinfer1::IRuntime>        runtime_;
    TrtPtr<nvinfer1::ICudaEngine>     engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;

    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    // 显存：name -> device ptr
    std::unordered_map<std::string, void*> device_buffers_;
    size_t input_bytes_{0};
    size_t output_bytes_{0};

    cudaStream_t stream_{nullptr};
};

}  // namespace trt
