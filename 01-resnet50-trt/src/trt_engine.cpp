// ---------------------------------------------------------------------------
// trt_engine.cpp —— TrtEngine 的实现
//
// 关键点（对应 P1 W5 学的五个对象）：
//   Builder  → 造引擎的工厂
//   Network  → 从 ONNX 解析出的图
//   Profile  → 动态 shape 的范围（min/opt/max）
//   Engine   → 编译产物（可序列化到磁盘）
//   Context  → 执行上下文（一个 context 不能多线程同时 enqueue）
// ---------------------------------------------------------------------------
#include "trt_engine.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <NvOnnxParser.h>

#include "cuda_utils.h"

namespace trt {

// ---------------------------------------------------------------------------
// 日志器：TRT 的所有消息都从这里出来。生产环境建议降到 kWARN。
// ---------------------------------------------------------------------------
namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

Logger& logger() {
    static Logger instance;
    return instance;
}

size_t dtypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}

size_t numElements(const nvinfer1::Dims& d) {
    if (d.nbDims <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) return 0;   // 动态维度
        n *= static_cast<size_t>(d.d[i]);
    }
    return n;
}

nvinfer1::Dims toDims(const std::vector<int64_t>& v) {
    nvinfer1::Dims d{};
    d.nbDims = static_cast<int32_t>(v.size());
    for (size_t i = 0; i < v.size(); ++i) d.d[i] = v[i];
    return d;
}

std::string dimsToStr(const nvinfer1::Dims& d) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) os << ", ";
        os << d.d[i];
    }
    os << "]";
    return os.str();
}

}  // namespace

// ---------------------------------------------------------------------------
TrtEngine::TrtEngine(const std::string& path,
                     const std::vector<std::pair<std::string, std::vector<int64_t>>>& opt_shapes,
                     int max_batch, bool use_fp16, bool allow_tf32) {
    const bool is_onnx = (path.size() > 5 && path.substr(path.size() - 5) == ".onnx");
    if (is_onnx) {
        parseOnnx(path, opt_shapes, max_batch, use_fp16, allow_tf32);
    } else {
        loadEngine(path);
    }
    buildRuntimeObjects();
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

TrtEngine::~TrtEngine() {
    // 按「后创建先释放」逆序：显存 → stream → context → engine → runtime
    for (auto& kv : device_buffers_) {
        if (kv.second) cudaFree(kv.second);
    }
    device_buffers_.clear();
    if (stream_) cudaStreamDestroy(stream_);
    // 其余由 TrtPtr 自动 destroy
}

// ---------------------------------------------------------------------------
// 从 ONNX 构建
// ---------------------------------------------------------------------------
void TrtEngine::parseOnnx(
    const std::string& onnx_path,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& opt_shapes,
    int max_batch, bool use_fp16, bool allow_tf32) {

    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));

    // TRT 10+ 默认就是 explicit batch；TRT 8 需要显式传 flag
    uint32_t flags = 0U;
#if NV_TENSORRT_MAJOR < 10
    flags |= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));

    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger()));
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::ostringstream os;
        os << "解析 ONNX 失败: " << onnx_path << "\n";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            os << "  " << parser->getError(i)->desc() << "\n";
        }
        throw std::runtime_error(os.str());
    }
    std::cout << "[TrtEngine] 解析成功: " << network->getNbLayers() << " 层, "
              << network->getNbInputs() << " 输入, "
              << network->getNbOutputs() << " 输出\n";

    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // FP16：TRT 11 去掉了 kFP16 标志，这里用版本宏兜住
#if NV_TENSORRT_MAJOR < 11
    if (use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TrtEngine] 精度: FP16\n";
    }
#else
    (void)use_fp16;
    std::cout << "[TrtEngine] 精度: 由 ONNX 的 dtype 决定（TRT 11 强类型）\n";
#endif

    // ---- TF32 ----
    // NVIDIA 从 Ampere 起默认开启 TF32：FP32 卷积用 10 位尾数累加（和 FP16 相同），
    // 速度接近 FP16、误差约 5e-4。做 < 1e-4 的精度验收时必须关掉。
    if (allow_tf32) {
        std::cout << "[TrtEngine] TF32: 开启（默认，速度快，相对误差 ~5e-4）\n";
    } else {
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        std::cout << "[TrtEngine] TF32: 已关闭（严格 FP32，慢但误差 ~1e-6）\n";
    }

    // ---- 动态 shape 必须建 profile ----
    //
    // ★ 所有权（有官方出处，不是猜的）：
    //   NvInfer.h:11672-11673 对 createOptimizationProfile() 的注释原文：
    //     "The builder retains ownership of the created optimization profile and
    //      returns a raw pointer, i.e. the users must not attempt to delete the
    //      returned pointer."
    //   即：**builder 拥有它**，调用方绝对不能 delete。
    //
    //   而且 IOptimizationProfile 的析构函数是 protected，编译器也会拦住你：
    //     error: 'virtual nvinfer1::IOptimizationProfile::~IOptimizationProfile()'
    //            is protected within this context
    //
    //   所以这里刻意用裸指针、且不 delete —— 用 unique_ptr 编译不过。
    //
    // ★ 生命周期约束：profile 活不过 builder。
    //   本函数里 builder 是 TrtPtr，和 profile 同一作用域，profile 用完即弃，安全。
    //   若把 profile 存成类成员、而 builder 是局部变量，就会拿到悬空指针。
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) throw std::runtime_error("createOptimizationProfile 返回空");
    for (const auto& kv : opt_shapes) {
        const std::string& name = kv.first;
        std::vector<int64_t> opt = kv.second;

        std::vector<int64_t> mn = opt, mx = opt;
        // 约定：opt[0] 是 batch 维，给它动态范围 [1, max_batch]
        if (!mn.empty()) {
            mn[0] = 1;
            mx[0] = max_batch;
        }
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, toDims(mn));
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, toDims(opt));
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, toDims(mx));
        std::cout << "[TrtEngine] profile " << name << ": min=" << dimsToStr(toDims(mn))
                  << " opt=" << dimsToStr(toDims(opt))
                  << " max=" << dimsToStr(toDims(mx)) << "\n";
    }
    // 成功后所有权归 config，不要再碰 profile
    if (config->addOptimizationProfile(profile) < 0) {
        throw std::runtime_error("addOptimizationProfile 失败（检查输入名是否匹配）");
    }

    std::cout << "[TrtEngine] 编译中（最慢的一步）...\n";
    TrtPtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("buildSerializedNetwork 失败");

    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine_) throw std::runtime_error("反序列化失败");
    std::cout << "[TrtEngine] 编译完成，引擎 " << serialized->size() / 1024 << " KB\n";
}

// ---------------------------------------------------------------------------
void TrtEngine::loadEngine(const std::string& engine_path) {
    std::ifstream in(engine_path, std::ios::binary);
    if (!in) throw std::runtime_error("打不开引擎文件: " + engine_path);
    std::vector<char> data((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());

    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) throw std::runtime_error("引擎反序列化失败（版本或 GPU 不匹配？）");
    std::cout << "[TrtEngine] 从文件加载: " << engine_path
              << " (" << data.size() / 1024 << " KB)\n";
}

// ---------------------------------------------------------------------------
void TrtEngine::buildRuntimeObjects() {
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("createExecutionContext 失败");

    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        TensorInfo info;
        // TRT 11 改名：getTensorName -> getIOTensorName，getTensorMode -> getTensorIOMode
        info.name = engine_->getIOTensorName(i);
        info.dims = engine_->getTensorShape(info.name.c_str());
        info.dtype = engine_->getTensorDataType(info.name.c_str());
        info.is_input = (engine_->getTensorIOMode(info.name.c_str()) ==
                         nvinfer1::TensorIOMode::kINPUT);
        info.bytes = numElements(info.dims) * dtypeSize(info.dtype);
        (info.is_input ? inputs_ : outputs_).push_back(info);
    }
}

// ---------------------------------------------------------------------------
void TrtEngine::setInputShape(const std::string& name, const std::vector<int64_t>& shape) {
    if (!context_->setInputShape(name.c_str(), toDims(shape))) {
        throw std::runtime_error("setInputShape 失败: " + name);
    }
}

std::vector<int64_t> TrtEngine::shapeOf(const std::string& name) const {
    nvinfer1::Dims d = context_->getTensorShape(name.c_str());
    std::vector<int64_t> out;
    for (int i = 0; i < d.nbDims; ++i) out.push_back(d.d[i]);
    return out;
}

// ---------------------------------------------------------------------------
void TrtEngine::infer(const std::vector<float>& input, std::vector<float>& output) {
    const TensorInfo& in = inputs_.at(0);
    const TensorInfo& out = outputs_.at(0);

    // 1) 根据当前输入形状，算出输出形状
    const std::vector<int64_t> in_shape = shapeOf(in.name);

    // ★ 动态 shape 的引擎，在 setInputShape() 之前 getTensorShape 会返回 -1。
    //   如果直接拿去乘，-1 会被当成 size_t 变成天文数字
    //   （例如 2^64-150528），报错信息完全看不出原因。所以这里显式拦一下。
    for (int64_t v : in_shape) {
        if (v < 0) {
            std::ostringstream os;
            os << "输入 \"" << in.name << "\" 的形状还没设置（当前含 -1）\n"
               << "  这是动态 shape 引擎的必经步骤，请先调用：\n"
               << "    engine.setInputShape(\"" << in.name
               << "\", {batch, 3, height, width});";
            throw std::runtime_error(os.str());
        }
    }

    size_t in_elems = 1;
    for (int64_t v : in_shape) in_elems *= static_cast<size_t>(v);
    if (input.size() != in_elems) {
        std::ostringstream os;
        os << "输入元素数不匹配: 期望 " << in_elems << " 实际 " << input.size();
        throw std::runtime_error(os.str());
    }

    // 2) 分配/复用显存（形状变了就重新分配）
    const std::vector<int64_t> out_shape = shapeOf(out.name);
    size_t out_elems = 1;
    for (int64_t v : out_shape) out_elems *= static_cast<size_t>(v);

    const size_t need_in = in_elems * dtypeSize(in.dtype);
    const size_t need_out = out_elems * dtypeSize(out.dtype);
    if (!device_buffers_.count(in.name) || input_bytes_ != need_in) {
        if (device_buffers_.count(in.name)) cudaFree(device_buffers_[in.name]);
        CUDA_CHECK(cudaMalloc(&device_buffers_[in.name], need_in));
        input_bytes_ = need_in;
    }
    if (!device_buffers_.count(out.name) || output_bytes_ != need_out) {
        if (device_buffers_.count(out.name)) cudaFree(device_buffers_[out.name]);
        CUDA_CHECK(cudaMalloc(&device_buffers_[out.name], need_out));
        output_bytes_ = need_out;
    }

    // 3) H2D → kernel → D2H
    CUDA_CHECK(cudaMemcpyAsync(device_buffers_[in.name], input.data(), need_in,
                               cudaMemcpyHostToDevice, stream_));

    context_->setTensorAddress(in.name.c_str(), device_buffers_[in.name]);
    context_->setTensorAddress(out.name.c_str(), device_buffers_[out.name]);
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 失败");
    }

    output.resize(out_elems);
    CUDA_CHECK(cudaMemcpyAsync(output.data(), device_buffers_[out.name], need_out,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---------------------------------------------------------------------------
// 零拷贝推理：数据全程留在显存
//
// 对比 infer()：
//   infer()        : host → H2D → kernel → D2H → host     （两趟拷贝）
//   inferDevice()  : device ─────────→ kernel ────────→ device（零趟拷贝）
//
// 这里的"零拷贝"指**不跨 PCIe**。kernel 内部当然还要读显存。
// ---------------------------------------------------------------------------
void TrtEngine::inferDevice(const void* d_input, void* d_output) {
    const TensorInfo& in = inputs_.at(0);
    const TensorInfo& out = outputs_.at(0);

    for (int64_t v : shapeOf(in.name)) {
        if (v < 0) {
            std::ostringstream os;
            os << "输入 \"" << in.name << "\" 的形状还没设置（含 -1），"
               << "请先调用 setInputShape()";
            throw std::runtime_error(os.str());
        }
    }

    // const_cast 是必要的：TRT 的 setTensorAddress 要 void*，但它不会写输入缓冲
    context_->setTensorAddress(in.name.c_str(), const_cast<void*>(d_input));
    context_->setTensorAddress(out.name.c_str(), d_output);

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 失败（检查输入形状是否在 profile 范围内）");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---------------------------------------------------------------------------
void TrtEngine::enqueueDevice(const void* d_input, void* d_output) {
    const TensorInfo& in = inputs_.at(0);
    const TensorInfo& out = outputs_.at(0);

    for (int64_t v : shapeOf(in.name)) {
        if (v < 0) {
            std::ostringstream os;
            os << "输入 \"" << in.name << "\" 的形状还没设置（含 -1），"
               << "请先调用 setInputShape()";
            throw std::runtime_error(os.str());
        }
    }

    context_->setTensorAddress(in.name.c_str(), const_cast<void*>(d_input));
    context_->setTensorAddress(out.name.c_str(), d_output);

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 失败（检查输入形状是否在 profile 范围内）");
    }
    // ★ 故意不同步：本函数给 CUDA Graph 捕获用
}

// ---------------------------------------------------------------------------
size_t TrtEngine::inputBytes() const {
    const TensorInfo& in = inputs_.at(0);
    auto shape = shapeOf(in.name);
    size_t n = 1;
    for (int64_t v : shape) {
        if (v < 0) return 0;   // 形状没设
        n *= static_cast<size_t>(v);
    }
    return n * dtypeSize(in.dtype);
}

size_t TrtEngine::outputBytes() const {
    const TensorInfo& out = outputs_.at(0);
    auto shape = shapeOf(out.name);
    size_t n = 1;
    for (int64_t v : shape) {
        if (v < 0) return 0;
        n *= static_cast<size_t>(v);
    }
    return n * dtypeSize(out.dtype);
}

// ---------------------------------------------------------------------------
void TrtEngine::save(const std::string& engine_path) const {
    TrtPtr<nvinfer1::IHostMemory> mem(engine_->serialize());
    if (!mem) throw std::runtime_error("引擎序列化失败");
    std::ofstream out(engine_path, std::ios::binary);
    if (!out) throw std::runtime_error("写不了文件: " + engine_path);
    out.write(static_cast<const char*>(mem->data()), static_cast<std::streamsize>(mem->size()));
    std::cout << "[TrtEngine] 已保存: " << engine_path
              << " (" << mem->size() / 1024 << " KB)\n";
}

// ---------------------------------------------------------------------------
std::string TrtEngine::describe() const {
    std::ostringstream os;
    os << "输入 (" << inputs_.size() << "):\n";
    for (const auto& t : inputs_) {
        os << "  " << t.name << "  " << dimsToStr(t.dims)
           << "  dtype=" << static_cast<int>(t.dtype) << "\n";
    }
    os << "输出 (" << outputs_.size() << "):\n";
    for (const auto& t : outputs_) {
        os << "  " << t.name << "  " << dimsToStr(t.dims)
           << "  dtype=" << static_cast<int>(t.dtype) << "\n";
    }
    return os.str();
}

}  // namespace trt
