// ---------------------------------------------------------------------------
// pillar_plugin.cpp —— C4：把点云 scatter 包成 IPluginV3
//
// 为什么这个 plugin 是**正当的**（三条同时成立，缺一不可）：
//   ① TRT 没有等价层        —— 不像 grid_sample 有 IGridSampleLayer（C2 实测 PASSED）
//   ② TRT 也没用内置 plugin 实现 —— 不像 ScatterElements 有 ScatterElements v2（C3 实测 PASSED）
//   ③ **输出规模由数据决定**   —— 有几根 pillar 只有算完才知道
//      TRT 的 plugin 同样无法表达"完全动态的输出形状"（输出形状要用 DimsExprs 从输入推导），
//      所以用标准解法：**固定容量 + count 标量**（EfficientNMS 也是这么做的）
//
// API 事实（从 NvInferPluginBase.h 逐条捞出来的，不是猜的）：
//   · IPluginV3 有两个纯虚：getCapabilityInterface(PluginCapabilityType) 和 clone()
//   · 能力接口是**多重继承**的，靠 getCapabilityInterface 按类型 static_cast 出去
//   · PluginCapabilityType: kCORE=0 / kBUILD=1 / kRUNTIME=2
//   · 所有 getOutputShapes / getOutputDataTypes / enqueue 返回 int32_t，**0 = 成功**
//   · network->addPluginV3(inputs, nbInputs, shapeInputs, nbShapeInputs, plugin) → IPluginV3Layer*
//
// 输出布局（4 个输出）：
//   out0 pillars    [max_pillars][max_pts][num_features]  float
//   out1 coords     [max_pillars][4]                     int32   (batch, z, y, x)
//   out2 counts     [max_pillars]                        int32
//   out3 num_pillars[1]                                  int32   ← 真实数量（runtime 决定）
// ---------------------------------------------------------------------------
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

#include "scatter_kernel.h"

using namespace nvinfer1;

namespace {

constexpr char const* kPLUGIN_NAME = "PillarScatter";
constexpr char const* kPLUGIN_VERSION = "1";
constexpr char const* kPLUGIN_NAMESPACE = "";

}  // namespace

// ---------------------------------------------------------------------------
class PillarScatterPlugin : public IPluginV3,
                            public IPluginV3OneCore,
                            public IPluginV3OneBuild,
                            public IPluginV3OneRuntime {
public:
    PillarScatterPlugin() = default;

    explicit PillarScatterPlugin(bev::PillarParams p) : p_(p) {
        buildFields();
    }

    // ======================= IPluginV3 =======================
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        // 文档要求：addPluginV3 加进去的对象必须对 CORE/BUILD/RUNTIME 三者都返回有效对象
        switch (type) {
            case PluginCapabilityType::kCORE:
                return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityType::kBUILD:
                return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityType::kRUNTIME:
                return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override {
        // 拷贝内部参数，返回一个**完全初始化**的新对象
        auto* c = new PillarScatterPlugin(p_);
        return c;
    }

    // ======================= Core =======================
    AsciiChar const* getPluginName() const noexcept override { return kPLUGIN_NAME; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPLUGIN_NAMESPACE; }

    // ======================= Build =======================
    int32_t getNbOutputs() const noexcept override { return 4; }

    int32_t configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbIn*/,
                            DynamicPluginTensorDesc const* /*out*/,
                            int32_t /*nbOut*/) noexcept override {
        return 0;    // 没有 per-shape 的状态要准备
    }

    int32_t getOutputDataTypes(DataType* outTypes, int32_t nbOutputs,
                               DataType const* /*inTypes*/,
                               int32_t /*nbInputs*/) const noexcept override {
        if (nbOutputs != 4) return 1;
        outTypes[0] = DataType::kFLOAT;    // pillars
        outTypes[1] = DataType::kINT32;    // coords
        outTypes[2] = DataType::kINT32;    // counts
        outTypes[3] = DataType::kINT32;    // num_pillars
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* /*inputs*/, int32_t /*nbInputs*/,
                            DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
                            DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& eb) noexcept override {
        // ★ 关键：4 个输出全部是**常量形状**（由 plugin 属性决定，与输入无关）。
        //   这是"固定容量 + count 标量"方案的直接后果 —— 也是它能被 TRT 接受的原因。
        if (nbOutputs != 4) return 1;
        auto C = [&eb](int v) { return eb.constant(static_cast<int64_t>(v)); };

        outputs[0].nbDims = 3;
        outputs[0].d[0] = C(p_.max_pillars);
        outputs[0].d[1] = C(p_.max_pts_per_pillar);
        outputs[0].d[2] = C(p_.num_features);

        outputs[1].nbDims = 2;
        outputs[1].d[0] = C(p_.max_pillars);
        outputs[1].d[1] = C(4);

        outputs[2].nbDims = 1;
        outputs[2].d[0] = C(p_.max_pillars);

        outputs[3].nbDims = 1;
        outputs[3].d[0] = C(1);
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        // DynamicPluginTensorDesc 的结构是 { PluginTensorDesc desc; Dims min; Dims max; }
        // → format 在 desc 里，不在外层
        if (inOut[pos].desc.format != TensorFormat::kLINEAR) return false;
        if (pos == 0) {                      // 输入：点云 [N][stride]，float
            return inOut[pos].desc.type == DataType::kFLOAT;
        }
        const int32_t o = pos - nbInputs;
        if (o == 0) return inOut[pos].desc.type == DataType::kFLOAT;
        return inOut[pos].desc.type == DataType::kINT32;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbIn*/,
                            DynamicPluginTensorDesc const* /*out*/,
                            int32_t /*nbOut*/) const noexcept override {
        // cell_to_pid 表：grid_x * grid_y 个 int，每次 enqueue 前要 memset 成 -1
        return static_cast<size_t>(p_.grid_x) * p_.grid_y * sizeof(int);
    }

    // ======================= Runtime =======================
    int32_t onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbIn*/,
                          PluginTensorDesc const* /*out*/,
                          int32_t /*nbOut*/) noexcept override {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override {
        const int n_points = static_cast<int>(inputDesc[0].dims.d[0]);
        if (n_points <= 0) return 0;

        int* cell_to_pid = static_cast<int*>(workspace);
        if (cell_to_pid == nullptr) return 2;

        // ① 清空映射表：0xFF 全字节 = -1（kNoPillar）
        const size_t cells = static_cast<size_t>(p_.grid_x) * p_.grid_y;
        const cudaError_t e1 = cudaMemsetAsync(cell_to_pid, 0xFF,
                                               cells * sizeof(int), stream);
        // ② 清零 num_pillars（它是"输出"，但语义上是计数器，必须先置 0）
        const cudaError_t e2 = cudaMemsetAsync(outputs[3], 0, sizeof(int), stream);
        if (e1 != cudaSuccess || e2 != cudaSuccess) return 3;

        bev::launchScatter(static_cast<const float*>(inputs[0]), n_points,
                           cell_to_pid,
                           static_cast<float*>(outputs[0]),
                           static_cast<int*>(outputs[1]),
                           static_cast<int*>(outputs[2]),
                           static_cast<int*>(outputs[3]),
                           p_, stream);
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* /*context*/) noexcept override {
        // 本 plugin 没有 per-context 的资源，直接 clone 即可
        // （不能返回 nullptr —— 那会被当成失败）
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &fields_;
    }

private:
    /// 把属性摊成 PluginField，供 TRT 序列化。
    ///
    /// 两个坑：
    ///   ① PluginField 只存**指针**，所以后面的 vector 一旦扩容，指针就悬空了
    ///      → 这里只在 buildFields() 里赋值一次，之后不再改动
    ///   ② 字段类型必须写对（kINT32 / kFLOAT），用 kUNKNOWN 会被序列化拒绝
    void buildFields() {
        ivals_ = {p_.grid_x, p_.grid_y, p_.max_pillars, p_.max_pts_per_pillar,
                  p_.num_features, p_.point_stride};
        fvals_ = {p_.min_x, p_.min_y, p_.inv_voxel_x, p_.inv_voxel_y};

        static char const* const in[] = {"grid_x", "grid_y", "max_pillars",
                                         "max_pts_per_pillar", "num_features",
                                         "point_stride"};
        static char const* const fn[] = {"min_x", "min_y", "inv_voxel_x", "inv_voxel_y"};

        field_vec_.clear();
        for (size_t i = 0; i < ivals_.size(); ++i) {
            field_vec_.push_back(PluginField{in[i], &ivals_[i],
                                             PluginFieldType::kINT32, 1});
        }
        for (size_t i = 0; i < fvals_.size(); ++i) {
            field_vec_.push_back(PluginField{fn[i], &fvals_[i],
                                             PluginFieldType::kFLOAT32, 1});
        }
        fields_.nbFields = static_cast<int32_t>(field_vec_.size());
        fields_.fields = field_vec_.data();
    }

    bev::PillarParams p_{};
    int point_stride_{4};
    std::vector<int32_t> ivals_;
    std::vector<float> fvals_;
    std::vector<PluginField> field_vec_;
    PluginFieldCollection fields_{};
};

// ---------------------------------------------------------------------------
class PillarScatterCreator : public IPluginCreatorV3One {
public:
    PillarScatterCreator() = default;

    IPluginV3* createPlugin(AsciiChar const* /*name*/, PluginFieldCollection const* fc,
                            TensorRTPhase /*phase*/) noexcept override {
        bev::PillarParams p;
        if (fc != nullptr && fc->fields != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const PluginField& f = fc->fields[i];
                const std::string n(f.name);
                if (n == "grid_x") p.grid_x = *static_cast<const int*>(f.data);
                else if (n == "grid_y") p.grid_y = *static_cast<const int*>(f.data);
                else if (n == "max_pillars") p.max_pillars = *static_cast<const int*>(f.data);
                else if (n == "max_pts_per_pillar") p.max_pts_per_pillar = *static_cast<const int*>(f.data);
                else if (n == "num_features") p.num_features = *static_cast<const int*>(f.data);
                else if (n == "point_stride") p.point_stride = *static_cast<const int*>(f.data);
                else if (n == "min_x") p.min_x = *static_cast<const float*>(f.data);
                else if (n == "min_y") p.min_y = *static_cast<const float*>(f.data);
                else if (n == "inv_voxel_x") p.inv_voxel_x = *static_cast<const float*>(f.data);
                else if (n == "inv_voxel_y") p.inv_voxel_y = *static_cast<const float*>(f.data);
            }
        }
        return new PillarScatterPlugin(p);
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    AsciiChar const* getPluginName() const noexcept override { return kPLUGIN_NAME; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPLUGIN_NAMESPACE; }

private:
    // getFieldNames() 只需要**名字和类型**（data 可以为空），
    // 但类型必须写对，否则 createPlugin 里按类型取数据会错。
    std::vector<PluginField> field_vec_{[] {
        std::vector<PluginField> v;
        char const* const in[] = {"grid_x", "grid_y", "max_pillars",
                                  "max_pts_per_pillar", "num_features", "point_stride"};
        char const* const fn[] = {"min_x", "min_y", "inv_voxel_x", "inv_voxel_y"};
        for (auto n : in) v.push_back(PluginField{n, nullptr, PluginFieldType::kINT32, 1});
        for (auto n : fn) v.push_back(PluginField{n, nullptr, PluginFieldType::kFLOAT32, 1});
        return v;
    }()};
    PluginFieldCollection fields_{static_cast<int32_t>(field_vec_.size()),
                                  field_vec_.data()};
};

// 静态注册：TRT 在加载时会找到这个 creator
REGISTER_TENSORRT_PLUGIN(PillarScatterCreator);
