# 项目①：ResNet50 TensorRT 部署

> **一句话**：把 PyTorch 的 ResNet50 编译成 TensorRT 引擎，用 C++ 封装成可复用的类，
> 前处理搬到 GPU，并用实测数据量化"精度 ↔ 性能"的权衡。

**验收标准**（来自《推理部署学习计划 V2》P2）：

- [x] `TrtEngine` 封装（build / load / serialize / infer / setInputShape）+ RAII
- [x] 前后处理 CPU → CUDA
- [x] **延迟表**：FP32 / TF32 / FP16 × batch 1/4/8
- [x] **精度对齐**：与 PyTorch 的 `max|abs diff| < 1e-4`

---

## 一、结果速览

### 延迟（RTX 5060 Laptop / sm_120，batch 1，零拷贝路径）

| 精度 | engine 大小 | infer | 吞吐 | 相对 FP32 严格 |
|---|---|---|---|---|
| **FP32 严格**（`--no-tf32`） | 101119 KB | **2.465 ms** | 402 img/s | 1.00× |
| **TF32**（TensorRT 默认） | 100225 KB | **1.736 ms** | 568 img/s | **1.42×** |
| **FP16** | 50642 KB | **1.132 ms** | 860 img/s | **2.18×** |

### 精度（与 PyTorch 对比，同一份输入）

| 精度 | `max abs` | `mean abs` | cos 相似度 | top-5 一致 |
|---|---|---|---|---|
| **FP32 严格** | **2.62e-06** | 3.81e-07 | 1.0000001 | ✅ |
| **TF32** | 2.70e-03 | — | 0.9999999 | ✅ |
| **FP16** | 6.13e-03 | 7.74e-04 | 0.9999993 | ✅ |

> **"快 2.18 倍"必须和"精度从 2.6e-6 掉到 6.1e-3"一起说。**
> 只报速度不报精度，等于没说。

---

## 二、架构与数据流

```
PyTorch 模型（Python）
    │  scripts/export_resnet50.py
    ▼
resnet50.onnx  (98 MB)          ←── 框架中立
    │  ① TrtEngine(path=".onnx")  → IBuilder 编译
    │  ② polygraphy convert --fp-to-fp16  → 权重变 FP16
    ▼
resnet50.engine  (50–100 MB)    ←── 绑死 GPU / TRT / CUDA 版本
    │  ③ TrtEngine(path=".engine") → 反序列化（毫秒级）
    ▼
resnet50_demo（C++ 可执行文件）
```

运行时流水线（几乎全在 GPU 上，CPU 只负责下指令）：

```
图像 uint8 (HWC)
   │  resizeBilinear          ← GPU kernel
   ▼
float (HWC, 224×224×3)
   │  hwcToChwNormalize*      ← GPU kernel（含 /255 + ImageNet 归一化）
   ▼
float (CHW) ──► enqueueV3 ──► logits (1000)
                   ↑ 内部启动约 150 个 kernel
```

**零拷贝路径**：前处理结果直接在显存里喂给 `enqueueV3`，中间不跨 PCIe。

---

## 三、环境

| 组件 | 版本 |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop 8 GB（sm_120, 26 SM, 384 GB/s） |
| CUDA | 12.8.93 |
| TensorRT | 11.4.0.106（x86_64, cuda-12.9 tar 包） |
| 编译器 | g++-14（CUDA 宿主）+ g++-15（纯 C++） |
| OS | Ubuntu 26.04 (WSL2) |
| PyTorch | 2.11.0+cu128 |

**必要环境变量**（`~/.bashrc`）：

```bash
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14 -Xcompiler -U_GNU_SOURCE"
export TRT_ROOT=$HOME/tools/TensorRT-11.4.0.106
export LD_LIBRARY_PATH=$TRT_ROOT/lib:$LD_LIBRARY_PATH
```

> `-U_GNU_SOURCE` 是为了绕开 **glibc 2.41 与 CUDA 12.8 的 `cospi` 声明冲突**。
> 代价是 `.cu` 里**不能用 `<string>` / `<vector>`**（`<cwchar>` 会缺声明），
> 所以 CPU 参考实现被拆到 `preprocess_cpu.cpp`（g++ 编译）。

---

## 四、快速开始

```bash
cd ~/projects/01-resnet50-trt

# 1. 导出 ONNX（一次，需要 torch venv）
~/.venvs/torch/bin/python scripts/export_resnet50.py --weights DEFAULT

# 2. 造 FP16 模型（可选）
~/.venvs/deploy/bin/polygraphy convert models/resnet50.onnx \
    --fp-to-fp16 -o models/resnet50_fp16.onnx

# 3. 配置 + 编译
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTRT_ROOT="$TRT_ROOT"
cmake --build build

# 4. 编译三个引擎
./build/bin/resnet50_demo --onnx models/resnet50.onnx \
    --save-engine models/resnet50_fp32strict.engine --no-tf32
./build/bin/resnet50_demo --onnx models/resnet50.onnx \
    --save-engine models/resnet50.engine
./build/bin/resnet50_demo --onnx models/resnet50_fp16.onnx \
    --save-engine models/resnet50_fp16.engine

# 5. 跑
./build/bin/resnet50_demo --engine models/resnet50_fp16.engine --batch 1
```

**精度对齐**：

```bash
mkdir -p dump
./build/bin/resnet50_demo --engine models/resnet50_fp32strict.engine \
    --dump-input dump/in.bin --dump-output dump/out_fp32.bin
./build/bin/resnet50_demo --engine models/resnet50_fp16.engine \
    --dump-output dump/out_fp16.bin
~/.venvs/torch/bin/python scripts/compare_precision.py \
    --input dump/in.bin --fp32 dump/out_fp32.bin --fp16 dump/out_fp16.bin
```

---

## 五、延迟表（完整）

`inferDevice()` 零拷贝路径，单位 ms；括号内为**每图成本**。

| 精度 | batch 1 | batch 4 | batch 8 |
|---|---|---|---|
| **FP32 严格** | 2.465 (2.465) | 5.522 (1.381) | 10.583 (1.323) |
| **TF32** | 1.736 (1.736) | 3.891 (0.973) | 7.154 (0.894) |
| **FP16** | 1.132 (1.132) | 1.631 (0.408) | **2.693 (0.337)** |

吞吐（images/s）：

| 精度 | batch 1 | batch 4 | batch 8 |
|---|---|---|---|
| FP32 严格 | 401.5 | 711.6 | 741.4 |
| TF32 | 568.0 | 996.0 | 1084.7 |
| **FP16** | 859.8 | 2302.9 | **2749.2** |

### 三个观察

**① batch 越大，FP16 的优势越明显**

| batch | FP16 相对 FP32 严格的加速 |
|---|---|
| 1 | 2.18× |
| 4 | 3.39× |
| 8 | **3.93×** |

batch=1 时 GPU 没喂饱（launch 开销 + 访存主导），tensor core 闲着；
batch=8 时 FP16 的 tensor core 吞吐才真正发挥。

**② FP32 严格 / TF32 在 batch 8 已经饱和**

FP32 严格：batch 1→8，工作量 ×8，吞吐只涨 **1.85 倍**。
说明瓶颈不在算力，在**显存带宽**（FP32 的权重与激活都是 FP16 的两倍）。

**③ 单次延迟线性增长，吞吐次线性增长**

FP16 batch 1→8：延迟 2.38 倍，吞吐 3.20 倍。
**没有免费的并发**——追求吞吐必然牺牲单次延迟。

---

## 六、优化记录

**每一项都有 A/B 数字，包括被证伪的那项。**

| # | 优化 | 预期 | 实测 | 结论 |
|---|---|---|---|---|
| 1 | **pinned memory**（`cudaMallocHost`） | 拷贝带宽 2× | H2D 900 KB = 0.066 ms（≈13.5 GB/s，已达 PCIe 实测上限） | ✅ 保留 |
| 2 | **零拷贝推理**（`inferDevice`） | 省两趟 PCIe | FP32: **10.5%**；FP16: **14.2%** | ✅ 保留 |
| 3 | **向量化 normalize**（float4 写） | 写带宽 4× | **±3%，在噪声内** | ❌ **建议回退** |
| 4 | **FP16** | 1.5–2× | batch1 **2.18×**，batch8 **3.93×** | ✅ 保留（须验精度） |

### 为什么"向量化"失败了

```
朴素版：32 个线程各写 4 字节，地址连续
        → 硬件自动合并成 128 字节事务，本来就是满带宽
向量化：一个线程写 16 字节，线程数降 4 倍
        → 事务数不变，还损失并行度
```

**优化前必须先证明"瓶颈真的在那里"。** 这次假设（写带宽是瓶颈）本身就是错的。

### 测量方法本身也踩过坑

| 问题 | 症状 | 解法 |
|---|---|---|
| GPU 时钟漂移 | 同一份代码两次结论反号（+46% / −35%） | `warmUpGpu()` 顶满时钟 |
| 测量顺序偏差 | 后测的那个吃亏 | `benchmarkPair()` **交错测量** |
| 离群值 | 一次降频拉偏平均值 | 取**中位数**（9 轮） |
| 迭代太少 | 2 次 kernel launch 的开销淹没差异 | 前处理跑 500 次 × 9 轮 |

---

## 七、踩坑记录

### ① 归一化少除了 255（最严重）

```cpp
// 错
dst = (v - mean) / std;              // v ∈ [0, 255]
// 对
dst = (v / 255.0f - mean) / std;     // ImageNet 的 mean/std 是给 [0,1] 的
```

**症状**：输入范围 `[-1.08, 1128.7]`（应为 `[-2.5, 2.5]`），logits 从 ±5 变成 ±500，
FP16 甚至溢出成 `nan`。

**为什么四道检查都没抓到**：

| 检查 | 结果 | 为什么 |
|---|---|---|
| A vs B（GPU 内部） | ✅ 一致 | **两个都错** |
| letterbox GPU vs CPU | ✅ 一致 | **两个都错** |
| logits "有数值" | ✅ | 500 也是个数 |
| **PyTorch vs TRT** | ❌ **2.5e-1** | **有独立基准** |

> **自洽性检查只能发现"不一致"，永远发现不了"一致地错"。**

### ② "FP32" 其实是 TF32

TensorRT 在 Ampere+ 上**默认开启 TF32**：FP32 卷积用 10 位尾数累加。

```
默认引擎：max abs = 2.70e-03   ← 远超 1e-4 验收标准
--no-tf32：max abs = 2.62e-06   ← 合格
```

**代价**：TF32 引擎 1.736 ms → 严格 FP32 引擎 2.465 ms（慢 42%）。

### ③ TensorRT 11 的 C++ API 大改

| TRT 10 写法 | TRT 11 |
|---|---|
| `p->destroy()` | **`delete p`** |
| `getTensorName(i)` | `getIOTensorName(i)` |
| `getTensorMode(name)` | `getTensorIOMode(name)` |
| `BuilderFlag::kFP16` | **已移除** |
| `trtexec --fp16` | **已移除**，改用 `polygraphy convert --fp-to-fp16` |

**还有一处所有权陷阱**（`NvInfer.h:11672` 原文）：

> "The builder retains ownership of the created optimization profile and returns
> a raw pointer, i.e. the users must not attempt to delete the returned pointer."

`IOptimizationProfile` 的析构函数是 `protected`，编译器也会拦住你。

### ④ CUDA 12.8 与 glibc 2.41 不兼容

```
error: exception specification is incompatible ... "cospi"
```

**解法**：`-Xcompiler -U_GNU_SOURCE`。
**副作用**：`.cu` 不能用 `<string>` / `<vector>` → CPU 参考实现拆到 `.cpp`。

### ⑤ 主机指针传给 kernel

```cpp
const float mean[3] = {0.485f, ...};       // 栈上
preprocess::hwcToChwNormalize(..., mean);  // ← illegal memory access
```

**必须先 `cudaMalloc` + `cudaMemcpy` 到显存。**

### ⑥ 动态 shape 忘了 setInputShape

```
[EXCEPTION] 输入元素数不匹配: 期望 18446744073709401088 实际 150528
```

`2^64 - 150528` = `-150528` 当无符号数。**`-1` 乘进 `size_t` 的经典灾难。**

### ⑦ polygraphy 参数陷阱

| 参数 | 作用 |
|---|---|
| `--fp-to-fp16` | **转换模型权重**（这才是要的） |
| `--fp16` | TRT 构建选项，对 `convert` 无效，**静默忽略** |

**判据**：转完 ONNX 大小必须**砍半**（98 MB → 49 MB）。没变就是没转。

### ⑧ WSL 里用不了 GPU 调试工具

| 工具 | 结果 |
|---|---|
| `compute-sanitizer` | `Failed to initialize WDDM debugger interface` |
| `nsys` | `prof.sqlite does not contain CUDA kernel data` |

**WSL 的 WDDM 驱动拿不到 GPU 内部数据。** 只能测端到端，测不了纯 kernel 时间。

---

## 八、目录结构

```
resnet50-trt/
├── CMakeLists.txt
├── include/
│   ├── trt_engine.h        # TrtEngine：构造/加载/序列化/推理/动态形状
│   ├── preprocess.h        # 4 个 kernel + 3 个 CPU 参考实现
│   └── cuda_utils.h        # CUDA_CHECK / DeviceInfo / Timer / benchmarkPair
├── src/
│   ├── trt_engine.cpp      # 核心实现（约 400 行）
│   ├── preprocess.cu       # device code（4 个 kernel）
│   ├── preprocess_cpu.cpp  # CPU 基准（g++ 编译）
│   └── main.cpp            # 流水线 + A/B 对比 + 导出
├── scripts/
│   ├── export_resnet50.py        # PyTorch → ONNX
│   ├── compare_precision.py      # PyTorch vs TRT 精度对齐（引擎等价性）
│   └── compare_preprocess.py     # 我的前处理 vs torchvision（算法正确性）
└── models/                 # onnx / engine 产物
```

---

## 九、已知限制：前处理**未**对齐官方流水线

这一节是**主动交底**——不是"还没做"，是"做了对比、量化了差距、决定暂不改"。

### 官方 vs 本项目

```python
# torchvision.models.ResNet50_Weights.DEFAULT.transforms()
T.Resize(256, interpolation=BILINEAR, antialias=True)   # ← 先缩到短边 256
T.CenterCrop(224)                                        # ← 居中裁 224
T.ToTensor()                                             # ← /255
T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
```

```
本项目：resizeBilinear(640×480 → 224×224) → /255 → Normalize
        ↑ 无抗锯齿        ↑ 无 CenterCrop
```

**两处不一致：缺抗锯齿、缺 CenterCrop。**

### 实测差距（`scripts/compare_preprocess.py`）

用 640×480 **随机噪声**图（抗锯齿的最坏情况），对比 torchvision 的四种配置：

| 变体 | max abs diff | 折算灰阶 | 说明 |
|---|---|---|---|
| `antialias=True, float` | **2.326** | **136** | ❌ 完全不同 |
| `antialias=True, uint8` | 2.331 | 136 | ❌ |
| **`antialias=False, float`** | **3.189e-04** | **0.019** | ✅ **本项目的实现** |
| `antialias=False, uint8` | 1.631e-02 | 0.93 | ⚠️ uint8 量化差 |

> 折算方式：`diff × std(0.229) × 255 = 灰阶`
> 3.19e-04 → **0.019 个灰阶**（肉眼不可见）；2.33 → **136 个灰阶**（天壤之别）

### 三个结论

**① 双线性公式本身是正确的**

最接近的是 `antialias=False + float 域 resize`，差 **3.19e-04**。
这点差异来自**源坐标计算的浮点舍入不同**：

```cpp
// 本项目：先乘后除
float fx = (x + 0.5f) * sw / dw - 0.5f;
// PyTorch：先算 scale 再乘
float fx = ((float)sw / dw) * (x + 0.5f) - 0.5f;
```

数学等价、浮点不等价 —— 当 `fx` 接近整数时取整可能跳到相邻像素。

**② 真正的差距是抗锯齿：2.33（136 个灰阶）**

2.86 倍缩小下，不做低通滤波 = 点采样 = 丢弃高频 = 摩尔纹。
**随机噪声图是这个差异的最坏情况**（全图都是高频）；真实照片的差异会小得多，但依然存在。

**③ uint8 域 resize 会引入 0.93 个灰阶的系统偏差**

PIL 路径在 uint8 域做 resize（中间量化回整数），本项目在 float 域做。
单张图看是量化误差，**批量时是系统性偏移**。

### 为什么现在不改

| 理由 | 说明 |
|---|---|
| 项目① 的验收标准不含此项 | 验收的是"前后处理有 CUDA 版"和"引擎等价性 < 1e-4"，两条都过 |
| **没有真实图片，改动无法验证** | 噪声图上的 2.33 不代表真实图；没有标签就测不出"top-1 掉几个点" |
| 抗锯齿 kernel 成本高 | 三角滤波 + 支撑区随缩放比变化 + 必须做成可分离两遍，约 50 行 |

### 生产部署时该怎么做

1. 先有**金标准**：Python 侧照抄官方 transforms，dump 出参考输入
2. 给 C++ 加 `--load-input`，让引擎验证**永远使用金标准输入**（与前处理解耦）
3. 手写 CUDA 版，逐元素对齐到 **1e-4**
4. 最后才做融合（resize + normalize + CHW 合成一个 kernel，省 2.6 倍访存）

> **顺序是先有正确答案、再写实现。本项目走的是反过来的路，所以多花了几天。**

---

## 十、待办

| 项 | 预期收益 | 状态 |
|---|---|---|
| **前处理对齐官方（抗锯齿 + CenterCrop）** | 可复现官方精度 | **已知差距，见第九节** |
| FP16 + CUDA Graph | 5–15% | 未做（需固定地址） |
| INT8 量化 | 2–3× | P3（需校准集） |
| 多流并发 | 吞吐 1.5–2× | 未做（车端多摄像头场景） |
| 真实图片输入 | 让 top-5 变得有意义 | 现在用合成噪声图 |
| 前处理融合（resize+norm+CHW 一个 kernel） | 访存降 2.6× | 现在仍是 2 个 kernel |
| batch=8 的精度对齐 | — | 目前只验了 batch=1 |
| 向量化 kernel 的最终判定 | — | 需 nsys 测纯 kernel（WSL 不支持） |

---

## 十一、验证体系（三层，缺一不可）

这次最大的方法论收获：**三层验证查的是三种不同的错误，不能互相替代。**

| 层 | 和谁比 | 能证明什么 | 本案结果 |
|---|---|---|---|
| **① 算法正确性** | torchvision / OpenCV | 我写的实现对不对 | 差 2.33（抗锯齿） |
| **② 端到端等价性** | PyTorch 模型 | 引擎数值等不等价 | **2.62e-06** ✅ |
| **③ 回归监控** | 自己的 CPU 版 | 改动有没有引入新 bug | 3.05e-05（FMA 舍入） |

**踩过的坑正好证明了这一点**：

> `/255` 缺失这个 bug **活了整整四轮检查**——
> A vs B 一致（两个都错）、CPU vs GPU 一致（两个都错）、logits 有数值（500 也是数）。
> **只有第 ② 层（有独立基准）把它揪出来。**

而第 ① 层又反过来发现：第 ② 层通过 ≠ 前处理正确。
**因为 ② 的两边喂的是同一份（可能错的）输入。**

---

## 十二、一句话总结

> **ONNX 是"可移植的中间表示"，engine 是"绑死这台卡的机器码"。**
> **每换一次精度（FP32 → TF32 → FP16），都要重新回答：精度掉多少，值不值。**
>
> **这个项目的产出不是"能跑"，是那两张表——它们让"值不值"有了数字。**
> **以及第九节那份交底：它让"哪里还不能用"也有了数字。**
