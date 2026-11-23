# P5 · Orin 实测方案

> **这份文档没有人替你跑过。** 我没有 Orin 硬件，所以这里只有"怎么做"和"预期看到什么"，
> 没有"实测数据"。**任何数字都必须你自己在板子上跑出来才能写进简历。**

---

## 0. 为什么这一步绕不过去

| 简历上写 | 面试官会问 | 没有 Orin 数据会怎样 |
|---|---|---|
| "部署到车端" | 在什么硬件上？多少帧？ | ❌ 答不出来 |
| "延迟 0.78 ms" | 哪块卡？ | ⚠️ "笔记本 5060" → 车端岗位不认 |
| "6 路 30fps" | 实测能跑几路？ | ❌ 没有数据 |

**Orin 是自动驾驶感知部署的"入场券硬件"。** 没跑过，前面的优化再漂亮也会被问穿。

---

## 1. 硬件与软件版本（先确认，别踩版本坑）

| 项 | 建议 | 为什么 |
|---|---|---|
| 板子 | **Jetson AGX Orin 32GB 或 64GB** | 64GB 才有完整的 2048 CUDA core；32GB 是 1792 |
| JetPack | **6.x**（对应 CUDA 12.x / TensorRT 10.x） | JetPack 7 才带 TRT 11；**如果只能拿到 JetPack 6，我们的 TRT 11 代码要改回去** |
| TensorRT | **尽量和开发机一致** | engine 不能跨版本；API 差异要重新适配 |
| 供电模式 | `sudo nvpmodel -m 0`（最大性能） | 默认模式会限频，测出来的数没意义 |

### ⚠️ 第一个大坑：TRT 版本可能不是 11

我们的代码大量依赖 TRT 11 的行为（strong typing、无 calibrator、IPluginV3）。
**如果 Orin 上是 TRT 10**：

- `BuilderFlag::kFP16` **还在** → FP16 可以用 `config->setFlag()`，不需要 polygraphy 转模型
- `IInt8Calibrator` **还在** → INT8 走传统校准器，不需要 QDQ
- **我们的 `enqueueDevice` / `CUDA Graph` 逻辑仍然可用**（`IExecutionContext::enqueueV3` 从 TRT 8.5 就有）

**所以先跑这一行，再决定移植策略**：

```bash
dpkg -l | grep -i tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## 2. 移植清单（按依赖顺序）

| # | 事项 | 命令 / 说明 |
|---|---|---|
| 1 | **engine 必须重编** | engine 绑定 GPU 架构和 TRT 版本。Orin 是 **sm_87**，我们的 5060 是 sm_120 → 直接拷过去加载会失败 |
| 2 | 改 CMake 的 arch | `set(CMAKE_CUDA_ARCHITECTURES "87")`（或 `87;120` 同时支持） |
| 3 | 依赖库 | Orin 上 CUDA/TensorRT 在系统路径，一般不需要 `TRT_ROOT`；`find_package` 可能直接可用 |
| 4 | `g++-14` 那段 | Orin 的 JetPack 6 是 Ubuntu 22.04，**gcc 11**。删掉 `CMAKE_CUDA_HOST_COMPILER` 那几行 |
| 5 | `-U_GNU_SOURCE` | 那是 Ubuntu 26.04 + glibc 2.41 的问题；**Orin 上大概率不需要**，先去掉 |
| 6 | 图片读取 | 我们的 PPM 读取是玩具。真车端要换成 V4L2 / GStreamer / argus |
| 7 | 功耗监控 | `tegrastats` 或 `jtop`，测延迟时同步记录温度和频率 |

### 最快路径（先不追求优雅）

```bash
# 在 Orin 上
sudo apt install -y cmake ninja-build build-essential
git clone <你的仓库> && cd yolov8-trt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

**如果 CMakeLists 里的 TRT_ROOT 逻辑报错**，把 `-DTRT_ROOT=` 留空，让 `find_package(TensorRT)` 走系统路径。

---

## 3. 测量协议（这部分必须严格，否则数字白测）

### 3.1 环境准备

```bash
sudo nvpmodel -m 0                    # 最大性能模式
sudo jetson_clocks                    # 锁定最高频率（关键！）
# 记录初始状态
tegrastats --interval 1000 --logfile tegrastats.log &
```

**`jetson_clocks` 是 Orin 上的 `nvidia-smi -lgc`** —— 它把 CPU/GPU/EMC 频率钉在最高。
**不锁频的话，你会重复我们在笔记本上遇到的 2.4 倍漂移。**

### 3.2 跑什么

```bash
# ① 单张基准（确认引擎能跑、数值对）
./build/bin/yolov8_demo --engine models/yolov8n_fp16.engine \
    --image models/bus.ppm --gpu-decode --cuda-graph

# ② 延迟分位数（2000 帧，6 路预算）
./build/bin/yolov8_service --engine models/yolov8n_fp16.engine \
    --image models/bus.ppm --frames 5000 --cameras 6 --csv orin_lat.csv

# ③ 对照：不做优化的路径（证明优化的价值在 Orin 上依然成立）
./build/bin/yolov8_service --engine models/yolov8n_fp16.engine \
    --image models/bus.ppm --frames 5000 --cameras 6 --cpu-path --no-graph \
    --csv orin_lat_base.csv
```

### 3.3 必须记录的四组数

| 项 | 怎么拿 | 为什么 |
|---|---|---|
| **p50 / p99 / max** | service 自带 | 车端判据 |
| **温度 / 频率** | `tegrastats.log` | 证明没有降频，或量化降频损失 |
| **功耗** | `tegrastats` 里的 `VDD_GPU_SOC` 等 | 车端有功耗墙 |
| **持续 10 分钟的稳态** | 重跑一次 `--frames 20000` | 短测会高估 |

### 3.4 一个容易漏的对照

**同一个 engine，`--no-graph` vs `--cuda-graph` 各跑一次。**

我们预测 CPU 派发的开销在 Orin 上**占比更高**（Orin 的 GPU 更快地完成计算，CPU 提交相对更慢）。
**如果实测 Graph 的收益在 Orin 上超过 69%，那就是一个很有说服力的结论。**

---

## 4. 预期数字（推算，不是实测）

**先纠正我之前一个不严谨的说法**：我说过"Orin 算力约是 5060 的 1/5"，**那是估的，没有依据**。按官方规格粗算：

| | RTX 5060 Laptop | Jetson AGX Orin 64GB | 比值 |
|---|---|---|---|
| CUDA core | 约 3300 | 2048 | 0.62 |
| GPU 频率 | 约 2.0 GHz | 1.3 GHz | 0.65 |
| FP32 算力（粗算） | 约 13 TFLOPS | 约 5.3 TFLOPS | **约 0.4** |
| 显存带宽 | 约 384 GB/s | 204.8 GB/s | **约 0.53** |

**所以 Orin 大概是 2–2.5 倍慢，不是 5 倍。**

**推算（必须在 Orin 上验证）**：

```
单帧 p50:  0.77 ms × 2.5  ≈ 1.9 ms
6 路串行:  1.9 × 6        ≈ 11.5 ms    预算 33.3 ms  →  占 35%
```

**这个推算比之前说的"23 ms / 70%"乐观得多。** 差距来自我之前的 1/5 估错了。

**但有两个会让它变慢的现实因素**：
1. Orin 上 `--gpu-decode` 的 kernel 更小，启动开销占比更高
2. 真车端还要加相机采集（GMSL 解串 + ISP），那几毫秒不在我们的测量里

---

## 5. 交付物清单（做完这些才算 P5 的 Orin 部分完成）

- [ ] `orin_tflops.txt`：TRT/CUDA/JetPack 版本 + `jetson_clocks` 后的频率
- [ ] `orin_lat.csv` + `orin_lat_base.csv`：5000 帧逐帧延迟
- [ ] `tegrastats.log`：同期温度/功耗
- [ ] **一张表：优化前 vs 优化后的 p50/p99/max**
- [ ] **一句话结论**：`在 AGX Orin 上，6 路 30fps 的单帧预算 5.55 ms，实测 p99 = ? ms`
- [ ] README 新增一节 §19「Orin 实测」，把上面这些写进去

---

## 6. 如果暂时拿不到 Orin

**替代方案（按可信度排序）**：

| 方案 | 可信度 | 成本 |
|---|---|---|
| **云上的 Orin**（如一些边缘计算云厂商提供 Jetson 实例） | 高 | 按小时计费，几十元/小时 |
| **Jetson Nano / Xavier NX** 二手板 | 中（架构不同，只能证明"能移植"） | 几百到一千多 |
| 借学校/实验室的板子 | 高 | 靠关系 |
| **只在简历里写"在 RTX 5060 上验证"，不提车端** | 低但诚实 | 0 |

**最后一条是可接受的底线**：写"在 RTX 5060 Laptop 上完成端到端优化"，面试时说明"车端移植方案已准备，缺硬件"。
**但绝不要写"部署到车端"——那句话会被一句"哪块板子？"问穿。**

---

## 7. 一句话

> **Orin 实测不是"跑个分"，是"用锁频 + 分位数 + 温度记录，证明这个系统在最坏情况下也满足 33.3 ms 预算"。**
>
> **而这套测量工具（`yolov8_service`）已经写好了——到板子上只需要改 CUDA arch 和跑三条命令。**
