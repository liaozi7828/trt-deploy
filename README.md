# TensorRT 部署实验台

[![CI](https://github.com/liaozi7828/trt-deploy/actions/workflows/ci.yml/badge.svg)](https://github.com/liaozi7828/trt-deploy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 四个项目，一条主线：**在 TensorRT 上把"模型能跑"变成"知道它为什么这么快、以及哪里不该优化"。**
>
> 每个结论都有可复算的数字；被推翻的判断也都留在原地，没有删。

**环境**：Ubuntu (WSL2) · CUDA 12.8 · TensorRT 11.4 · RTX 5060 Laptop 8 GB (sm_120, 26 SM)
**规模说明**：模型都很小（ResNet50 / YOLOv8n / 40 M 的 GPT / 3 层 Deformable Attention）——
**目的是看清单个瓶颈，不是刷 SOTA 吞吐。**

---

## 目录

| # | 项目 | 一句话结论 | 详细 |
|---|---|---|---|
| 01 | **ResNet50 分类部署** | 用 CUDA 重写前处理 + 三层精度验证体系；FP16 吞吐 **3.93×** | [README](01-resnet50-trt/README.md) |
| 02 | **YOLOv8 检测 + 端到端优化** | 端到端 p50 **3.02×**、p99 **4.02×**；定位到"推理 2.0 ms 里 1.4 ms 是 CPU 派发" | [README](02-yolov8-trt/README.md) |
| 03 | **Transformer 推理优化** | 证明**一个引擎优化不了 prefill 和 decode**；KV cache 进引擎后 argmax **19/19** 一致 | [README](03-gpt-trt/README.md) |
| 04 | **BEV 算子边界** | "TRT 没有这个算子" ≠ "该写 plugin"：**四个算子逐个实测**，两个不用写、一个写了但不划算 | [README](04-bev-ops/README.md) |

---

## 三条贯穿全部项目的东西

### ① 测量纪律

同一个引擎文件在同一台笔记本上两次运行能差 **16%**（项目② 里最大见过 2.4×）。
所以：

- **不跨进程比较**，一律同进程内配对/交错测量
- 每个数字都标注口径：`per-call`（每次 sync）/ `block`（块末 sync 一次）/ `配对` / `两趟法`
- 报中位数 + σ + 95% CI，不报"最好那次"
- 用 `--null-test`（同一对象测两次）验证残余偏差

### ② 验证分层

不同层要的严格程度不同，混用会得出错误结论：

| 层 | 做法 | 例子 |
|---|---|---|
| 引擎等价性 | 同一输入喂两条路径，比相对误差 | 项目① `2.62e-06` |
| 优化语义不变 | **先填 NaN / 清零再跑**，要求逐位相同 | 项目② `max\|abs\| = 0` |
| 无参考实现时 | **定义级不变量**（数量守恒、格子唯一） | 项目② 5 万点 0 丢失 |
| 原子累加算子 | **放宽到机器精度**（浮点加法不满足结合律） | 项目④ `1.16e-10` |

### ③ 被推翻的判断（**这是最花时间的部分**）

| 我原本以为 | 实测 | 在哪 |
|---|---|---|
| 小模型的 seq 影响能直接测出来 | seq=1 和 seq=32 延迟一样——测得的是固定开销 | 项目③ §4 |
| decode 是访存受限 | 直连只有 66 GB/s；上 Graph 后 144 GB/s → **瓶颈是 launch** | 项目③ §5.2 |
| FP16 让 decode 快 1.94×（字节减半） | **1.64×**，且 FP16 的到达带宽反而低 16% | 项目③ §5.3 |
| 某个 Graph 校验失败说明是空图 | `max\|diff\| = 0` —— **是对比两侧用了不同随机输入** | 项目③ §8.4 |
| O(P) 的 KV cache 拷贝是 decode 瓶颈 | P 从 1→511 只涨 **13%**（理论 16%），被权重读取压住 | 项目③ §6.3 |
| Deformable Attention 要写 `grid_sample` plugin | TRT 11 **已内置** `IGridSampleLayer`，快 **16.64×** | 项目③ §10 |
| BEV pooling 要写 plugin | `ScatterElements(reduction='add')` 已内置（但需 `init_libnvinfer_plugins`） | 项目④ §2 |
| 写了 kernel 就该包成 plugin | 收益 **2%**，代价是静默丢点 → **判定不该** | 项目④ §4 |
| GPU 版 NMS 一定更快 | 跨主机边界时慢 **180×**；但融进同一 kernel 后快 **10×** | 项目② §9 |
| 把 NMS 做进引擎更省 | 它与 CUDA Graph 互斥：省 0.06 ms，**赔 1.4 ms** | 项目② §19 |

---

## 一条关于 TensorRT 11 的经验

从 TRT 10 到 11 的破坏性变化，回头看只有**两条主线**：

**A. 把应用负责的生命周期收回库内部**
`destroy()` 删除 → `delete`；参数从指针改引用；`IExecutionContext::setWorkspace` 消失（插件声明需求，TRT 负责分配）。

**B. 把可选模式变成唯一模式**
`BuilderFlag::kFP16/kINT8` 删除（精度由模型 dtype 决定）；`IInt8Calibrator` 删除（量化必须在 ONNX 里）；
`NetworkDefinitionCreationFlag::kEXPLICIT_BATCH` 删除（explicit batch 是唯一模式）。

> 遇到新的"TRT 11 又删了什么"，先问它属于 A 还是 B —— 比逐条记忆有用。

---

## 快速开始

每个项目都自带 README 和可复现脚本，这里只给最短路径：

```bash
# 01 分类（C++）
cd 01-resnet50-trt && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 02 检测（C++）
cd 02-yolov8-trt && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 03 Transformer（Python，需 torch + onnxruntime + tensorrt）
cd 03-gpt-trt
python export_gpt_onnx.py          # 导出 ONNX
python build_engine.py             # 建引擎 + 精度/延迟
python w2_matrix.py                # 四路交错对比（Graph × FP16）
python w3_cache.py --steps 20      # KV cache 逐步解码
python c2_verify_engine.py         # Deformable Attention 引擎验证

# 04 BEV 算子
cd 04-bev-ops/x && python c3_bev_pool.py       # BEV pooling 探测
cd c4 && cmake -S . -B build -DTRT_ROOT=$TRT_ROOT && cmake --build build -j
```

**注意**：`*.onnx` / `*.engine` 都在 `.gitignore` 里，需要用各项目的脚本自己生成。

---

## 说明

- **权重是随机初始化的**（`WEIGHT_SEED`），因为目标是测量延迟与数值一致性，不是模型效果。
- **所有延迟都是 RTX 5060 Laptop（sm_120）上的**，车端（Orin）需要重测——方案见 [P5-orin-plan](02-yolov8-trt/docs/P5-orin-plan.md)。
- **没有在真实硬件上跑过 Orin**，也没有生产级服务框架（无 continuous batching、无调度器）。
  这两点写在每个 README 的「已知限制」里。
