# BEV 算子边界：哪些该写 kernel/plugin，哪些不该

> 这个项目和项目②（`02-yolov8-trt`）是同一根线：**BEV 感知的前处理与后处理怎么放进 TRT**。
> 项目② 解决的是 decode/NMS 与点云 scatter；这里补上 **BEV pooling** 和一个更重要的东西——
> **判断"该不该自己写"的方法**。

---

## 0. 一句话结论

> **"TRT 没有这个算子"不等于"该写 plugin"。**
> **我原计划写两个 plugin，实测发现——一个都不用写；第三个虽然写了，但收益 2%，所以也不该写。**

| 算子 | TRT 支持吗 | 该怎么办 | 实测 |
|---|---|---|---|
| `grid_sample`（注意力采样） | ✅ **原生层** `IGridSampleLayer` | 直接用 | PASSED，见项目③ §10 |
| **BEV pooling**（固定输出 scatter-add） | ✅ **内置 plugin** `ScatterElements` v2 | **调 `init_libnvinfer_plugins()` 后直接用** | vs 累加 **1.164e-10** |
| **点云 pillarization**（动态输出规模） | ❌ | 手写 kernel（**不封装成 plugin**） | 见 §4 |

---

## 1. 判据：三层"能不能" + 一层"值不值"

前三层是**能力判断**，第四层是**成本判断** —— 而第四层是我最初漏掉的。

| 层 | 问题 | 做法 |
|---|---|---|
| **①** | TRT 原生层里有吗？ | 直接用（`grid_sample`） |
| **②** | TRT 用内置 plugin 实现了吗？ | **`trt.init_libnvinfer_plugins(logger, "")`** 后用（`ScatterElements`、`EfficientNMS`） |
| **③** | 都没有？ | 写 kernel |
| **④** | **写 kernel 之后还要不要包成 plugin？** | **先算收益**（本文 §4 就是这么否掉的） |

**第 ④ 层的关键**：包成 plugin 只省一次 launch（本项目实测 **0.015 ms**），
代价是**固定容量约束**（超了就静默丢点）和**调试难度**。
**在端到端 0.75 ms 的流水线里，2% 的收益换不来说明不了任何问题，只会换来一个静默失败的模式。**

---

## 2. BEV pooling：TRT 支持，但有两个坑

### 2.1 它是什么

```python
bev[b, :, rank[m]] += feat[b, :, m] * weight[b, 0, m]
```

对每个来源 `m`（相机 × 深度 × 像素），按 `rank` 累加到 BEV 格子上。**多个来源会落到同一格** → 必须原子累加。

### 2.2 坑一：`trtexec` PASSED 但你的代码加载不了

ONNX 导出后是 `ScatterElements(reduction='add')`，`trtexec` **PASSED**。
但在纯 Python 进程里 `createExecutionContext()` 直接失败：

```
[TRT] [E] Cannot find plugin: ScatterElements, version: 2, namespace:
[TRT] [E] Cannot deserialize plugin since corresponding IPluginCreatorInterface 
         not found in Plugin Registry
```

**原因**：`ScatterElements` 被 TRT 实现成一个**内置 plugin**，需要显式初始化插件库。

```python
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
```

> **`trtexec` 自己初始化了插件库，你的进程不会自动初始化。**
> **所以"trtexec PASSED"从来不等于"我的代码能用这个引擎"。**

### 2.3 坑二：`reduction='add'` 会不会被忽略

`ScatterElements` 的 `reduction` 属性如果是 `none`，语义就从**累加**变成**覆盖写**——
引擎照样能建、能跑，**但结果全错**。

**所以必须用"两个参照答案"来判别**（`rank` 重复率 68%，平均每格 3 个来源，差别足够大）：

| 参照答案 | 引擎输出 vs 它 |
|---|---|
| **累加**（`reduction='add'`） | **`1.164e-10`** ✅ |
| **覆盖写**（`reduction='none'`） | `1.069e-03` |

**结论：TRT 正确实现了累加。**

### 2.4 顺带发现：scatter 类算子**做不到逐位复现**

连跑两次差 `1.164e-10`（相对 1e-7）。

**因为 scatter-add 是原子浮点累加，而浮点加法不满足结合律**（`(a+b)+c ≠ a+(b+c)`），线程到达顺序每次不同。

> **所以"先填 NaN 再跑、要求 `max|abs| = 0`"这个在项目② 里很好用的判据，
> 在 scatter 类算子上**必须放宽到 float32 机器精度**。**

---

## 3. 点云 pillarization：TRT 真的没有

### 3.1 为什么和 BEV pooling 不一样

| 操作 | 输出形状 | TRT |
|---|---|---|
| BEV pooling | 固定 `bev_H × bev_W`（预分配） | ✅ `ScatterElements` |
| **pillarization** | **有几根 pillar 由数据决定** | ❌ |

**两者都是 "scatter"，区别只在**输出规模是否静态可知**。**

**而 TRT 的 plugin 同样无法表达完全动态的输出形状**（输出形状要用 `DimsExprs` 从输入推导），
所以标准解法是：**固定容量 + count 标量**（`EfficientNMS` 也是这么做）。

### 3.2 手写 kernel（这部分**该写**）

`c4/scatter_kernel.cu` 从项目② 直接搬来，那边通过了 5 层验证
（手算 / 定义级不变量 / NumPy 参考 / CPU 参考 / 对比器自检）：**5 万点 → 28403 根 pillar，0 点丢失**。

**两个并发铁律**（各踩过一次）：

```cuda
// ① 自旋等共享标志必须 volatile 读，否则编译器可以把读提到循环外 → 死循环
volatile int* slot = &cell_to_pid[cell];

// ② 发布 pillar 编号前必须初始化完 + __threadfence()
//    否则别的线程看到半成品 → 数据静默出错（当时 69 个格子）
__threadfence();
atomicExch(&cell_to_pid[cell], pid);
```

**★ 还有一个编译期前提**：这个 kernel 用原子抢占 + 自旋等待，**需要 Volta 及以后的独立线程调度（sm_70+）**。
如果编成了 compute_52，会**真死锁**（见 §5）。所以加了防护：

```cuda
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "scatter_kernel 需要 sm_70+（Volta 及以后的独立线程调度）"
#endif
```

---

## 4. 为什么**不**把它包成 plugin（第 ④ 层的答案）

我做了，验证通过，然后判定**不该做**。

### 4.1 做出来的东西是真的

`c4/pillar_plugin.cpp`（276 行）实现了完整的 `IPluginV3`：三件套能力 + creator + 静态注册 + workspace。

**四个输出**（"固定容量 + count"方案）：

```
pillars     [max_pillars][max_pts][F]  float
coords      [max_pillars][4]           int32   (batch, z, y, x)
counts      [max_pillars]              int32
num_pillars [1]                        int32   ← 真实数量（runtime 决定）
```

**验证结果**：

```
[①] 直接调 kernel: 28403 根 pillar
[②] creator 找到了 / network: 1 层, 4 输出 / engine 26.9 KB
[③] plugin 28403 == 直接 28403  ✅
    按集合比（排序后逐元素）: ✅ 完全一致，共 50000 个点
[④] 直接 0.104 ms → plugin 0.089 ms
```

### 4.2 但收益不成立

```
省下 0.015 ms
端到端（项目② 的 0.752 ms）里占 2%
```

**代价**：

| 代价 | 具体 |
|---|---|
| **固定容量** | `max_pillars=60000`，某帧超过就**静默丢点** |
| **调试更难** | 出问题要看 TRT 内部，不能直接断点 kernel |
| **多 740 行维护** | 而这 740 行只是把已有 kernel 包了一层 |

> **用 2% 的速度换一个静默失败的风险 —— 这个交易不该做。**
>
> **而之所以还是做了，原因是"计划里写了要写"，不是"部署里需要"。**
> **这正是项目①②里反复强调的那个判据（先确认瓶颈，再动手）失效的一次。**

### 4.3 那这 740 行白写了吗

**学到的东西留下了，产物不该留。**

留下的部分：
- `IPluginV3` 的完整机制（`getCapabilityInterface` 按类型分发 + `clone()` + 三组能力 + workspace 声明）
- **`IPluginV3` 的正确类结构是"多重继承 + 能力分发"，不是"注册回调"** —— 这个靠读头文件才能确认，猜不出来
- 什么时候**非写不可**：输出规模动态 + 那条路径真的在关键路径上

---

## 5. 最贵的教训：**架构配错会让自旋等待死锁**

**症状**：程序卡在 `cudaDeviceSynchronize()` 不返回，`GPU util 69%` 但 **SM 频率只有 457 MHz**。

**真因是一条三条连锁**：

```
① CMAKE_CUDA_ARCHITECTURES 默认成了 52
   （我用 if(NOT DEFINED) 兜底 —— 无效，因为 CMake **默认就会定义**它）
        ↓
② kernel 只编出 compute_52 的 PTX，在 sm_120 上走 PTX JIT
        ↓
③ compute_52 的编程模型是 **warp 锁步**（Volta 之前没有独立线程调度），
   而且 __nanosleep 被 #if __CUDA_ARCH__ >= 700 编译掉了
        ↓
★ 自旋等待的线程卡住同 warp 里持有 claim 的线程 → **真死锁**
```

**正确写法**：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "" FORCE)   # 必须 FORCE
```

**诊断口诀**：

| GPU util | SM 频率 | 含义 |
|---|---|---|
| 高 | **低** | **空转**（自旋/轮询）← 本例 |
| 高 | 高 | 真在算 |
| 低 | 低 | 在等数据（host 侧或 PCIe） |

---

## 6. 文件

| 文件 | 作用 |
|---|---|
| `c3_bev_pool.py` | BEV pooling（`scatter_add`）+ 暴力参考对拍 + 导出 ONNX + trtexec |
| `c3_verify_engine.py` | **判别 `reduction='add'` 是否被正确实现**（两个参照答案） |
| `c4/scatter_kernel.{h,cu}` | 点云 scatter kernel（从项目② 搬来，含 sm_70+ 编译期防护） |
| `c4/pillar_plugin.cpp` | `IPluginV3` 实现（**保留作为 API 参考，不建议用于生产**） |
| `c4/main.cpp` | driver：直接调 kernel vs plugin-in-engine 对拍 |

---

## 7. 已知限制

1. **plugin 用默认流**（TRT 警告 `Using default stream in enqueueV3()`），非默认流还能再快一点。
2. **只验证了单 batch、单一网格参数**；`attachToContext` 的 clone 路径没被多 context 场景覆盖。
3. **BEV pooling 只验证了 2D、单 batch、固定 `bev_n`**；真实 BEVPoolv2 的 argsort + rank 路径更复杂。
4. **`reduction='add'` 的验证依赖"rank 有重复"** —— 如果输入没有冲突，累加与覆盖写无法区分（本例重复率 68%，所以有区分力）。
