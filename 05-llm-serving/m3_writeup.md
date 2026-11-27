# M3：分页 KV cache —— 省显存、解锁 CUDA Graph，但没让 attention 变快

> 环境：RTX 5060 Laptop（26 SM，8 GB）· WSL2 Ubuntu 26.04 · torch 2.11.0+cu128 · transformers 5.24.1 · triton 3.6.0 · Qwen2.5-0.5B-Instruct FP16
> 代码：`m3a_paged.py`（分配器）· `m3b_paged_attn.py`（Triton kernel + 两个基线）· `m3c_qwen_paged.py`（接进模型）· `m3d_graph_paged.py`（上 CUDA Graph）
> 图：`m3_summary.png` · 原始数据见正文表格
> 所有对比都是**同一进程、同一次运行**内的配对；跨脚本、跨配置的数字不相互引用。

## TL;DR

1. **分页不该卖"更快"**。单独测 attention，分页 kernel 相对连续 KV 的 SDPA 是 **1.07× / 0.98×**（seq 192 / 512）—— 打平。它确实赢"gather + SDPA" 3.44× / 1.98×，但那个基线是**为了能上分页才不得不用**的中转方案。
2. 分页真正的产出是**显存**和**形状固定**。显存：8 行、长度跨度 3× 时 **36.0 → 11.2 MiB（省 69%）**；48 行但长度都接近时只省 **27%** —— 省多少完全取决于长度有多不齐。
3. 形状固定才轮到 CUDA Graph：整个 decode 步捕获成一张图后，**每步 22.8 → 7.3 ms（3.1×）**。48 行同时生成 **6559 tok/s**。
4. 但**步时几乎不随行数变**（1 行 6.36 ms → 48 行 7.32 ms，只 +15%）——说明这条路还是**固定成本**主导：一步约 1900 个 kernel，图只拿掉了 CPU 侧的 launch，没拿掉 kernel 本身。
5. 我在这条路上犯了三次同类错误：**"模型错了"其实是"我的验收工具错了"**（详见第 3 节）。所有 token 逐位对拍通过之前，任何时间数字都不算数。

---

## 1. 要回答的问题

M2 结束时的结论是"连续批处理只快 7%"，原因写的是"省下的步数被 prefill 还回去了"。但那个数字可疑：**它是在 dense 静态缓存上测的**，而 dense 缓存必须维护一个全局水位 `level` —— 每步 +1，形状就跟着变，CUDA Graph 永远捕获不了。

于是 M3 的问题是：**把 KV cache 换成分页之后，能不能让"形状固定"成立，进而让整个 decode 步进图？**

这条路上有三个必须先各自成立的小问题：

- 显存上真的省吗？（分配器）
- 分页读 KV 会不会更慢？（kernel）
- 接进真实模型还对吗？（端到端对拍）

## 2. 实现（真的做了什么）

### 2.1 分配器：物理池 + 每行一张 block table

```python
pk[i], pv[i]   # 每层一个 (num_blocks, block_size=16, kv_heads=2, head_dim=64) fp16
bt             # (rows, per_row) int32 —— 每行的块号表
seq            # (rows,) int32 —— 每行当前长度
```

写一个 token 就是一次高级索引（一个 `index_put_` kernel）：

```python
st   = seq.long() - 1                      # 约定：seq 是「已含当前 token」的长度
blk  = bt[arange(rows), st // bs].long()
off  = st % bs
pk[li][blk, off] = k                       # 同时写 k 和 v
```

读一个 token 是 `kernel` 里的间接寻址：`blk = bt[b, i]`，再读 `pk[blk, :, :]`。

**约定 `seq` 含当前 token** 这件事不是风格问题，它是 bug 的分界线 —— 第一版写成"写在下标 `seq`、又用长度 `seq` 去 attend"，新 token 就被自己 mask 掉了。

### 2.2 分页 attention kernel（Triton）

一个 program 负责 `(batch, head)` 的一行，逐块沿 block table 走：

```python
for i in range(0, nblk):
    blk = tl.load(BT + b * s_bt_b + i)
    k   = tl.load(K + blk * s_k_blk + tl.arange(0, BS)[:, None] * s_k_t + kvh * s_k_h
                  + tl.arange(0, D)[None, :])                       # (BS, D)
    s   = tl.sum(k * q[None, :], axis=1) * SM_SCALE
    s   = tl.where(pos < seq, s, -inf)                              # 这一行还没写到的位置
    ...
```

GQA 用算术映射，不物化：`kvh = h * HKV // HQ`（14 个 q 头 → 2 个 kv 头）。

### 2.3 接进 Qwen2.5：只换每层的 `self_attn.forward`

prefill 仍走 HF 原路径（逐条，DynamicCache），把每层 KV 按块拷进池子；decode 时 monkeypatch 掉 24 层的 attention：

```python
q, k, v = q_proj(x), k_proj(x), v_proj(x)      # 复用 HF 的 weight
q, k = apply_rotary_pos_emb(q, k, cos, sin)    # 复用 HF 的 rotary
write_tokens(...)                              # 写一个新 token 进池子
o = paged_attn_triton(q[:, :, 0, :], pk[li], pv[li], bt, seq, bs, scaling)
return o_proj(o.reshape(*input_shape, -1)), None
```

两个坑记一下（都是 v5 的接口变化，不是算法）：

- `Qwen2Config` **没有 `head_dim`**，要 `getattr(cfg, "head_dim", hidden_size // num_heads)`；
- attention 模块上**只有** `head_dim` / `num_key_value_groups` / `scaling`，**没有** `num_heads` / `num_key_value_heads` —— 头数必须从 `self.config` 读；
- `position_ids` 必须 2-D，传 1-D 会在 `masking_utils.py:756 find_packed_sequence_indices` 里 `IndexError: too many indices for tensor of dimension 1`。

### 2.4 上 CUDA Graph：整步只有原地写

```python
def step():
    seq.add_(1)                                        # 进图
    out = model(input_ids=ids, position_ids=pos, use_cache=False)
    tokbuf.copy_(out.logits[:, -1, :].argmax(-1, keepdim=True))
    ids.copy_(tokbuf)
    pos.add_(1)
```

`ids / pos / seq / bt / KV 池` 全部形状恒定，所以整步可以被捕获成一张图，一次 `replay()` 代替约 1900 次 launch。

对照一下 **dense 为什么做不到**：dense 的 `level` 是全局水位、每步 +1 —— 它是"每行可见到哪"的唯一载体，于是它的值一变，形状就变。分页把它换成"每行一个 `seq` 标量 + 一张只读 `bt`"，形状这才恒定。

热身的做法是：先用一次性状态跑 3 步（Triton JIT 必须在此之前），然后 `seq.sub_(3) / pos.sub_(3) / ids.copy_(首 token)` 退回起点再捕获。**KV 不用清** —— attention 读的范围由 `seq` 决定，退回去再前进会在同样位置重写同样内容。

## 3. 怎么证明它是对的（三次"是我错了"）

对拍链条：HF 原路径（逐条、无分页、无图）→ 分页 + gather+SDPA → 分页 + Triton → 分页 + Triton + 图。**每一跳都要求 token 逐位一致**。

| 出错的东西 | 症状 | 真相 |
|---|---|---|
| 我写的 `write_tokens` 初值取自 dense 跑完后的 `nxt` | 两条路径**整体错开一格**：首 token 不同、其余全同 | 序列对齐问题，不是精度问题 |
| 我把 `o.transpose(1,2)` 再 `reshape` | layer0 输出 0.217，参考 0.0058（**37 倍**） | `(Hq,D)` 被转成 `(D,Hq)`，896 维全错位 |
| 我的验收脚本拿第 0 步 logits 直接比 | logits 差 **1.9e+01**，看起来完全崩了 | 参考的第 0 步来自 prefill（预测 token 1），我的第 0 步来自喂进 token 1 之后（预测 token 2）—— 对齐后 **4.3e-02** |

第三条最值得记：**在宣布"模型错了"之前，先证明"验收工具没错"**。这次要不是先做了一次单行、单步的最小对拍，那 19 的差距足以让人去改一个根本没坏的 kernel。

最终对拍结果（24 行 × 15 步，prompt 64/106/149/192）：

```
图路径 token 逐位=True  全程一致的步=15/15  不一致的行=[]
  第 0 行 mine [916, 279, 15678, 5562, 13]
  第 0 行 ref  [916, 279, 15678, 5562, 13]
logits 相对差 max = 2.212e-03
```

logits 那个 2.2e-3 里有一部分来自 **B=1（参考）vs B=24（我的）** 的 GEMM shape 不同 —— 同一份权重、同一个算法，不同 kernel 的归约顺序不同。这也是为什么最终判据是 token 而不是 logits。

## 4. 结果

### 4.1 kernel 本身不比连续 KV 快（`m3b`）

| seq | dense SDPA（连续 KV） | gather + SDPA | 分页 kernel |
|---|---|---|---|
| 192 | 23.4 µs | 75.1 µs | **21.9 µs（3.44× vs gather，1.07× vs dense）** |
| 512 | 38.3 µs | 77.4 µs | **39.1 µs（1.98× vs gather，0.98× vs dense）** |

正确性：`7.629e-06`（相对 `2.597e-04`，seq 192）、`1.907e-06`（`1.293e-04`，seq 512）、长度不一时 `3.815e-06` —— 都在 fp16 eps（4.9e-4）量级。

**这里我改过一次基线，值得单独写**：第一版报的是 11.13× / 15.45×，那是假的。两个原因叠加：

1. 基线带 `attn_mask` 时 SDPA 退回了 **math 后端**（489.7 µs vs 23.4 µs，21 倍差距全是基线自己的问题）；
2. 为 GQA 写了 `repeat_interleave` 把 KV 物化了 7 倍。

把基线修成"满长、无 mask、`enable_gqa=True`"之后，收益从 11 倍掉到 3.4 倍，再掉到"相对 dense 打平"。**比之前先问"基线是不是它的最好状态"** —— 这和当初 COCO mAP 那次同源。

### 4.2 分配器确实省显存（`m3a`）

```
对拍：逐位一致 OK (192 个 token)
步时中位数  dense 20.09 ms   paged 24.28 ms   (+21%)
物理块：已分配 60（每行上限 7，共 84）
KV 显存：按需 11.2 MiB / 预留 15.8 MiB / dense 对照组 36.0 MiB
```

这一版的 +21% 全是 gather 的代价（多读一遍、多写一遍）—— 也正是它逼着我去写 4.1 那个 kernel。

### 4.3 端到端：21.7× 来自 batching，不来自 kernel（`m3c`）

| 路径 | 每步 | 24 行吞吐 |
|---|---|---|
| HF 原路径（逐条跑，24 条一轮 479.54 ms） | 每条约 19.98 ms | 50 tok/s |
| 分页 + Triton（批量一次前向） | **22.12 ms** | **1085 tok/s（21.7×）** |

但这一步 **B=1 21.60 / B=8 21.83 / B=24 22.12 ms（+1.3%）** —— 24 倍的算力和 KV 读取只换来 1.3% 的时间。**这条路是 CPU 绑定的**，21.7× 是"24 行共享一次 Python 开销"买来的。

### 4.4 上图之后：3.1–3.3×，且随行数几乎不变（`m3d`）

| rows | eager 每步 | 图每步 | 加速 | tok/s | 是权重下界 2.75 ms 的 |
|---|---|---|---|---|---|
| 1 | 20.947 ms | **6.363** | 3.29× | 157 | 2.3× |
| 8 | 21.026 | **6.469** | 3.25× | 1237 | 2.3× |
| 24 | 21.413 | **6.918** | 3.10× | 3469 | 2.5× |
| 48 | 22.797 | **7.318** | 3.12× | **6559** | 2.7× |

每个配置都是**所有行、全部步 token 逐位一致**（1/8/24/48 行各 7/7 步；24 行另有一次 15/15 步）。

## 5. 为什么步时不随行数涨

一步的组成：

```
理论：942 MB 权重 ÷ 334 GB/s = 2.75 ms
      + 约 1900 个 kernel × 1.15 µs 的缝 = 2.2 ms      （1.15 µs 由 m0b 实测）
      ≈ 4.95 ms
实测：6.36（1 行）→ 7.32（48 行）
```

行数涨 48 倍只多 0.96 ms，说明权重读取和 kernel 缝都是**固定成本**；而 48 行那一步的 KV 才 200 MiB，比权重小得多。

多出来的 1.5–2.4 ms，我判断是**我那个 Triton kernel 本身慢**（`num_warps=1`、逐块循环、间接寻址），不是 launch 开销 —— 但 WSL2 里 `ncu` 用不起来（CUPTI 报 `CUPTI_ERROR_INVALID_DEVICE`），**这条只是判断，不是结论**。

## 6. 结论

1. **分页不是性能优化，是布局优化**。它把"KV 在哪"从"一个全局连续的 `(B,H,L,D)`"变成"每行一张表 + 一个池"，换来两样东西：按需的显存，和不随序列增长的形状。
2. **形状固定才是那 5 倍**。M0 就算过：单请求 decode 16.75 ms 里 GPU 只用了 2.76 ms，剩下全是 CPU。图把这部分拿掉，于是 3.1–3.3× 立刻出现，而且**行数越多越划算**（同样的固定成本摊给更多行）。
3. **显存收益取决于长度有多不齐**。8 行跨度 3× 时省 69%，48 行但长度接近时只省 27%。分页不是无条件省显存，它省的是"按最长的那一行预留"造成的浪费。
4. **21.7×（4.3 节）和 3.1×（4.4 节）是两个杠杆**，别混在一起说：前者是 batching（24 行共享一次 Python），后者是 CUDA Graph（拿掉 CPU launch）。分页提供了让后者成立的前提。

## 7. 已知限制

1. **prefill 仍是逐条、走 HF 原路径**，没有分页也没有批量 —— 它现在占墙钟的一半以上，是 M4 的事。
2. **block table 是静态的**：每行在初始化时就占住一段连续块号，`bt` 之后不再重排。这不是真正的分配器（没有 free list、没有回收、没有跨行复用），换请求需要外部重新填表。
3. **Triton kernel 是教学实现**：`num_warps=1`、单 program 走完所有块、没有 split-K、没有向量化。M3b 的单 kernel 数据说明它在**这个规模**下与 SDPA 打平，不代表更大规模或更长上下文也打平。
4. **那 1.5–2.4 ms 的额外开销没有归因**（`ncu` 用不了）。
5. **规模上限**：序列 ≤ 192、行数 ≤ 48、单模型单卡。8 GB 显存下 48 行 × 266 token 的池子只要 200 MiB，所以瓶颈不是显存而是这里没测更长的上下文。
6. **logits 有 2.2e-3 的相对差**，其中一部分是 B=1 vs B=24 的 GEMM shape 差异，没有分离出来。
7. **`tok/s` 的口径**是"所有行的 prompt 都已预填好、只测 decode 步"。TTFT 与 prefill 完全没算进去，不能当成"server 能跑 6559 tok/s"。

## 8. 复现

```bash
# 0) 依赖：torch + transformers 5.x + triton（本仓库不含模型权重，需自己下载 Qwen2.5-0.5B-Instruct）
pip install torch transformers triton

# 1) 分配器（不接模型，只看显存与对拍）
~/.venvs/torch/bin/python m3a_paged.py

# 2) kernel 微基准（三条路 vs 两个基线）
~/.venvs/torch/bin/python m3b_paged_attn.py

# 3) 接进 Qwen：token 逐位对拍 + 步时
bash run_m3c.sh 24 16

# 4) 上 CUDA Graph：四档行数
bash scan_m3d.sh 8

# 5) 重画图
bash plot_m3.sh
```

依赖：`torch 2.11.0+cu128`、`transformers 5.24.1`、`triton 3.6.0`、`matplotlib 3.11.2`。模型需提前下到本地（脚本用 `local_files_only=True`，否则每次都会去联网重试）。
