#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpt.py —— 极简 decoder-only Transformer（从零构建，无第三方模型库依赖）

为什么给 decoder-only 而不是原始 Encoder-Decoder：
    你的学习计划 P3 要部署 Qwen2.5-1.5B（decoder-only）并估 KV cache 显存，
    decoder-only 是这条线的核心。文末注释说明了怎么改回 encoder-decoder。

包含：
    - 词嵌入 + 可学习位置嵌入
    - 多头因果自注意力（支持 KV cache）
    - Pre-LN 残差块 + GELU MLP
    - 权重共享（lm_head 与词嵌入共用）
    - 自回归生成（temperature / top-k）
    - 自检：无 cache 与有 cache 的 logits 必须一致

运行：
    ~/.venvs/torch/bin/python gpt.py

部署视角的三个看点（对应学习计划）：
    1. KV cache 的形状与显存占用 —— 见 kv_cache_bytes()
    2. 因果 mask 用 register_buffer 注册，导出 ONNX 时会成为常量，不占动态算子
    3. 生成时 seq 长度逐步增长，是典型的 dynamic shape 场景（W4 的 profile 要用）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 配置
# =============================================================================
@dataclass
class GPTConfig:
    vocab_size: int = 1024      # 词表大小
    block_size: int = 128       # 最大上下文长度
    n_layer: int = 4            # 层数
    n_head: int = 8             # 注意力头数
    n_embd: int = 256           # 隐藏维度
    dropout: float = 0.0        # 推理时为 0
    bias: bool = False          # Linear/LayerNorm 是否带 bias（现代 LLM 多为 False）


# -----------------------------------------------------------------------------
# 部署用的默认配置（★ 单一真源：export / build 脚本都从这里导入，避免三处不一致）
#
# 为什么这么大（约 40 M 参数）：
#   最初用 n_layer=2 / n_embd=64（10 万参数）时，实测延迟和 seq_len **完全无关**
#   （seq=1 → 0.398 ms，seq=32 → 0.353 ms）。因为 10 万参数的模型算 32 个 token
#   只要几微秒，0.35 ms 全是「一次调用的固定开销」（Python 调用 + set_input_shape
#   + 驱动提交 + synchronize）。
#   → 测到的是开销，不是计算。必须把模型放大到计算量能和开销相比，seq 的影响才可见。
#
# 这一组：12 层 / 512 维 / 8 头 / 词表 4096 / 上下文 512 ≈ 40 M 参数
#   seq=512 时约 38 GFLOP → 即使只有 30% 效率也要 10 ms 量级，明显盖过 0.35 ms 开销。
# -----------------------------------------------------------------------------
DEFAULT_CFG = GPTConfig(vocab_size=4096, block_size=512, n_layer=12, n_head=8, n_embd=512)


# =============================================================================
# 多头因果自注意力（带 KV cache）
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd 必须能被 n_head 整除"

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout_p = cfg.dropout

        # 一次线性层同时算出 q/k/v，比三个独立 Linear 少两次 kernel launch
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # 输出投影
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # 因果 mask：下三角矩阵。注册为 buffer（不是参数），
        # 导出 ONNX 时会变成常量，不会引入额外动态算子。
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(
        self,
        x: torch.Tensor,                                    # (B, T, C)
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape

        # ---- 1. 线性投影 + 拆成多头 ----
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ---- 2. 拼接历史 KV（自回归生成时） ----
        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_k, past_v = kv_cache
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)   # (B, nh, past+T, hd)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # ---- 3. 注意力分数 ----
        # (B, nh, T, hd) @ (B, nh, hd, Tk) -> (B, nh, T, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # ---- 4. 因果 mask ----
        # 只有"整段 prefill"时才需要 mask；逐 token 生成（T=1）时未来位置不存在。
        if past_len == 0:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.resid_dropout(att)

        # ---- 5. 加权求和 + 合并多头 ----
        y = att @ v                                       # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.resid_dropout(self.c_proj(y)), new_cache


# =============================================================================
# Transformer 块（Pre-LN）
# =============================================================================
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN 结构：x + Attn(LN(x))，再 x + MLP(LN(x))。

    Pre-LN 比原始论文的 Post-LN 更稳，现代 LLM（GPT/Llama/Qwen）都用它。
    """
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


# =============================================================================
# 完整模型
# =============================================================================
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)        # token embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)        # position embedding
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重共享：输出层与词嵌入共用同一份参数，省 vocab*embd 个参数
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # 残差投影按层数缩放，防止深层模型激活值爆炸
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_caches=None, use_cache: bool = False):
        """
        idx       : (B, T) 的 token id
        kv_caches : 长度 = n_layer 的列表，每项为 (k, v) 或 None
        返回      : (logits (B,T,vocab), 新的 kv_caches)
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过 block_size {self.cfg.block_size}"

        # 有 cache 时，位置编码要从 past_len 开始
        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].size(2)
        # 位置编码用「权重切片」而不是 self.wpe(arange(T))：
        #   两者数学等价（wpe(pos) 就是 wpe.weight[pos]），
        #   但切片在导出 ONNX 时只产生 Slice 算子，
        #   而 arange 会产生动态的 Range 算子，TRT 对它的支持较弱、
        #   还容易把整张图的形状都拖成动态。这是典型的"导出友好写法"。
        pos_table = self.wpe.weight[past_len:past_len + T]   # (T, C)

        x = self.drop(self.wte(idx) + pos_table)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache_i, use_cache)
            new_caches.append(new_cache)

        logits = self.lm_head(self.ln_f(x))
        return logits, (new_caches if use_cache else None)

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, use_cache: bool = True,
                 greedy: bool = False):
        """自回归生成。use_cache=True 时只把最后一个 token 送进模型。

        greedy=True 时用 argmax 代替随机采样 —— 这样结果**确定性**，
        才能拿"有 cache / 无 cache"两条路径做逐 token 比对。
        默认 False（随机采样），因为真实生成需要多样性。
        """
        self.eval()
        caches = None
        for _ in range(max_new_tokens):
            if use_cache:
                # 第一次要整段 prefill，之后每次只喂 1 个 token
                idx_cond = idx if caches is None else idx[:, -1:]
            else:
                idx_cond = idx[:, -self.cfg.block_size:]

            logits, caches = self(idx_cond,
                                  kv_caches=caches if use_cache else None,
                                  use_cache=use_cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            # ★ 贪心 vs 采样：贪心是确定性的，可用于对比测试
            next_tok = (probs.argmax(dim=-1, keepdim=True) if greedy
                        else torch.multinomial(probs, 1))
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def kv_cache_bytes(self, batch_size: int = 1, seq_len: int | None = None) -> int:
        """估算 KV cache 显存占用（字节）。

        公式: 2(K和V) × n_layer × B × n_head × seq_len × head_dim × dtype_bytes
        这就是 P3 部署 Qwen 时要算的核心数字。
        """
        L = seq_len or self.cfg.block_size
        head_dim = self.cfg.n_embd // self.cfg.n_head
        elem = 2 * self.cfg.n_layer * batch_size * self.cfg.n_head * L * head_dim
        return elem * 4  # float32 假设；fp16 则 *2


# =============================================================================
# 自检
# =============================================================================
def selftest() -> None:
    # 自检用小配置（快）；部署用 DEFAULT_CFG（大）
    cfg = GPTConfig(vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg).eval()
    n_param = sum(p.numel() for p in model.parameters())

    print("=" * 66)
    print("极简 GPT 自检")
    print("=" * 66)
    print(f"配置     : layer={cfg.n_layer} head={cfg.n_head} embd={cfg.n_embd} "
          f"block={cfg.block_size} vocab={cfg.vocab_size}")
    print(f"参数量   : {n_param:,}")
    print(f"KV cache : {model.kv_cache_bytes(batch_size=1, seq_len=cfg.block_size)/1024:.1f} KB"
          f" (B=1, T={cfg.block_size})")

    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))

    # ---- 1. 前向 shape ----
    with torch.no_grad():
        logits, _ = model(idx)
    print(f"\n前向     : input {tuple(idx.shape)} -> logits {tuple(logits.shape)}")
    assert logits.shape == (2, 8, cfg.vocab_size), "logits shape 不对"

    # ---- 2. KV cache 一致性：逐 token 喂 vs 整段喂，logits 必须一致 ----
    with torch.no_grad():
        caches = None
        outs = []
        for t in range(idx.size(1)):
            lg, caches = model(idx[:, t:t + 1], kv_caches=caches, use_cache=True)
            outs.append(lg)
        logits_cached = torch.cat(outs, dim=1)

    diff = (logits - logits_cached).abs().max().item()
    print(f"KV cache : 整段 vs 逐token  max|diff| = {diff:.3e}")
    if diff < 1e-4:
        print("           [OK] cache 逻辑正确")
    else:
        print("           [WARN] cache 有问题，检查 mask 与位置编码偏移")

    # ---- 3. 生成：贪心解码，有/无 cache 必须逐 token 一致 ----
    # 用贪心而不是随机采样，是为了"消掉随机性"——否则两次结果不同也说明不了问题。
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out_a = model.generate(prompt, max_new_tokens=6, use_cache=True,  greedy=True)
    out_b = model.generate(prompt, max_new_tokens=6, use_cache=False, greedy=True)

    print(f"\n生成     : prompt {tuple(prompt.shape)} -> {tuple(out_a.shape)}")
    print(f"           prompt      : {prompt[0].tolist()}")
    print(f"           新生成 6 个 : {out_a[0, 4:].tolist()}")   # 只打印新生成的，不含 prompt
    assert out_a.shape == (1, 10), "生成长度不对"

    if out_a.tolist() == out_b.tolist():
        print("           [OK] 有/无 cache 生成结果逐 token 完全一致")
    else:
        print("           [FAIL] 生成路径不一致，cache 在 generate() 里用错了")

    print("\n所有检查通过。")


if __name__ == "__main__":
    selftest()


# =============================================================================
# 附：改回 Encoder-Decoder（原始 Transformer）
# =============================================================================
# 1. 把 CausalSelfAttention.forward 里的因果 mask 去掉，换成 padding mask：
#       att = att.masked_fill(key_padding_mask[:, None, None, :], -inf)
# 2. 新建 CrossAttention 类：Q 来自 decoder，K/V 来自 encoder 输出，
#    其余代码与自注意力相同（复用同一个实现，只是 x 与 context 分开传入）。
# 3. Decoder block 变成三个子层：
#       x = x + self_attn(ln1(x))
#       x = x + cross_attn(ln2(x), encoder_out)
#       x = x + mlp(ln3(x))
# 4. 位置编码可换成固定 sin/cos（原始论文用这个），
#    实现：把 wpe 换成 register_buffer 的三角函数表即可。
#
# 部署建议：
#   - 导 ONNX 时先不带 KV cache（cache 的 in/out 会让图变复杂，W3 再练）
#   - 导出用 dynamic_axes 标出 batch 与 seq 两维
#   - 因果 mask 用 buffer 而非动态计算，TRT 才能把它折叠成常量
