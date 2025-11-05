import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# 位置编码 (Positional Encoding)
# -------------------------------
class PositionalEncoding(nn.Module):
    """
    给序列中的每个位置添加位置编码，使模型能够感知顺序信息
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 位置索引 (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位使用 cos
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # 将位置编码加到输入 embedding 上
        return x + self.pe[:, :x.size(1), :].to(x.dtype)

# -------------------------------
# 多头注意力 (Multi-Head Attention)
# -------------------------------
class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个 head 的维度

        # Q/K/V 的线性映射
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)  # 最终输出线性映射
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: (B, L_q, d_model)
        key:   (B, L_k, d_model)
        value: (B, L_v, d_model)
        mask:  可选掩码 (B, 1, L_q, L_k) 或 (B, L_q, L_k)
        """
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()
        B, L_v, _ = value.size()

        # 1. 线性映射
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # 2. reshape 为多头
        q = q.view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)  # (B, heads, L_q, d_k)
        k = k.view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, L_v, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 计算 Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, heads, L_q, L_k)

        # 4. 应用 mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B,1,L_q,L_k)
            elif mask.dim() == 4 and mask.size(1) == 1:
                pass  # 已经是正确形状
            else:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L_k)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # (B, heads, L_q, d_k)

        # 5. 合并 heads 并通过输出线性层
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        output = self.out(output)
        return output, attn

# -------------------------------
# 前馈网络 (Feed Forward)
# -------------------------------
class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络，逐位置独立应用两层全连接
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()  # 激活函数
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

# -------------------------------
# Encoder 层
# -------------------------------
class EncoderLayer(nn.Module):
    """
    Transformer Encoder 单层
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

# -------------------------------
# Decoder 层
# -------------------------------
class DecoderLayer(nn.Module):
    """
    Transformer Decoder 单层
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 自注意力
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 编码器-解码器注意力
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # 1. 自注意力
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

        # 2. 编码器-解码器注意力
        enc_attn_output, _ = self.enc_attn(x, memory, memory, memory_mask)
        x = x + self.dropout(enc_attn_output)
        x = self.norm2(x)

        # 3. 前馈网络
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)
        return x

# -------------------------------
# Encoder / Decoder 堆叠
# -------------------------------
class Encoder(nn.Module):
    """
    Transformer Encoder 堆叠层
    """
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    """
    Transformer Decoder 堆叠层
    """
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

# -------------------------------
# 完整 Transformer 模型
# -------------------------------
class Transformer(nn.Module):
    """
    完整 Transformer 模型，包括编码器、解码器、embedding 和输出投影
    """
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=8, d_ff=512,
                 enc_layers=2, dec_layers=2, dropout=0.1, max_len=512, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding 层
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # 编码器与解码器
        self.encoder = Encoder(enc_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(dec_layers, d_model, n_heads, d_ff, dropout)

        # 输出投影层
        self.out_proj = nn.Linear(d_model, tgt_vocab)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        前向传播
        src: (B, L_src)
        src_mask: (B, 1, 1, L_src)
        tgt: (B, L_tgt)
        tgt_mask: (B, 1, L_tgt, L_tgt)
        """
        # embedding + 位置编码
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_enc(src_emb)
        tgt_emb = self.pos_enc(tgt_emb)

        # 编码器
        memory = self.encoder(src_emb, src_mask)
        # 解码器
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)

        # 输出投影
        output = self.out_proj(decoder_output)
        return output
