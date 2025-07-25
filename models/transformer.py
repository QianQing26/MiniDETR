# models/transformer.py

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: [HW, B, D]
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(nn.ReLU()(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # tgt: (N, B, D), memory: (HW, B, D)
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(nn.ReLU()(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class SimpleTransformer(nn.Module):
    def __init__(
        self, num_encoder_layers=5, num_decoder_layers=5, d_model=256, nhead=8
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)]
        )

    def forward(self, src, tgt):
        # src: (B, D, H,  W), tgt: (N, B, D)
        B, D, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory)

        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory)

        return output
