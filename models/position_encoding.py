# models/position_encoding.py

import torch
import math
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    正弦形式的位置编码，借鉴DETR官方库实现
    输入： [B, C, H, W]
    输出： [B, C, H, W]
    """

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)

        y_embed = torch.cumsum(~mask, dim=1, dtype=torch.float32)
        x_embed = torch.cumsum(~mask, dim=2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
