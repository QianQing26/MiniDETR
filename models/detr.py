# models/detr.py

import torch
import torch.nn as nn
from torchvision.models import resnet50
from .transformer import SimpleTransformer
from .position_encoding import PositionEmbeddingSine


class MiniDETR(nn.Module):
    def __init__(self, num_classes=20, num_queries=100, hidden_dim=256):
        super().__init__()

        # Backbone
        backbone = resnet50(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # Transformer
        self.position_encoding = PositionEmbeddingSine(hidden_dim // 2)
        self.transformer = SimpleTransformer()

        # Learned queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # FFN heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.bbox_embed = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        features = self.backbone(x)
        pos = self.position_encoding(features)
        src = self.input_proj(features) + pos

        B = x.size(0)
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        hs = self.transformer(src, tgt)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {
            "pred_logits": outputs_class.transpose(0, 1),
            "pred_boxes": outputs_coord.transpose(0, 1),
        }
