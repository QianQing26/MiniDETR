# utils/loss.py

import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou

from utils.matcher import HungarianMatcher


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        weight = torch.ones(num_classes + 1)
        weight[-1] = 0.1
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=weight.to(self.device))
        self.bbox_loss_fn = nn.L1Loss()

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        bs, num_queries = outputs["pred_logits"].shape[:2]
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_cls = self.cls_loss_fn(src_logits.transpose(1, 2), target_classes)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)])
        loss_bbox = self.bbox_loss_fn(src_boxes, target_boxes)
        giou_loss = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes)).mean()

        loss = (
            self.weight_dict["loss_ce"] * loss_cls
            + self.weight_dict["loss_bbox"] * loss_bbox
            + self.weight_dict["loss_giou"] * giou_loss
        )
        return loss, {
            "loss_ce": loss_cls.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_giou": giou_loss.item(),
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
