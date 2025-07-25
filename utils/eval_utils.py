# utils/eval_utils.py

import torch
import numpy as np
from collections import defaultdict


def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4], boxes2: [M, 4], both in [x1, y1, x2, y2] normalized format
    return: IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


class Evaluator:
    def __init__(self, num_classes=20, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.pred_by_class = defaultdict(list)
        self.gt_by_class = defaultdict(list)

    def update(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        """
        收集每张图片的预测与GT，加入缓冲区用于后续计算
        - pred_boxes: [N, 4], normalized
        - pred_labels: [N]
        - pred_scores: [N]
        - gt_boxes: [M, 4], normalized
        - gt_labels: [M]
        """
        # 转为绝对值更利于计算IoU
        pred_boxes = pred_boxes.detach()
        gt_boxes = gt_boxes.detach()
        pred_labels = pred_labels.detach()
        gt_labels = gt_labels.detach()

        for cls in range(self.num_classes):
            # 选出该类的GT和预测
            gt_mask = gt_labels == cls
            pred_mask = pred_labels == cls

            cls_gt_boxes = gt_boxes[gt_mask]
            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]

            n_gt = len(cls_gt_boxes)
            self.gt_by_class[cls].append(n_gt)

            if len(cls_pred_boxes) == 0:
                continue

            # IoU匹配
            if len(cls_gt_boxes) == 0:
                for score in cls_pred_scores:
                    self.pred_by_class[cls].append((score.item(), 0))  # 0表示FP
                continue

            ious = box_iou(cls_pred_boxes, cls_gt_boxes)  # [N_pred, N_gt]
            assigned_gt = set()
            for i in range(len(cls_pred_boxes)):
                max_iou, gt_idx = ious[i].max(0)
                if max_iou >= self.iou_threshold and gt_idx.item() not in assigned_gt:
                    self.pred_by_class[cls].append((cls_pred_scores[i].item(), 1))  # TP
                    assigned_gt.add(gt_idx.item())
                else:
                    self.pred_by_class[cls].append((cls_pred_scores[i].item(), 0))  # FP

    def compute_ap(self, scores_and_labels):
        """
        传入 [(score1, is_tp1), (score2, is_tp2), ...]，计算 AP
        """
        if len(scores_and_labels) == 0:
            return 0.0

        scores_and_labels.sort(key=lambda x: -x[0])
        tps = np.array([x[1] for x in scores_and_labels])
        fps = 1 - tps

        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)

        recalls = tp_cumsum / max(tp_cumsum[-1], 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            precisions_at_t = precisions[recalls >= t]
            if precisions_at_t.size > 0:
                ap += precisions_at_t.max()
        return ap / 11

    def summarize(self):
        aps = []
        print(f"\n=== Evaluation Result (IoU ≥ {self.iou_threshold}) ===")
        for cls in range(self.num_classes):
            preds = self.pred_by_class[cls]
            n_gt = sum(self.gt_by_class[cls])
            if n_gt == 0:
                continue
            ap = self.compute_ap(preds)
            aps.append(ap)
            print(f"Class {cls:2d}: AP = {ap:.4f} (GT: {n_gt})")
        mAP = sum(aps) / len(aps) if aps else 0.0
        print(f"\n👉 mAP: {mAP:.4f}")
