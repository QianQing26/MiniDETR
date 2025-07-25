# utils/matcher.py

import torch
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def __call__(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            cost_class = -out_prob[b][:, tgt_ids]  # shape: [num_queries, num_targets]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)  # L1

            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
