# eval.py

import torch
from torch.utils.data import DataLoader
from data.voc_dataset import VOCDetectionDataset
from data.transforms import ResizeAndToTensor
from models.detr import MiniDETR
from tqdm import tqdm
import os

from utils.eval_utils import Evaluator  # 下一步我们会写这个
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argparse 让你支持命令行运行
parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, default="data/processed/voc2012_trainval.json")
parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
parser.add_argument("--iou_threshold", type=float, default=0.5)
args = parser.parse_args()

# 加载模型
model = MiniDETR()
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.to(device)
model.eval()

# 加载数据集
dataset = VOCDetectionDataset(args.json, transform=ResizeAndToTensor((320, 320)))
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
)

print(f"Loaded {len(dataset)} samples for evaluation.")

# 评估及初始化
evaluator = Evaluator(num_classes=20, iou_threshold=args.iou_threshold)
# 初始化评估器
evaluator = Evaluator(num_classes=20, iou_threshold=args.iou_threshold)

# 推理每张图
with torch.no_grad():
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        image = images[0].to(device)
        target = targets[0]

        image_tensor = image.unsqueeze(0)
        output = model(image_tensor)

        pred_logits = output["pred_logits"][0]  # (num_queries, num_classes+1)
        pred_boxes = output["pred_boxes"][0]  # (num_queries, 4)

        probs = pred_logits.softmax(-1)
        scores, labels = probs[:, :-1].max(-1)  # 去掉 no-object 类

        evaluator.update(
            pred_boxes.cpu(),
            labels.cpu(),
            scores.cpu(),
            torch.tensor(target["boxes"]),
            torch.tensor(target["labels"]),
        )

# 输出结果
evaluator.summarize()
