# train.py

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from models.detr import MiniDETR
from data.voc_dataset import VOCDetectionDataset
from data.transforms import ResizeAndToTensor
from utils.matcher import HungarianMatcher
from utils.loss import SetCriterion
from tqdm import tqdm
import os
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configs
batch_size = 8
num_epochs = 3
lr = 1e-3
best_loss = float("inf")

# dataset
dataset = VOCDetectionDataset(
    "data/processed/voc2012_trainval.json", transform=ResizeAndToTensor((320, 320))
)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
)
print(f"Number of samples: {len(dataset)} loaded in {len(dataloader)} batches")

# model
model = MiniDETR()
model.to(device)
model.load_state_dict(torch.load("checkpoints/last.pth"))

# freeze backbone
# for param in model.backbone.parameters():
#     param.requires_grad = False

print("Model loaded on device: ", device)

# loss & matcher
matcher = HungarianMatcher()
criterion = SetCriterion(
    num_classes=20,
    matcher=matcher,
    weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
)

print("Criterion and matcher initialized")


# optimizer
def build_optimizer(model, base_lr=1e-5, backbone_lr=5e-7):
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.transformer.parameters(), "lr": base_lr},
            {"params": model.query_embed.parameters(), "lr": base_lr},
            {"params": model.class_embed.parameters(), "lr": base_lr},
            {"params": model.bbox_embed.parameters(), "lr": base_lr},
        ],
        weight_decay=1e-4,
    )


def get_warmup_scheduler(optimizer, warmup_steps, base_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = build_optimizer(model)
scheduler = get_warmup_scheduler(optimizer, warmup_steps=1000, base_lr=1e-4)


print("Optimizer initialized")

# checkpoint & log dir
os.makedirs("checkpoints", exist_ok=True)
log_file = open("logs/training_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["epoch", "loss", "loss_ce", "loss_bbox", "loss_giou"])

print("Training started")

# training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss, total_ce, total_bbox, total_giou = 0, 0, 0, 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        images_tensor = torch.stack(images)
        outputs = model(images_tensor)

        loss, loss_dict = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_ce += loss_dict["loss_ce"]
        total_bbox += loss_dict["loss_bbox"]
        total_giou += loss_dict["loss_giou"]

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.3f}",
                "ce": f"{loss_dict['loss_ce']:.2f}",
                "bbox": f"{loss_dict['loss_bbox']:.2f}",
                "giou": f"{loss_dict['loss_giou']:.2f}",
            }
        )

    # avg losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_ce = total_ce / n_batches
    avg_bbox = total_bbox / n_batches
    avg_giou = total_giou / n_batches

    # save log
    log_writer.writerow([epoch, avg_loss, avg_ce, avg_bbox, avg_giou])
    log_file.flush()

    # save checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"checkpoints/best.pth")
    if epoch == num_epochs:
        torch.save(model.state_dict(), f"checkpoints/last.pth")

log_file.close()
