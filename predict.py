# predict.py

import torch
from PIL import Image
from torchvision import transforms
from models.detr import MiniDETR
from utils.visualize import draw_detections
import sys
import os

# 设置类别名（VOC 20类）
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = MiniDETR()
model.load_state_dict(torch.load("checkpoints/best.pth", map_location=device))
model.to(device)
model.eval()

# 图像预处理 pipeline（和训练时一致）
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 读取图像路径
if len(sys.argv) != 2:
    print("Usage: python predict.py path_to_image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# 推理
with torch.no_grad():
    outputs = model(input_tensor)

logits = outputs["pred_logits"][0]  # (num_queries, num_classes+1)
boxes = outputs["pred_boxes"][0]  # (num_queries, 4)

probs = logits.softmax(-1)
scores, labels = probs[..., :-1].max(-1)  # 排除最后一个no-object类

# 可视化
from utils.visualize import draw_detections

fig = draw_detections(
    input_tensor[0].cpu(),
    boxes.cpu(),
    labels.cpu(),
    scores.cpu(),
    label_names=CLASSES,
    score_threshold=0.5,
)

# 保存结果
save_path = f"predicted_{os.path.basename(img_path)}"
fig.savefig(save_path, bbox_inches="tight")
print(f"✅ Prediction saved to: {save_path}")
