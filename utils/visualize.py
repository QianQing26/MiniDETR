# utils/visualize.py
import torch
from PIL import Image, ImageDraw, ImageFont


def draw_detections(
    image_tensor: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    label_names=None,
    score_threshold: float = 0.5,
    save_path: str = "out.jpg",
):
    """
    image_tensor: [3, H, W]  float/uint8 tensor on CPU
    boxes: [N,4]  0~1 归一化  [x1, y1, x2, y2]
    labels: [N]   int
    scores: [N]   float
    label_names: list[str]  可选
    save_path: 保存图片路径
    """
    # tensor -> PIL
    if image_tensor.dtype.is_floating_point:
        image_tensor = torch.clamp(image_tensor * 255, 0, 255).byte()
    img = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # 使用默认字体，也可指定本地 ttf
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = (box * torch.tensor([W, H, W, H])).tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        name = label_names[label.item()] if label_names else f"{label.item()}"
        text = f"{name}:{score:.2f}"
        tw, th = draw.textsize(text, font=font)
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill="red")
        draw.text((x1, y1 - th), text, fill="white", font=font)

    img.save(save_path)
    print(f"Saved result to {save_path}")
