# data/transforms.py

from torchvision import transforms as T
from PIL import Image
import torch


class ResizeAndToTensor:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.transform = T.Compose(
            [
                T.Resize(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image, target):
        w_orig, h_orig = image.size
        image = self.transform(image)
        _, h_new, w_new = image.shape

        # 变换 boxes 坐标
        scale_x = w_new / w_orig
        scale_y = h_new / h_orig

        boxes = torch.tensor(target["boxes"], dtype=torch.float)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # !!!!归一化到 [0, 1] 区间
        boxes[:, [0, 2]] /= w_new
        boxes[:, [1, 3]] /= h_new

        labels = torch.tensor(target["labels"], dtype=torch.long)
        target = {"boxes": boxes, "labels": labels}
        return image, target
