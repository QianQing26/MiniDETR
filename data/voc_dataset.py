# data/voc_dataset.py

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class VOCDetectionDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.data = json.load(open(json_file, "r"))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        target = {"boxes": item["boxes"], "labels": item["labels"]}

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
