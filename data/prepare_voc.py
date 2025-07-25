# data/prepare_voc.py

import os
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from tqdm import tqdm

VOC_CLASSES = [
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


def voc_to_dict(xml_path, img_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_name = root.find("filename").text
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)

    boxes = []
    labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text.lower().strip()
        if label not in VOC_CLASSES:
            continue
        bbox = obj.find("bndbox")
        box = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        boxes.append(box)
        labels.append(VOC_CLASSES.index(label))

    return {
        "image_path": str(img_dir / image_name),
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels,
    }


def convert_voc_to_json(
    voc_root, split="trainval", output_file="data/processed/voc2012_trainval.json"
):
    annotations_dir = Path(voc_root) / "Annotations"
    images_dir = Path(voc_root) / "JPEGImages"
    split_file = Path(voc_root) / "ImageSets" / "Main" / f"{split}.txt"

    with open(split_file) as f:
        image_ids = [line.strip() for line in f.readlines()]

    all_data = []
    for img_id in tqdm(image_ids):
        xml_path = annotations_dir / f"{img_id}.xml"
        info = voc_to_dict(xml_path, images_dir)
        if len(info["boxes"]) > 0:
            all_data.append(info)

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved {len(all_data)} samples to {output_file}")


if __name__ == "__main__":
    convert_voc_to_json("data/raw/VOCdevkit/VOC2012")
