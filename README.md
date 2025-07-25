# 🧩 Mini-DETR

> A lightweight and educational implementation of [DETR (End-to-End Object Detection with Transformers)](https://arxiv.org/abs/2005.12872) using PyTorch.  
> Designed for small GPU devices (≤8GB), educational use, and clarity-first principles.

---

## ✨ Highlights

- ✅ Clean implementation with **no external detection libraries** (e.g. no HuggingFace, MMDetection)
- ✅ Fully working **training pipeline** (ResNet50 + Transformer)
- ✅ Built-in **Hungarian matching loss**
- ✅ Supports **learning rate warmup**, **backbone freezing**, **differential LR**
- ✅ Includes **mAP evaluation** using only PyTorch and NumPy
- ✅ Structure & code style inspired by open-source conventions

---

## 📂 Project Structure

```
MiniDETR/
 ├── configs/              # (Optional) configs
 ├── data/                 # dataset JSON & transform
 │   ├── voc_dataset.py
 │   ├── transforms.py
 ├── models/               # model components
 │   ├── detr.py
 │   ├── transformer.py
 │   ├── position_encoding.py
 ├── utils/                # matcher, loss, evaluation
 │   ├── matcher.py
 │   ├── loss.py
 │   ├── eval_utils.py
 ├── train.py              # training script
 ├── eval.py               # mAP evaluation
 ├── logs/                 # training_log.csv
 ├── checkpoints/          # model checkpoints

```

---

## 📦 Dataset Format

The model expects a **COCO-like JSON**, but simplified.  
Each entry should contain:

```json
{
  "image_path": "data/voc/JPEGImages/000001.jpg",
  "boxes": [[xmin, ymin, xmax, ymax], ...],
  "labels": [int, int, ...]
}
```

Bounding boxes must be **absolute pixel coordinates** (they will be normalized internally).

You can easily preprocess VOC, COCO, or your own dataset to this format.

--------

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision tqdm
```

### 2. Prepare dataset

Place your `*.json` file under `data/processed/your_file.json`.

### 3. Train

```bash
python train.py
```

- Supports **training log**, **checkpoint saving**, **tqdm progress bar**

You can modify configs directly in `train.py`:

```python
num_epochs = 100
batch_size = 8
base_lr = 1e-4
backbone_lr = 5e-6
```

### 4. Evaluate

```bash
python eval.py
```

- Uses training set as validation (if no val split is provided)
- Outputs **per-class AP** and **mean AP**

------

## 🛠 Features Implemented

| Feature                              | Status |
| ------------------------------------ | ------ |
| ResNet-50 Backbone (Frozen/Unfrozen) | ✅      |
| Sinusoidal Position Encoding         | ✅      |
| Transformer Encoder + Decoder        | ✅      |
| Object Queries                       | ✅      |
| Set Prediction Loss (Hungarian)      | ✅      |
| CrossEntropy + L1 + GIoU Loss        | ✅      |
| Training Log CSV                     | ✅      |
| Checkpoint Save (Best + Last)        | ✅      |
| Learning Rate Warmup                 | ✅      |
| Different LR for backbone            | ✅      |
| mAP Evaluation (IoU ≥ 0.5)           | ✅      |

------

## 🧪 Notes

- The project is designed to **fit in 8GB GPU memory**
- Input image is resized to 320×320 during training & evaluation
- You can **freeze backbone** in early epochs and unfreeze later
- **mAP is computed only with PyTorch/NumPy** — no COCO APIs

------

## 📘 References

- [DETR Paper (Facebook AI)](https://arxiv.org/abs/2005.12872)
- [Official DETR Repo](https://github.com/facebookresearch/detectron2)
- [A guide to Object Detection mAP](https://github.com/rafaelpadilla/Object-Detection-Metrics)

------

## 🧑‍💻 Author

This project is a personal implementation by a graduate student exploring vision-language models.
 Feel free to fork, modify, or contribute!

> 📫 Feedback or issue reports are welcome via GitHub Issues.

------

