#!/usr/bin/env python3
"""
Fridge Type Classifier Training Script

Features:
- CLI arguments for common training parameters
- Dataset validation before training
- Cross-platform path handling
"""

import argparse
import csv
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================
# Default Configuration
# ============================
NUM_CLASSES = 6
TYPE_LABELS = {
    0: "vertical",
    1: "horizontal",
    2: "vertical_open",
    3: "horizontal_open",
    4: "combination",
    5: "coldroom",
}

IMG_SIZE = 224
LR = 1e-3

def set_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path_str)


def normalize_device(device_str: str) -> str:
    if device_str.lower() == "cpu":
        return "cpu"
    if device_str.isdigit():
        return f"cuda:{device_str}"
    return device_str


def load_yaml_config(path_str: str) -> dict:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"[WARN] Config not found: {path}. Using CLI/defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_run_dir(output_dir: str, name: str) -> str:
    if not name:
        name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(output_dir, name)
    if os.path.exists(run_dir):
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"{run_dir}_{suffix}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _parse_range(value, default):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return default
    return default


def build_transforms(augmentations: dict, img_size: int):
    defaults = {
        "random_resized_crop": {"scale": (0.6, 1.0), "ratio": (0.75, 1.33)},
        "horizontal_flip": 0.5,
        "rotation": 10,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.15,
            "hue": 0.02,
        },
        "random_grayscale": 0.05,
        "random_perspective": {"distortion_scale": 0.1, "p": 0.2},
        "val": {"resize": 256, "center_crop": img_size},
    }
    aug = augmentations if isinstance(augmentations, dict) else {}
    train_cfg = aug.get("train", {}) if isinstance(aug.get("train", {}), dict) else {}
    val_cfg = aug.get("val", {}) if isinstance(aug.get("val", {}), dict) else {}

    crop_cfg = train_cfg.get("random_resized_crop", {})
    crop_scale = _parse_range(crop_cfg.get("scale"), defaults["random_resized_crop"]["scale"])
    crop_ratio = _parse_range(crop_cfg.get("ratio"), defaults["random_resized_crop"]["ratio"])

    flip_p = float(train_cfg.get("horizontal_flip", defaults["horizontal_flip"]))
    rotation = float(train_cfg.get("rotation", defaults["rotation"]))

    cj_cfg = train_cfg.get("color_jitter", {})
    if not isinstance(cj_cfg, dict):
        cj_cfg = {}
    cj_defaults = defaults["color_jitter"]
    brightness = float(cj_cfg.get("brightness", cj_defaults["brightness"]))
    contrast = float(cj_cfg.get("contrast", cj_defaults["contrast"]))
    saturation = float(cj_cfg.get("saturation", cj_defaults["saturation"]))
    hue = float(cj_cfg.get("hue", cj_defaults["hue"]))

    gray_p = float(train_cfg.get("random_grayscale", defaults["random_grayscale"]))

    persp_cfg = train_cfg.get("random_perspective", {})
    if not isinstance(persp_cfg, dict):
        persp_cfg = {}
    persp_defaults = defaults["random_perspective"]
    distortion_scale = float(persp_cfg.get("distortion_scale", persp_defaults["distortion_scale"]))
    persp_p = float(persp_cfg.get("p", persp_defaults["p"]))

    val_resize = int(val_cfg.get("resize", defaults["val"]["resize"]))
    val_center_crop = int(val_cfg.get("center_crop", defaults["val"]["center_crop"]))

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=crop_scale, ratio=crop_ratio),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.RandomRotation(degrees=rotation),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
        transforms.RandomGrayscale(p=gray_p),
        transforms.RandomPerspective(distortion_scale=distortion_scale, p=persp_p),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(val_center_crop),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfm, val_tfm


def count_labeled_rows(csv_path: str, split: str) -> int:
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("split") != split:
                continue
            if r.get("type_class", "").strip() == "":
                continue
            count += 1
    return count


def validate_dataset(data_dir: str, csv_path: str) -> bool:
    img_base_dir = os.path.join(data_dir, "images")
    train_dir = os.path.join(img_base_dir, "train")
    val_dir = os.path.join(img_base_dir, "val")

    valid = True

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        valid = False
    else:
        train_rows = count_labeled_rows(csv_path, "train")
        val_rows = count_labeled_rows(csv_path, "val")
        print(f"[INFO] Labeled rows: train={train_rows}, val={val_rows}")
        if train_rows == 0 or val_rows == 0:
            print("[ERROR] CSV has no labeled rows for train or val")
            valid = False

    if not os.path.isdir(train_dir):
        print(f"[ERROR] Training images dir not found: {train_dir}")
        valid = False
    else:
        train_images = list(Path(train_dir).glob("*.[jJ][pP][gG]")) + list(
            Path(train_dir).glob("*.[pP][nN][gG]")
        )
        print(f"[INFO] Training images: {len(train_images)}")

    if not os.path.isdir(val_dir):
        print(f"[ERROR] Validation images dir not found: {val_dir}")
        valid = False
    else:
        val_images = list(Path(val_dir).glob("*.[jJ][pP][gG]")) + list(
            Path(val_dir).glob("*.[pP][nN][gG]")
        )
        print(f"[INFO] Validation images: {len(val_images)}")

    return valid


# ============================
# Dataset
# ============================

class FridgeTypeDataset(Dataset):
    def __init__(self, csv_path: str, img_base_dir: str, split: str, transform=None):
        self.transform = transform
        self.img_base_dir = img_base_dir

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"] != split:
                    continue
                if r["type_class"].strip() == "":
                    continue
                rows.append(r)

        if not rows:
            raise RuntimeError(f"No rows for split={split} in {csv_path}")

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = os.path.join(self.img_base_dir, r["split"], r["image_name"])
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Failed to load {img_path}: {e}. Using placeholder.")
            # dummy image
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        type_class = int(r["type_class"])
        if type_class < 0 or type_class >= NUM_CLASSES:
            raise ValueError(
                f"Invalid type_class={type_class} in {r['image_name']}. "
                f"Must be in range [0, {NUM_CLASSES-1}]"
            )
        y = torch.tensor(type_class, dtype=torch.long)

        return img, y


# ============================
# Model: ResNet18 + Single Classification Head
# ============================

class TypeNet(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.fc = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.fc(feat)
        return out


# ============================
# Training Loop
# ============================

def train(
    csv_path: str,
    img_base_dir: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
    run_name: str,
    patience: int,
    img_size: int,
    lr: float,
    workers: int,
    use_amp: bool,
    augmentations: dict,
):
    train_tfm, val_tfm = build_transforms(augmentations, img_size)

    train_ds = FridgeTypeDataset(csv_path, img_base_dir, split="train", transform=train_tfm)
    val_ds = FridgeTypeDataset(csv_path, img_base_dir, split="val", transform=val_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = TypeNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    criterion = nn.CrossEntropyLoss()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = amp.GradScaler(enabled=amp_enabled)

    run_dir = build_run_dir(output_dir, run_name)

    best_val_acc = -1.0
    epochs_no_improve = 0
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    log_path = os.path.join(run_dir, "train_log.csv")
    per_class_path = os.path.join(run_dir, "per_class_metrics.csv")
    summary_path = os.path.join(run_dir, "metrics.json")

    with open(log_path, "w", newline="", encoding="utf-8") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "lr",
                "epoch_sec",
            ]
        )

    with open(per_class_path, "w", newline="", encoding="utf-8") as per_class_f:
        per_class_writer = csv.writer(per_class_f)
        header = ["epoch"]
        for idx in range(NUM_CLASSES):
            name = TYPE_LABELS.get(idx, f"class_{idx}")
            header.extend(
                [
                    f"acc_{name}",
                    f"prec_{name}",
                    f"rec_{name}",
                    f"f1_{name}",
                ]
            )
        per_class_writer.writerow(header)

    best_confusion = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            train_loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
        with torch.no_grad(), torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                labels_cpu = labels.detach().cpu()
                preds_cpu = preds.detach().cpu()
                indices = labels_cpu * NUM_CLASSES + preds_cpu
                confusion += torch.bincount(
                    indices, minlength=NUM_CLASSES * NUM_CLASSES
                ).reshape(NUM_CLASSES, NUM_CLASSES)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        confusion_f = confusion.to(torch.float32)
        row_sum = confusion_f.sum(dim=1)
        col_sum = confusion_f.sum(dim=0)
        diag = torch.diag(confusion_f)

        per_class_acc = torch.where(row_sum > 0, diag / row_sum, torch.zeros_like(diag))
        per_class_rec = per_class_acc
        per_class_prec = torch.where(col_sum > 0, diag / col_sum, torch.zeros_like(diag))
        per_class_f1 = torch.where(
            (per_class_prec + per_class_rec) > 0,
            2 * per_class_prec * per_class_rec / (per_class_prec + per_class_rec),
            torch.zeros_like(diag),
        )

        valid_rec = row_sum > 0
        valid_prec = col_sum > 0
        macro_recall = per_class_rec[valid_rec].mean().item() if valid_rec.any() else 0.0
        macro_precision = (
            per_class_prec[valid_prec].mean().item() if valid_prec.any() else 0.0
        )
        macro_f1 = (
            per_class_f1[valid_rec].mean().item() if valid_rec.any() else 0.0
        )

        epoch_sec = time.perf_counter() - epoch_start
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )
        print(
            f"  Macro P/R/F1: {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as log_f:
            log_writer = csv.writer(log_f)
            log_writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{macro_precision:.6f}",
                    f"{macro_recall:.6f}",
                    f"{macro_f1:.6f}",
                    f"{lr:.8f}",
                    f"{epoch_sec:.3f}",
                ]
            )

        with open(per_class_path, "a", newline="", encoding="utf-8") as per_class_f:
            per_class_writer = csv.writer(per_class_f)
            row = [epoch]
            for i in range(NUM_CLASSES):
                row.extend(
                    [
                        f"{per_class_acc[i].item():.6f}",
                        f"{per_class_prec[i].item():.6f}",
                        f"{per_class_rec[i].item():.6f}",
                        f"{per_class_f1[i].item():.6f}",
                    ]
                )
            per_class_writer.writerow(row)

        torch.save(
            {
                "model_state": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "type_labels": TYPE_LABELS,
            },
            last_path,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_epoch = epoch
            best_confusion = confusion.clone()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "type_labels": TYPE_LABELS,
                },
                best_path,
            )
            print(f"  -> New best val acc! Saved to {best_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"[INFO] Early stopping at epoch {epoch} "
                f"(no val acc improvement for {patience} epochs)"
            )
            break

    if best_confusion is not None:
        cm_path = os.path.join(run_dir, "confusion_matrix.csv")
        with open(cm_path, "w", newline="", encoding="utf-8") as cm_f:
            cm_writer = csv.writer(cm_f)
            cm_writer.writerow([""] + [TYPE_LABELS.get(i, f"class_{i}") for i in range(NUM_CLASSES)])
            for i in range(NUM_CLASSES):
                cm_writer.writerow(
                    [TYPE_LABELS.get(i, f"class_{i}")]
                    + [str(int(v)) for v in best_confusion[i].tolist()]
                )

    with open(summary_path, "w", encoding="utf-8") as summary_f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "epochs_ran": epoch,
                "output_dir": run_dir,
                "best_checkpoint": best_path,
                "last_checkpoint": last_path,
            },
            summary_f,
            indent=2,
        )

    print(f"\n[OK] Training finished. Best val acc: {best_val_acc:.4f}")
    print(f"[OK] Outputs saved under: {run_dir}")


def main():
    set_seed(42)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/fridge_type.yaml",
        help="Path to YAML config",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train fridge type classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override base dataset directory containing images/train and images/val",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override CSV labels file with split/image_name/type_class",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override input image size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=None,
        help="Enable automatic mixed precision (CUDA only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device to use (0, 1, cpu, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory for saving results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override run name (default: auto timestamp)",
    )

    args = parser.parse_args(remaining)

    def pick(arg_value, key: str, default):
        if arg_value is not None:
            return arg_value
        return config.get(key, default)

    global NUM_CLASSES
    global TYPE_LABELS
    if "type_labels" in config:
        TYPE_LABELS = {int(k): v for k, v in config["type_labels"].items()}
    if "num_classes" in config:
        NUM_CLASSES = int(config["num_classes"])
    else:
        NUM_CLASSES = len(TYPE_LABELS)

    data_value = pick(args.data, "data", "data")
    csv_value = pick(args.csv, "csv", "data/fridge_type_labels.csv")
    epochs_value = int(pick(args.epochs, "epochs", 50))
    batch_value = int(pick(args.batch, "batch", 4))
    imgsz_value = int(pick(args.imgsz, "img_size", IMG_SIZE))
    lr_value = float(pick(args.lr, "lr", LR))
    device_value = str(pick(args.device, "device", "0"))
    output_value = pick(args.output, "output", "runs/type")
    name_value = pick(args.name, "name", "")
    amp_value = bool(pick(args.amp, "amp", False))
    patience_value = int(config.get("patience", 5))
    workers_value = int(config.get("workers", 0))
    augmentations_value = config.get("augmentations", {})

    data_dir = resolve_path(data_value)
    csv_path = resolve_path(csv_value)
    output_dir = resolve_path(output_value)
    img_base_dir = os.path.join(data_dir, "images")

    device_str = normalize_device(device_value)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("Fridge Type Classifier Training")
    print("=" * 50)
    print(f"Data Dir: {data_dir}")
    print(f"CSV     : {csv_path}")
    print(f"Epochs  : {epochs_value}")
    print(f"Batch   : {batch_value}")
    print(f"ImgSize : {imgsz_value}")
    print(f"LR      : {lr_value}")
    print(f"AMP     : {amp_value}")
    print(f"Device  : {device}")
    print(f"Output  : {output_dir}")
    print(f"Name    : {name_value or '(auto)'}")
    print("=" * 50)

    print("\nType classes:")
    for idx in sorted(TYPE_LABELS.keys()):
        print(f"  {idx}: {TYPE_LABELS[idx]}")

    print("\n[INFO] Validating dataset...")
    if not validate_dataset(data_dir, csv_path):
        print("[ERROR] Dataset validation failed. Exiting.")
        sys.exit(1)

    print("\n[INFO] Starting training...")
    train(
        csv_path=csv_path,
        img_base_dir=img_base_dir,
        epochs=epochs_value,
        batch_size=batch_value,
        device=device,
        output_dir=output_dir,
        run_name=name_value,
        patience=patience_value,
        img_size=imgsz_value,
        lr=lr_value,
        workers=workers_value,
        use_amp=amp_value,
        augmentations=augmentations_value,
    )


if __name__ == "__main__":
    main()
