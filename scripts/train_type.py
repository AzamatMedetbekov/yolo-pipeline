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
from pathlib import Path

import torch
import torch.nn as nn
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
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        type_class = int(r["type_class"])
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
    patience: int,
):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.02,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = FridgeTypeDataset(csv_path, img_base_dir, split="train", transform=train_tfm)
    val_ds = FridgeTypeDataset(csv_path, img_base_dir, split="val", transform=val_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = TypeNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "typenet_fridge.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

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

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "type_labels": TYPE_LABELS,
                },
                save_path,
            )
            print(f"  -> New best val acc! Saved to {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"[INFO] Early stopping at epoch {epoch} "
                f"(no val acc improvement for {patience} epochs)"
            )
            break

    print(f"\n[OK] Training finished. Best val acc: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train fridge type classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="yolov12/data/fridge_attr10",
        help="Base dataset directory containing images/train and images/val",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/fridge_type_labels.csv",
        help="CSV labels file with split/image_name/type_class",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (0, 1, cpu, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="type_runs",
        help="Output directory for saving model checkpoints",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without val acc improvement)",
    )

    args = parser.parse_args()

    data_dir = resolve_path(args.data)
    csv_path = resolve_path(args.csv)
    output_dir = resolve_path(args.output)
    img_base_dir = os.path.join(data_dir, "images")

    device_str = normalize_device(args.device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("Fridge Type Classifier Training")
    print("=" * 50)
    print(f"Data Dir: {data_dir}")
    print(f"CSV     : {csv_path}")
    print(f"Epochs  : {args.epochs}")
    print(f"Batch   : {args.batch}")
    print(f"Device  : {device}")
    print(f"Output  : {output_dir}")
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
        epochs=args.epochs,
        batch_size=args.batch,
        device=device,
        output_dir=output_dir,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
