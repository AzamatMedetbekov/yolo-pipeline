"""
Train a 6-class refrigerator type classifier using ResNet18.

Type classes:
    0: vertical        - Normal refrigerator (drink freezer)
    1: horizontal      - Horizontal (ice cream freezer)
    2: vertical_open   - Vertical without door
    3: horizontal_open - Horizontal without door
    4: combination     - Two types combined
    5: coldroom        - Walk-in cold room
"""
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ============================
# Configuration
# ============================
BASE_DIR = "yolov12/data/fridge_attr10"
IMG_BASE_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = "data/fridge_type_labels.csv"

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
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("CUDA is available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available, running on CPU")


# ============================
# Dataset
# ============================

class FridgeTypeDataset(Dataset):
    def __init__(self, csv_path: str, split: str, transform=None):
        self.transform = transform

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"] != split:
                    continue
                # Skip rows with empty type_class
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
        img_path = os.path.join(IMG_BASE_DIR, r["split"], r["image_name"])
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

def train():
    # ----- Dataset / Dataloader -----
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = FridgeTypeDataset(CSV_PATH, split="train", transform=tfm)
    val_ds = FridgeTypeDataset(CSV_PATH, split="val", transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = TypeNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # =========================
        #        TRAIN PHASE
        # =========================
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

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

        # =========================
        #        VAL PHASE
        # =========================
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        # =========================
        #        PRINT
        # =========================
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("type_runs", exist_ok=True)
            save_path = os.path.join("type_runs", "typenet_fridge.pt")
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "type_labels": TYPE_LABELS,
            }, save_path)
            print(f"  -> New best val acc! Saved to {save_path}")

    print(f"\n[OK] Training finished. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
