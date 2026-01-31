import os
import csv
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F


# used to normalize the values to common metric for loss calculation, consequently, training


def calculate_dataset_stats(csv_path, split="train"):
    print(f"[INFO] Calculating stats from {csv_path} ({split})...")

    # storage
    values = {k: [] for k in REG_ATTRS}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue

        for col in REG_ATTRS:
            try:
                val = float(row[col])
                values[col].append(val)
            except ValueError:
                continue

    mean = {}
    std = {}

    for col, val_list in values.items():
        if len(val_list) == 0:
            print(f"[WARNING] No data found for {col}, using default 0/1")
            mean[col] = 0.0
            std[col] = 1.0
            continue

        t = torch.tensor(val_list, dtype=torch.float32)
        mean[col] = t.mean().item()
        std[col] = t.std().item()

        if std[col] < 1e-6:
            std[col] = 1.0

    print("Calculated Means:", mean)
    print("Calculated Stds: ", std)
    return mean, std


# ============================
# Configuration
# ============================
BASE_DIR = "yolov12/data/fridge_attr10"
IMG_BASE_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = "data/fridge_attr10_labels.csv"

# Define which attrs are regression / classification
# For attrs with large values, apply scaling and translation
REG_SCALE = {
    "attr3_internal_volume": 1000.0,  # Liters → scaled to ~0-2 range
    "attr8_year": 50.0,  # (year - 2000) / 50
}
REG_SHIFT = {
    "attr3_internal_volume": 0.0,
    "attr8_year": 2000.0,
}

REG_ATTRS: List[str] = [
    "attr1_power_kw",
    "attr2_u_value",
    "attr3_internal_volume",
    "attr4_temp_class",
    "attr8_year",
    "attr10_misc",
]

CLS_ATTRS: Dict[str, int] = {
    "attr5_refrigerant": 3,
    "attr6_door_type": 3,
    "attr7_cabinet_type": 2,
    "attr9_insulation_type": 2,
}

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 200  # Set higher for overfitting test
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


class FridgeAttrDataset(Dataset):
    def __init__(self, csv_path: str, split: str, stats=None, transform=None):
        self.transform = transform
        self.rows = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"] != split:
                    continue
                self.rows.append(r)

        if stats is None:
            self.means = {k: 0.0 for k in REG_ATTRS}
            self.stds = {k: 1.0 for k in REG_ATTRS}
        else:
            self.means = stats["mean"]
            self.stds = stats["std"]

        # Check if regression / classification columns exist
        # for col in REG_ATTRS:
        #     if col not in self.rows[0]:
        #         raise KeyError(f"Missing regression column: {col}")
        # for col in CLS_ATTRS.keys():
        #     if col not in self.rows[0]:
        #         raise KeyError(f"Missing classification column: {col}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = os.path.join(IMG_BASE_DIR, r["split"], r["image_name"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # regression target
        reg_vals = []
        for col in REG_ATTRS:
            v = float(r[col])

            mu = self.means.get(col, 0.0)
            sigma = self.stds.get(col, 1.0)

            norm_v = (v - mu) / sigma

            reg_vals.append(norm_v)

        y_reg = torch.tensor(reg_vals, dtype=torch.float32)

        # Classification targets: LongTensor per attribute
        y_cls = {}
        for col in CLS_ATTRS.keys():
            v = int(r[col])
            y_cls[col] = torch.tensor(v, dtype=torch.long)

        return img, y_reg, y_cls


# ============================
# Model: ResNet18 + Multi-head
# ============================


class AttrNet(nn.Module):
    def __init__(self, reg_attrs: List[str], cls_attrs: Dict[str, int]):
        super().__init__()
        self.reg_attrs = reg_attrs
        self.cls_attrs = cls_attrs

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # regression head
        self.reg_head = None
        if len(reg_attrs) > 0:
            self.reg_head = nn.Linear(in_feat, len(reg_attrs))

        # classification heads
        self.cls_heads = nn.ModuleDict()
        for name, num_classes in cls_attrs.items():
            self.cls_heads[name] = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        feat = self.backbone(x)  # (B, in_feat)

        out = {}
        if self.reg_head is not None:
            out["reg"] = self.reg_head(feat)  # (B, len(REG_ATTRS))

        cls_out = {}
        for name, head in self.cls_heads.items():
            cls_out[name] = head(feat)  # (B, C)
        out["cls"] = cls_out

        return out


# ============================
# Training Loop
# ============================


def print_loss_table(title, names, train_losses, val_losses):
    print(f"\n[{title}]")
    # Auto-adjust column width based on attribute names
    col1_w = max(len("attr"), max(len(n) for n in names))
    header = f"{'attr'.ljust(col1_w)} | {'train':>10} | {'val':>10}"
    print(header)
    print("-" * len(header))
    for name in names:
        t = train_losses[name]
        v = val_losses[name]
        print(f"{name.ljust(col1_w)} | {t:10.4f} | {v:10.4f}")


def train():
    # ----- Dataset / DataLoader -----

    train_means, train_stds = calculate_dataset_stats(CSV_PATH, split="train")
    stats_bundle = {"mean": train_means, "std": train_stds}

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01
            ),
            transforms.ToTensor(),
            norm,
        ]
    )

    # The old tranformation resized all to square of size(IMG_SIZE, IMG_SIZE), which make the attributes like the height and width
    # to be overlooked and squashed, i.e. all images becomes a squashed square, therefore we changed it. The old one is below.

    # tfm = transforms.Compose([
    #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225],
    #     ),
    # ])

    val_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            norm,
        ]
    )

    train_ds = FridgeAttrDataset(
        CSV_PATH, split="train", stats=stats_bundle, transform=train_tfm
    )

    val_ds = FridgeAttrDataset(
        CSV_PATH, split="val", stats=stats_bundle, transform=val_tfm
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AttrNet(REG_ATTRS, CLS_ATTRS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    cls_crits = {name: nn.CrossEntropyLoss() for name in CLS_ATTRS.keys()}

    print("Using device:", DEVICE)

    for epoch in range(1, EPOCHS + 1):
        # =========================
        #        TRAIN PHASE
        # =========================
        model.train()
        train_total_sum = 0.0

        # Per-attribute accumulator (train)
        train_reg_sums = {name: 0.0 for name in REG_ATTRS}
        train_cls_sums = {name: 0.0 for name in CLS_ATTRS.keys()}

        for imgs, y_reg, y_cls in train_loader:
            imgs = imgs.to(DEVICE)
            y_reg = y_reg.to(DEVICE)
            for k in y_cls.keys():
                y_cls[k] = y_cls[k].to(DEVICE)

            bs = imgs.size(0)

            optimizer.zero_grad()
            out = model(imgs)

            loss = 0.0

            # ---- SmoothL1 (train) ----
            if "reg" in out:
                pred_reg = out["reg"]  # (B, D)

                raw_loss = F.smooth_l1_loss(pred_reg, y_reg, reduction='none')
                per_dim_loss = raw_loss.mean(dim=0)
                loss_reg = per_dim_loss.sum()

                # Per-attribute MSE
                for i, name in enumerate(REG_ATTRS):
                    # We use .item() to grab the python number
                    train_reg_sums[name] += per_dim_loss[i].item() * bs

                loss = loss + loss_reg

            # ---- Classification (train) ----
            loss_cls_sum = 0.0
            for name, head_out in out["cls"].items():
                target = y_cls[name]
                loss_j = cls_crits[name](head_out, target)
                train_cls_sums[name] += loss_j.item() * bs
                loss_cls_sum += loss_j

            loss = loss + loss_cls_sum

            loss.backward()
            optimizer.step()

            train_total_sum += loss.item() * bs

        avg_train_total = train_total_sum / len(train_ds)
        avg_train_reg = {n: train_reg_sums[n] / len(train_ds) for n in REG_ATTRS}
        avg_train_cls = {n: train_cls_sums[n] / len(train_ds) for n in CLS_ATTRS.keys()}

        # =========================
        #        VAL PHASE
        # =========================
        model.eval()
        val_total_sum = 0.0
        val_reg_sums = {name: 0.0 for name in REG_ATTRS}
        val_cls_sums = {name: 0.0 for name in CLS_ATTRS.keys()}

        with torch.no_grad():
            for imgs, y_reg, y_cls in val_loader:
                imgs = imgs.to(DEVICE)
                y_reg = y_reg.to(DEVICE)
                for k in y_cls.keys():
                    y_cls[k] = y_cls[k].to(DEVICE)

                bs = imgs.size(0)

                out = model(imgs)

                loss = 0.0

                # ---- Regression (val) ----
                if "reg" in out:
                    pred_reg = out["reg"]
                    
                    raw_loss = F.smooth_l1_loss(pred_reg, y_reg, reduction='none')
                    per_dim_loss = raw_loss.mean(dim=0)
                    loss_reg = per_dim_loss.sum()

                    for i, name in enumerate(REG_ATTRS):
                        val_reg_sums[name] += per_dim_loss[i].item() * bs

                    loss = loss + loss_reg

                # ---- Classification (val) ----
                loss_cls_sum = 0.0
                for name, head_out in out["cls"].items():
                    target = y_cls[name]
                    loss_j = cls_crits[name](head_out, target)
                    val_cls_sums[name] += loss_j.item() * bs
                    loss_cls_sum += loss_j

                loss = loss + loss_cls_sum
                val_total_sum += loss.item() * bs

        avg_val_total = val_total_sum / len(val_ds)
        avg_val_reg = {n: val_reg_sums[n] / len(val_ds) for n in REG_ATTRS}
        avg_val_cls = {n: val_cls_sums[n] / len(val_ds) for n in CLS_ATTRS.keys()}

        # =========================
        #        PRINT TABLE
        # =========================
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")
        print(f"Total loss: train={avg_train_total:.4f}  val={avg_val_total:.4f}")

        # Regression attributes table
        print_loss_table(
            title="Reg Attr Losses",
            names=REG_ATTRS,
            train_losses=avg_train_reg,
            val_losses=avg_val_reg,
        )

        # Classification attributes table
        print_loss_table(
            title="Cls Attr Losses",
            names=list(CLS_ATTRS.keys()),
            train_losses=avg_train_cls,
            val_losses=avg_val_cls,
        )

    # Save model
    os.makedirs("attr_runs", exist_ok=True)
    save_path = os.path.join("attr_runs", "attrnet_fridge10.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "reg_attrs": REG_ATTRS,
            "cls_attrs": CLS_ATTRS,
            "stats_means": train_means,
            "stats_std": train_stds,
        },
        save_path,
    )
    print("[OK] Saved attr model to", save_path)


if __name__ == "__main__":
    train()
