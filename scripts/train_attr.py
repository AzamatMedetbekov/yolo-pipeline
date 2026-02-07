import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import torch.cuda.amp as amp
import random
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================
# Configuration
# ============================
W_REG = 1
W_CLS = 1

# Define which attrs are regression / classification

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


def count_rows(csv_path: str, split: str) -> int:
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("split") != split:
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
        train_rows = count_rows(csv_path, "train")
        val_rows = count_rows(csv_path, "val")
        print(f"[INFO] CSV rows: train={train_rows}, val={val_rows}")
        if train_rows == 0 or val_rows == 0:
            print("[ERROR] CSV has no rows for train or val")
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
# Dataset
# ============================


class FridgeAttrDataset(Dataset):
    def __init__(self, csv_path: str, img_base_dir: str, split: str, stats=None, transform=None):
        self.transform = transform
        self.img_base_dir = img_base_dir
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
        img_path = os.path.join(self.img_base_dir, r["split"], r["image_name"])
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Failed to load {img_path}: {e}. Using placeholder.")
            # dummy image
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        # regression target
        reg_vals = []
        reg_mask = []

        for col in REG_ATTRS:
            raw_str = r.get(col, "").strip()

            if raw_str == "" or raw_str.lower() in ["na", "nan", "?", "none"]:
                reg_vals.append(0.0)
                reg_mask.append(0.0)
            else:
                try:
                    v = float(raw_str)
                    mu = self.means.get(col, 0.0)
                    sigma = self.stds.get(col, 1.0)

                    norm_v = (v - mu) / sigma

                    reg_vals.append(norm_v)
                    reg_mask.append(1.0)

                except ValueError:

                    reg_vals.append(0.0)
                    reg_mask.append(0.0)

        y_reg = torch.tensor(reg_vals, dtype=torch.float32)
        mask_reg = torch.tensor(reg_mask, dtype=torch.float32)

        # Classification targets: LongTensor per attribute
        y_cls = {}
        for col in CLS_ATTRS.keys():
            raw_str = r.get(col, "").strip()

            if raw_str == "" or raw_str.lower() in ["na", "nan", "?", "none"]:
                y_cls[col] = torch.tensor(-1, dtype=torch.long)
            else:
                try:
                    v = int(r[col])
                    y_cls[col] = torch.tensor(v, dtype=torch.long)
                except ValueError:
                    y_cls[col] = torch.tensor(-1, dtype=torch.long)

        return img, y_reg, mask_reg, y_cls


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

def filter_bad_images(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None 
    return torch.utils.data.dataloader.default_collate(batch)

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
    w_reg: float,
    w_cls: float,
):
    # ----- Dataset / DataLoader -----

    train_means, train_stds = calculate_dataset_stats(csv_path, split="train")
    stats_bundle = {"mean": train_means, "std": train_stds}

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
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
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm,
        ]
    )

    train_ds = FridgeAttrDataset(
        csv_path, img_base_dir, split="train", stats=stats_bundle, transform=train_tfm
    )

    val_ds = FridgeAttrDataset(
        csv_path, img_base_dir, split="val", stats=stats_bundle, transform=val_tfm
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn = filter_bad_images,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn = filter_bad_images,
    )

    model = AttrNet(REG_ATTRS, CLS_ATTRS).to(device)

    backbone_parameters = list(model.backbone.parameters())

    head_parameters = []
    if model.reg_head is not None:
        head_parameters.extend(model.reg_head.parameters())
    if len(model.cls_heads) > 0:
        head_parameters.extend(model.cls_heads.parameters())

    # slower learning rate for ResNet, because it is pretrained
    optimizer = torch.optim.AdamW([
        {'params': backbone_parameters, 'lr': lr*0.1},
        {'params': head_parameters, 'lr': lr},
    ], weight_decay=1e-2)

    print("Using device:", device)

    run_dir = build_run_dir(output_dir, run_name)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    log_path = os.path.join(run_dir, "train_log.csv")
    summary_path = os.path.join(run_dir, "metrics.json")

    best_val_loss = float("inf")
    counter = 0
    best_epoch = 0

    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = [lr*0.1, lr],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    cls_crits = {name: nn.CrossEntropyLoss(ignore_index=-1).to(device) for name in CLS_ATTRS}

    with open(log_path, "w", newline="", encoding="utf-8") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow([
            "epoch",
            "train_total",
            "val_total",
            "lr",
            "epoch_sec",
        ])

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        # =========================
        #        TRAIN PHASE
        # =========================
        model.train()
        train_total_sum = 0.0

        # Per-attribute accumulator (train)
        train_reg_sums = {name: 0.0 for name in REG_ATTRS}
        train_cls_sums = {name: 0.0 for name in CLS_ATTRS.keys()}

        for imgs, y_reg, mask_reg, y_cls in train_loader:

            imgs, y_reg, mask_reg = (
                imgs.to(device),
                y_reg.to(device),
                mask_reg.to(device),
            )

            for k in y_cls.keys():
                y_cls[k] = y_cls[k].to(device)

            bs = imgs.size(0)

            optimizer.zero_grad()
            loss = 0.0

            device_type = "cuda" if device.type == "cuda" else "cpu"

            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                out = model(imgs)

                # ---- SmoothL1 (train) ----
                loss_reg_sum = 0.0
                if "reg" in out:
                    pred_reg = out["reg"]  # (B, D)

                    raw_loss = F.smooth_l1_loss(pred_reg, y_reg, reduction="none")
                    masked_loss = raw_loss * mask_reg

                    valid_loss_sum = masked_loss.sum(dim=0)
                    valid_count = mask_reg.sum(0)

                    per_dim_loss = valid_loss_sum / valid_count.clamp(min=1.0)

                    loss_reg_sum = per_dim_loss.sum()

                    # Per-attribute MSE
                    for i, name in enumerate(REG_ATTRS):
                        # We use .item() to grab the python number
                        train_reg_sums[name] += per_dim_loss[i].item() * bs

                # ---- Classification (train) ----
                loss_cls_sum = 0.0
                num_cls_heads = len(out["cls"])

                if num_cls_heads > 0:
                    cls_losses_accumulated = 0

                    for name, head_out in out["cls"].items():
                        target = y_cls[name]
                        loss_j = cls_crits[name](head_out, target)

                        cls_losses_accumulated += loss_j

                        train_cls_sums[name] += loss_j.item() * bs

                    loss_cls_sum = cls_losses_accumulated

                loss = (w_reg * loss_reg_sum) + (w_cls * loss_cls_sum)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

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

        with torch.no_grad(), torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
            for imgs, y_reg, mask_reg, y_cls in val_loader:
                imgs, y_reg, mask_reg = (
                    imgs.to(device),
                    y_reg.to(device),
                    mask_reg.to(device),
                )
                for k in y_cls.keys():
                    y_cls[k] = y_cls[k].to(device)

                bs = imgs.size(0)

                out = model(imgs)

                loss = 0.0

                # ---- Regression (val) ----
                if "reg" in out:
                    pred_reg = out["reg"]

                    raw_loss = F.smooth_l1_loss(pred_reg, y_reg, reduction="none")
                    masked_loss = raw_loss * mask_reg

                    valid_loss_sum = masked_loss.sum(dim=0)
                    valid_count = mask_reg.sum(0)

                    per_dim_loss = valid_loss_sum / valid_count.clamp(min=1.0)
                    loss_reg = per_dim_loss.sum()

                    for i, name in enumerate(REG_ATTRS):
                        val_reg_sums[name] += per_dim_loss[i].item() * bs

                    loss = loss + (loss_reg * w_reg)

                # ---- Classification (val) ----
                loss_cls_sum = 0.0
                num_cls_heads = len(out["cls"])
                loss_cls_accumulated = 0.0

                if num_cls_heads > 0:
                    for name, head_out in out["cls"].items():
                        target = y_cls[name]
                        loss_j = cls_crits[name](head_out, target)

                        loss_cls_accumulated += loss_j

                        val_cls_sums[name] += loss_j.item() * bs

                    loss_cls_sum = loss_cls_accumulated
                    loss = loss + (loss_cls_sum * w_cls)

                val_total_sum += loss.item() * bs

        avg_val_total = val_total_sum / len(val_ds)
        avg_val_reg = {n: val_reg_sums[n] / len(val_ds) for n in REG_ATTRS}
        avg_val_cls = {n: val_cls_sums[n] / len(val_ds) for n in CLS_ATTRS.keys()}

        # =========================
        #        PRINT TABLE
        # =========================
        lr_now = scheduler.get_last_lr()[0]
        epoch_sec = time.perf_counter() - epoch_start

        print(f"\n========== Epoch {epoch}/{epochs} ==========")
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

        # Save Model
        if avg_val_total < best_val_loss:
            diff = best_val_loss - avg_val_total
            print(f"[INFO] Validation Loss improved by {diff:.4f}. Saving best model...")

            best_val_loss = avg_val_total
            counter = 0
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                   #"optimizer_state": optimizer.state_dict(), removed because we do not need the training trace for the best one
                    "reg_attrs": REG_ATTRS,
                    "cls_attrs": CLS_ATTRS,
                    "stats_means": train_means,
                    "stats_std": train_stds,
                    "best_loss": best_val_loss,
                },
                best_path,
            )

        else:
            counter += 1
            print(f"[INFO] No improvement. Patience: {counter}/{patience}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "reg_attrs": REG_ATTRS,
                "cls_attrs": CLS_ATTRS,
                "stats_means": train_means,
                "stats_std": train_stds,
                "best_loss": best_val_loss,
            },
            last_path,
        )

        with open(log_path, "a", newline="", encoding="utf-8") as log_f:
            log_writer = csv.writer(log_f)
            log_writer.writerow([
                epoch,
                f"{avg_train_total:.6f}",
                f"{avg_val_total:.6f}",
                f"{lr_now:.8f}",
                f"{epoch_sec:.3f}",
            ])

        if counter >= patience:
            print(f"\n[STOP] Early stopping triggered! Best loss was {best_val_loss:.4f}")
            break

    with open(summary_path, "w", encoding="utf-8") as summary_f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_ran": epoch,
                "output_dir": run_dir,
                "best_checkpoint": best_path,
                "last_checkpoint": last_path,
            },
            summary_f,
            indent=2,
        )

    print(f"\n[OK] Training finished. Best val loss: {best_val_loss:.4f}")
    print(f"[OK] Outputs saved under: {run_dir}")


def main():

    set_seed(42)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/fridge_attr.yaml",
        help="Path to YAML config",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train fridge attribute predictor",
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
        help="Override CSV labels file with split/image_name/attr columns",
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
    parser.add_argument(
        "--w-reg",
        type=float,
        default=None,
        help="Override weight for regression loss",
    )
    parser.add_argument(
        "--w-cls",
        type=float,
        default=None,
        help="Override weight for classification loss",
    )

    args = parser.parse_args(remaining)

    def pick(arg_value, key: str, default):
        if arg_value is not None:
            return arg_value
        return config.get(key, default)

    global REG_ATTRS
    global CLS_ATTRS
    if "reg_attrs" in config:
        REG_ATTRS = list(config["reg_attrs"])
    if "cls_attrs" in config:
        CLS_ATTRS = {k: int(v) for k, v in config["cls_attrs"].items()}

    data_value = pick(args.data, "data", "data")
    csv_value = pick(args.csv, "csv", "data/fridge_attr10_labels.csv")
    epochs_value = int(pick(args.epochs, "epochs", 200))
    batch_value = int(pick(args.batch, "batch", 4))
    imgsz_value = int(pick(args.imgsz, "img_size", IMG_SIZE))
    lr_value = float(pick(args.lr, "lr", LR))
    device_value = str(pick(args.device, "device", "0"))
    output_value = pick(args.output, "output", "runs/attr")
    name_value = pick(args.name, "name", "")
    weights_cfg = config.get("weights", {}) if isinstance(config.get("weights", {}), dict) else {}
    w_reg_value = float(weights_cfg.get("reg", W_REG))
    w_cls_value = float(weights_cfg.get("cls", W_CLS))
    if args.w_reg is not None:
        w_reg_value = float(args.w_reg)
    if args.w_cls is not None:
        w_cls_value = float(args.w_cls)
    patience_value = int(config.get("patience", 15))
    workers_value = int(config.get("workers", 0))

    data_dir = resolve_path(data_value)
    csv_path = resolve_path(csv_value)
    output_dir = resolve_path(output_value)
    img_base_dir = os.path.join(data_dir, "images")

    device_str = normalize_device(device_value)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("Fridge Attribute Training")
    print("=" * 50)
    print(f"Data Dir: {data_dir}")
    print(f"CSV     : {csv_path}")
    print(f"Epochs  : {epochs_value}")
    print(f"Batch   : {batch_value}")
    print(f"ImgSize : {imgsz_value}")
    print(f"LR      : {lr_value}")
    print(f"Device  : {device}")
    print(f"Output  : {output_dir}")
    print(f"Name    : {name_value or '(auto)'}")
    print("=" * 50)

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
        w_reg=w_reg_value,
        w_cls=w_cls_value,
    )


if __name__ == "__main__":
    main()
