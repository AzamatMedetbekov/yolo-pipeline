#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import yaml
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.cuda.amp as amp

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================
# Global Configuration
# ============================
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

NUM_TYPE_CLASSES = 6
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


# ============================
# Utility Functions
# ============================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path_str: str) -> str:
    """Resolve path relative to project root."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path_str)


def normalize_device(device_str: str) -> str:
    """Normalize device string."""
    if device_str.lower() == "cpu":
        return "cpu"
    if device_str.isdigit():
        return f"cuda:{device_str}"
    return device_str


def load_yaml_config(path_str: str) -> dict:
    """Load YAML configuration file."""
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
    """Create output directory with timestamp."""
    if not name:
        name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(output_dir, name)
    if os.path.exists(run_dir):
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"{run_dir}_{suffix}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def validate_dataset(data_dir: str, csv_path: str, mode: str) -> bool:
    """Validate dataset structure."""
    img_base_dir = os.path.join(data_dir, "images")
    train_dir = os.path.join(img_base_dir, "train")
    val_dir = os.path.join(img_base_dir, "val")

    valid = True

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
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


def calculate_dataset_stats(csv_path: str, split: str = "train") -> Tuple[dict, dict]:
    """Calculate mean and std for regression attributes."""
    print(f"[INFO] Calculating regression stats from {csv_path} ({split})...")

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
                except (ValueError, KeyError):
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

    print(f"[INFO] Calculated means: {mean}")
    print(f"[INFO] Calculated stds:  {std}")
    return mean, std


# ============================
# Model
# ============================

class FridgeNet(nn.Module):
    """Unified model supporting attribute prediction and type classification."""
    
    def __init__(
        self,
        mode: str = 'both',
        reg_attrs: Optional[List[str]] = None,
        cls_attrs: Optional[Dict[str, int]] = None,
        num_type_classes: int = 6
    ):
        super().__init__()
        self.mode = mode
        
        # Shared backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # Attribute heads
        self.reg_head = None
        self.cls_heads = None
        
        if mode in ['attr', 'both']:
            if reg_attrs and len(reg_attrs) > 0:
                self.reg_head = nn.Linear(in_feat, len(reg_attrs))
            
            if cls_attrs:
                self.cls_heads = nn.ModuleDict()
                for name, num_classes in cls_attrs.items():
                    self.cls_heads[name] = nn.Linear(in_feat, num_classes)
        
        # Type head
        self.type_head = None
        if mode in ['type', 'both']:
            self.type_head = nn.Linear(in_feat, num_type_classes)
    
    def forward(self, x):
        """Forward pass through conditional heads."""
        feat = self.backbone(x)
        outputs = {}
        
        # Attribute outputs
        if self.reg_head is not None:
            outputs['reg'] = self.reg_head(feat)
        
        if self.cls_heads is not None:
            outputs['cls_attr'] = {}
            for name, head in self.cls_heads.items():
                outputs['cls_attr'][name] = head(feat)
        
        # Type output
        if self.type_head is not None:
            outputs['type'] = self.type_head(feat)
        
        return outputs


# ============================
# Dataset
# ============================

class FridgeDataset(Dataset):
    """Unified dataset supporting all label types."""
    
    def __init__(
        self,
        csv_path: str,
        img_base_dir: str,
        split: str,
        mode: str = 'both',
        stats: Optional[dict] = None,
        transform = None
    ):
        self.mode = mode
        self.transform = transform
        self.img_base_dir = img_base_dir
        self.rows = []
        
        # Load rows based on mode
        with open(csv_path, 'r', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r['split'] != split:
                    continue
                
                has_attrs = self._has_attr_labels(r)
                has_type = self._has_type_label(r)
                
                if mode == 'attr' and has_attrs:
                    self.rows.append(r)
                elif mode == 'type' and has_type:
                    self.rows.append(r)
                elif mode == 'both' and (has_attrs or has_type):
                    self.rows.append(r)
        
        if not self.rows:
            raise RuntimeError(
                f"No valid rows for mode='{mode}' split='{split}' in {csv_path}"
            )
        
        # Initialize stats for regression
        if mode in ['attr', 'both']:
            if stats is None:
                self.means = {k: 0.0 for k in REG_ATTRS}
                self.stds = {k: 1.0 for k in REG_ATTRS}
            else:
                self.means = stats["mean"]
                self.stds = stats["std"]
        else:
            self.means = {}
            self.stds = {}
    
    def _has_attr_labels(self, row):
        """Check if row has any attribute labels."""
        for col in REG_ATTRS + list(CLS_ATTRS.keys()):
            val = row.get(col, '').strip()
            if val and val.lower() not in ['na', 'nan', '?', 'none', '']:
                return True
        return False
    
    def _has_type_label(self, row):
        """Check if row has type label."""
        val = row.get('type_class', '').strip()
        return val and val.lower() not in ['na', 'nan', '?', 'none', '']
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = os.path.join(self.img_base_dir, r["split"], r["image_name"])
        
        # Load image with error handling
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Failed to load {img_path}: {e}. Using placeholder.")
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            img = self.transform(img)
        
        # Build labels dictionary
        labels = {}
        
        if self.mode in ['attr', 'both']:
            labels.update(self._parse_attr_labels(r))
        
        if self.mode in ['type', 'both']:
            labels['type'] = self._parse_type_label(r)
        
        return img, labels
    
    def _parse_attr_labels(self, row):
        """Parse regression and classification attributes."""
        # Regression values and masks
        reg_vals = []
        reg_mask = []
        
        for col in REG_ATTRS:
            raw_str = row.get(col, '').strip()
            
            if raw_str == '' or raw_str.lower() in ['na', 'nan', '?', 'none']:
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
        
        # Classification attributes
        y_cls = {}
        for col in CLS_ATTRS.keys():
            raw_str = row.get(col, '').strip()
            
            if raw_str == '' or raw_str.lower() in ['na', 'nan', '?', 'none']:
                y_cls[col] = torch.tensor(-1, dtype=torch.long)
            else:
                try:
                    v = int(row[col])
                    y_cls[col] = torch.tensor(v, dtype=torch.long)
                except ValueError:
                    y_cls[col] = torch.tensor(-1, dtype=torch.long)
        
        return {
            'reg': y_reg,
            'reg_mask': mask_reg,
            'cls_attr': y_cls
        }
    
    def _parse_type_label(self, row):
        """Parse type classification label."""
        raw_str = row.get('type_class', '').strip()
        
        if raw_str == '' or raw_str.lower() in ['na', 'nan', '?', 'none']:
            return torch.tensor(-1, dtype=torch.long)
        
        try:
            type_class = int(raw_str)
            # Validate range
            if type_class < 0 or type_class >= NUM_TYPE_CLASSES:
                raise ValueError(
                    f"Invalid type_class={type_class} in {row.get('image_name', 'unknown')}. "
                    f"Must be in range [0, {NUM_TYPE_CLASSES-1}]"
                )
            return torch.tensor(type_class, dtype=torch.long)
        except ValueError as e:
            print(f"[WARNING] Invalid type_class: {e}")
            return torch.tensor(-1, dtype=torch.long)


# ============================
# Training Functions
# ============================

def print_loss_table(title, names, train_losses, val_losses):
    """Print formatted loss table."""
    print(f"\n[{title}]")
    col1_w = max(len("attr"), max(len(n) for n in names))
    header = f"{'attr'.ljust(col1_w)} | {'train':>10} | {'val':>10}"
    print(header)
    print("-" * len(header))
    for name in names:
        t = train_losses[name]
        v = val_losses[name]
        print(f"{name.ljust(col1_w)} | {t:10.4f} | {v:10.4f}")


def train_epoch(
    model, loader, optimizer, scheduler, scaler, device, mode,
    w_reg, w_cls_attr, w_type, cls_attr_crits, type_crit
):
    """Run one training epoch."""
    model.train()
    total_loss_sum = 0.0
    total_samples = 0
    
    # Per-task metrics
    metrics = {
        'reg_sums': {name: 0.0 for name in REG_ATTRS} if mode in ['attr', 'both'] else {},
        'cls_attr_sums': {name: 0.0 for name in CLS_ATTRS.keys()} if mode in ['attr', 'both'] else {},
        'type_loss_sum': 0.0 if mode in ['type', 'both'] else 0.0,
        'type_correct': 0 if mode in ['type', 'both'] else 0,
        'type_total': 0 if mode in ['type', 'both'] else 0,
    }
    
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    for batch_data in loader:
        imgs, labels = batch_data
        imgs = imgs.to(device)
        
        bs = imgs.size(0)
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = 0.0
            
            # Attribute losses
            if mode in ['attr', 'both'] and 'reg' in labels:
                y_reg = labels['reg'].to(device)
                mask_reg = labels['reg_mask'].to(device)
                
                if 'reg' in outputs:
                    pred_reg = outputs['reg']
                    raw_loss = F.smooth_l1_loss(pred_reg, y_reg, reduction="none")
                    masked_loss = raw_loss * mask_reg
                    valid_loss_sum = masked_loss.sum(dim=0)
                    valid_count = mask_reg.sum(0).clamp(min=1.0)
                    per_dim_loss = valid_loss_sum / valid_count
                    loss_reg = per_dim_loss.sum()
                    loss += w_reg * loss_reg
                    
                    for i, name in enumerate(REG_ATTRS):
                        metrics['reg_sums'][name] += per_dim_loss[i].item() * bs
                
                if 'cls_attr' in outputs and 'cls_attr' in labels:
                    y_cls = labels['cls_attr']
                    for name, head_out in outputs['cls_attr'].items():
                        target = y_cls[name].to(device)
                        loss_j = cls_attr_crits[name](head_out, target)
                        loss += w_cls_attr * loss_j
                        metrics['cls_attr_sums'][name] += loss_j.item() * bs
            
            # Type loss
            if mode in ['type', 'both'] and 'type' in labels:
                y_type = labels['type'].to(device)
                
                if 'type' in outputs:
                    logits = outputs['type']
                    loss_type = type_crit(logits, y_type)
                    loss += w_type * loss_type
                    
                    metrics['type_loss_sum'] += loss_type.item() * bs
                    
                    # Accuracy
                    valid_mask = y_type != -1
                    if valid_mask.sum() > 0:
                        preds = logits.argmax(dim=1)
                        metrics['type_correct'] += (preds[valid_mask] == y_type[valid_mask]).sum().item()
                        metrics['type_total'] += valid_mask.sum().item()
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss_sum += loss.item() * bs
        total_samples += bs
    
    # Compute averages
    avg_total = total_loss_sum / total_samples if total_samples > 0 else 0.0
    
    avg_metrics = {'total_loss': avg_total}
    
    if mode in ['attr', 'both']:
        avg_metrics['reg'] = {n: metrics['reg_sums'][n] / total_samples for n in REG_ATTRS}
        avg_metrics['cls_attr'] = {n: metrics['cls_attr_sums'][n] / total_samples for n in CLS_ATTRS.keys()}
    
    if mode in ['type', 'both']:
        avg_metrics['type_loss'] = metrics['type_loss_sum'] / total_samples if total_samples > 0 else 0.0
        avg_metrics['type_acc'] = metrics['type_correct'] / metrics['type_total'] if metrics['type_total'] > 0 else 0.0
    
    return avg_metrics


def validate_epoch(model, loader, device, mode, w_reg, w_cls_attr, w_type, cls_attr_crits, type_crit):
    """Run one validation epoch."""
    model.eval()
    total_loss_sum = 0.0
    total_samples = 0
    
    metrics = {
        'reg_sums': {name: 0.0 for name in REG_ATTRS} if mode in ['attr', 'both'] else {},
        'cls_attr_sums': {name: 0.0 for name in CLS_ATTRS.keys()} if mode in ['attr', 'both'] else {},
        'type_loss_sum': 0.0 if mode in ['type', 'both'] else 0.0,
        'type_correct': 0 if mode in ['type', 'both'] else 0,
        'type_total': 0 if mode in ['type', 'both'] else 0,
    }
    
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device.type == "cuda")):
        for batch_data in loader:
            imgs, labels = batch_data
            imgs = imgs.to(device)
            
            bs = imgs.size(0)
            outputs = model(imgs)
            loss = 0.0
            
            # Attribute losses
            if mode in ['attr', 'both'] and 'reg' in labels:
                y_reg = labels['reg'].to(device)
                mask_reg = labels['reg_mask'].to(device)
                
                if 'reg' in outputs:
                    pred_reg = outputs['reg']
                    raw_loss = F.smooth_l1_loss(pred_reg, y_reg, reduction="none")
                    masked_loss = raw_loss * mask_reg
                    valid_loss_sum = masked_loss.sum(dim=0)
                    valid_count = mask_reg.sum(0).clamp(min=1.0)
                    per_dim_loss = valid_loss_sum / valid_count
                    loss_reg = per_dim_loss.sum()
                    loss += w_reg * loss_reg
                    
                    for i, name in enumerate(REG_ATTRS):
                        metrics['reg_sums'][name] += per_dim_loss[i].item() * bs
                
                if 'cls_attr' in outputs and 'cls_attr' in labels:
                    y_cls = labels['cls_attr']
                    for name, head_out in outputs['cls_attr'].items():
                        target = y_cls[name].to(device)
                        loss_j = cls_attr_crits[name](head_out, target)
                        loss += w_cls_attr * loss_j
                        metrics['cls_attr_sums'][name] += loss_j.item() * bs
            
            # Type loss
            if mode in ['type', 'both'] and 'type' in labels:
                y_type = labels['type'].to(device)
                
                if 'type' in outputs:
                    logits = outputs['type']
                    loss_type = type_crit(logits, y_type)
                    loss += w_type * loss_type
                    
                    metrics['type_loss_sum'] += loss_type.item() * bs
                    
                    valid_mask = y_type != -1
                    if valid_mask.sum() > 0:
                        preds = logits.argmax(dim=1)
                        metrics['type_correct'] += (preds[valid_mask] == y_type[valid_mask]).sum().item()
                        metrics['type_total'] += valid_mask.sum().item()
            
            total_loss_sum += loss.item() * bs
            total_samples += bs
    
    # Compute averages
    avg_total = total_loss_sum / total_samples if total_samples > 0 else 0.0
    
    avg_metrics = {'total_loss': avg_total}
    
    if mode in ['attr', 'both']:
        avg_metrics['reg'] = {n: metrics['reg_sums'][n] / total_samples for n in REG_ATTRS}
        avg_metrics['cls_attr'] = {n: metrics['cls_attr_sums'][n] / total_samples for n in CLS_ATTRS.keys()}
    
    if mode in ['type', 'both']:
        avg_metrics['type_loss'] = metrics['type_loss_sum'] / total_samples if total_samples > 0 else 0.0
        avg_metrics['type_acc'] = metrics['type_correct'] / metrics['type_total'] if metrics['type_total'] > 0 else 0.0
    
    return avg_metrics


def train(
    csv_path: str,
    img_base_dir: str,
    mode: str,
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
    w_cls_attr: float,
    w_type: float,
):
    """Main training function."""
    set_seed(42)
    
    # Calculate stats if needed
    stats_bundle = None
    if mode in ['attr', 'both']:
        train_means, train_stds = calculate_dataset_stats(csv_path, split='train')
        stats_bundle = {'mean': train_means, 'std': train_stds}
    
    # Build transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        transforms.ToTensor(),
        norm,
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])
    
    # Create datasets
    train_ds = FridgeDataset(
        csv_path, img_base_dir, split='train',
        mode=mode, stats=stats_bundle, transform=train_tfm
    )
    val_ds = FridgeDataset(
        csv_path, img_base_dir, split='val',
        mode=mode, stats=stats_bundle, transform=val_tfm
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=device.type == 'cuda',
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == 'cuda',
        persistent_workers=workers > 0,
    )
    
    print(f"\n[INFO] Mode: {mode}")
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Create model
    model = FridgeNet(
        mode=mode,
        reg_attrs=REG_ATTRS if mode in ['attr', 'both'] else None,
        cls_attrs=CLS_ATTRS if mode in ['attr', 'both'] else None,
        num_type_classes=NUM_TYPE_CLASSES if mode in ['type', 'both'] else None,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = []
    if model.reg_head:
        head_params.extend(model.reg_head.parameters())
    if model.cls_heads:
        head_params.extend(model.cls_heads.parameters())
    if model.type_head:
        head_params.extend(model.type_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params, 'lr': lr},
    ], weight_decay=1e-2)
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr * 0.1, lr],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    
    # Loss criteria
    cls_attr_crits = None
    type_crit = None
    
    if mode in ['attr', 'both']:
        cls_attr_crits = {
            name: nn.CrossEntropyLoss(ignore_index=-1).to(device)
            for name in CLS_ATTRS.keys()
        }
    
    if mode in ['type', 'both']:
        type_crit = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    
    # AMP scaler
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Setup output
    run_dir = build_run_dir(output_dir, run_name)
    best_path = os.path.join(run_dir, 'best.pt')
    last_path = os.path.join(run_dir, 'last.pt')
    log_path = os.path.join(run_dir, 'train_log.csv')
    summary_path = os.path.join(run_dir, 'metrics.json')
    
    # Initialize CSV log
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_loss', 'val_loss']
        if mode in ['type', 'both']:
            header.extend(['train_type_acc', 'val_type_acc'])
        writer.writerow(header)
    
    # Training loop
    best_val_metric = float('inf') if mode in ['attr'] else -1.0
    counter = 0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, mode,
            w_reg, w_cls_attr, w_type, cls_attr_crits, type_crit
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, device, mode,
            w_reg, w_cls_attr, w_type, cls_attr_crits, type_crit
        )
        
        epoch_sec = time.perf_counter() - epoch_start
        lr_now = scheduler.get_last_lr()[0]
        
        # Print metrics
        print(f"\n========== Epoch {epoch}/{epochs} ==========")
        print(f"Total loss: train={train_metrics['total_loss']:.4f}  val={val_metrics['total_loss']:.4f}")
        
        if mode in ['attr', 'both']:
            print_loss_table("Reg Attr Losses", REG_ATTRS, train_metrics['reg'], val_metrics['reg'])
            print_loss_table("Cls Attr Losses", list(CLS_ATTRS.keys()), 
                           train_metrics['cls_attr'], val_metrics['cls_attr'])
        
        if mode in ['type', 'both']:
            print(f"Type: train_acc={train_metrics['type_acc']:.4f}, val_acc={val_metrics['type_acc']:.4f}")
        
        print(f"LR: {lr_now:.6f}, Time: {epoch_sec:.1f}s")
        
        # Log to CSV
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [epoch, f"{train_metrics['total_loss']:.6f}", f"{val_metrics['total_loss']:.6f}"]
            if mode in ['type', 'both']:
                row.extend([f"{train_metrics['type_acc']:.6f}", f"{val_metrics['type_acc']:.6f}"])
            writer.writerow(row)
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'mode': mode,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'reg_attrs': REG_ATTRS if mode in ['attr', 'both'] else None,
            'cls_attrs': CLS_ATTRS if mode in ['attr', 'both'] else None,
            'type_labels': TYPE_LABELS if mode in ['type', 'both'] else None,
            'stats_means': train_means if mode in ['attr', 'both'] else None,
            'stats_std': train_stds if mode in ['attr', 'both'] else None,
            'best_metric': best_val_metric,
        }, last_path)
        
        # Check improvement
        current_val_metric = val_metrics['type_acc'] if mode == 'type' else val_metrics['total_loss']
        improved = current_val_metric > best_val_metric if mode == 'type' else current_val_metric < best_val_metric
        
        if improved:
            best_val_metric = current_val_metric
            best_epoch = epoch
            counter = 0
            
            torch.save({
                'epoch': epoch,
                'mode': mode,
                'model_state': model.state_dict(),
                'reg_attrs': REG_ATTRS if mode in ['attr', 'both'] else None,
                'cls_attrs': CLS_ATTRS if mode in ['attr', 'both'] else None,
                'type_labels': TYPE_LABELS if mode in ['type', 'both'] else None,
                'stats_means': train_means if mode in ['attr', 'both'] else None,
                'stats_std': train_stds if mode in ['attr', 'both'] else None,
                'best_metric': best_val_metric,
            }, best_path)
            
            print(f"[INFO] New best model saved!")
        else:
            counter += 1
            print(f"[INFO] No improvement. Patience: {counter}/{patience}")
        
        if counter >= patience:
            print(f"\n[STOP] Early stopping triggered!")
            break
    
    # Save summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': mode,
            'best_val_metric': best_val_metric,
            'best_epoch': best_epoch,
            'epochs_ran': epoch,
            'output_dir': run_dir,
            'best_checkpoint': best_path,
            'last_checkpoint': last_path,
        }, f, indent=2)
    
    print(f"\n[OK] Training finished. Best val metric: {best_val_metric:.4f}")
    print(f"[OK] Outputs saved under: {run_dir}")


# ============================
# Main
# ============================

def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser(
        description='Unified Fridge Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str, choices=['attr', 'type', 'both'], default='both',
        help='Training mode: attr, type, or both'
    )
    
    # Data
    parser.add_argument('--data', type=str, default='data', help='Data directory')
    parser.add_argument('--csv', type=str, default='data/labels.csv', help='CSV labels file')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    
    # Loss weights
    parser.add_argument('--w-reg', type=float, default=1.0, help='Weight for regression loss')
    parser.add_argument('--w-cls-attr', type=float, default=1.0, help='Weight for cls attr loss')
    parser.add_argument('--w-type', type=float, default=2.0, help='Weight for type loss')
    
    # Device & optimization
    parser.add_argument('--device', type=str, default='0', help='Device (0, 1, cpu, etc.)')
    
    # Output
    parser.add_argument('--output', type=str, default='runs/fridge', help='Output directory')
    parser.add_argument('--name', type=str, default='', help='Run name')
    
    # Config file
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_yaml_config(args.config)
    
    # Resolve paths
    data_dir = resolve_path(args.data)
    csv_path = resolve_path(args.csv)
    img_base_dir = os.path.join(data_dir, 'images')
    
    device_str = normalize_device(args.device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    print('=' * 60)
    print('Fridge Unified Training')
    print('=' * 60)
    print(f'Mode    : {args.mode}')
    print(f'Data Dir: {data_dir}')
    print(f'CSV     : {csv_path}')
    print(f'Epochs  : {args.epochs}')
    print(f'Batch   : {args.batch}')
    print(f'Device  : {device}')
    if args.mode == 'both':
        print(f'Weights : reg={args.w_reg}, cls_attr={args.w_cls_attr}, type={args.w_type}')
    print('=' * 60)
    
    if not validate_dataset(data_dir, csv_path, args.mode):
        print('[ERROR] Dataset validation failed. Exiting.')
        sys.exit(1)
    
    print('\n[INFO] Starting training...')
    train(
        csv_path=csv_path,
        img_base_dir=img_base_dir,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch,
        device=device,
        output_dir=resolve_path(args.output),
        run_name=args.name,
        patience=args.patience,
        img_size=args.imgsz,
        lr=args.lr,
        workers=args.workers,
        w_reg=args.w_reg,
        w_cls_attr=args.w_cls_attr,
        w_type=args.w_type,
    )


if __name__ == '__main__':
    main()

# Train only attributes
# python train_fridge.py --mode attr --data ./data --csv data/labels.csv --epochs 100

# # Train only type
# python train_fridge.py --mode type --data ./data --csv data/labels.csv --epochs 50

# # Train both (multi-task)
# python train_fridge.py --mode both --data ./data --csv data/labels.csv \
#   --epochs 100 --w-reg 1.0 --w-cls-attr 1.0 --w-type 2.0