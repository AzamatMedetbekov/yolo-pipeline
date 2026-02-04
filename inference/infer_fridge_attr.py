import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import yaml

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================
# Configuration
# ============================
# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Regression attribute scale/shift (must match values used during training)
# - attr3_internal_volume: Liters → divided by 1000 during training
# - attr8_year           : (year - 2000) / 50 during training
REG_SCALE = {
    "attr3_internal_volume": 1000.0,
    "attr8_year": 50.0,
}


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


def default_weights_from_config(config: dict) -> str:
    if config.get("weights"):
        return str(config["weights"])
    project = config.get("project", "runs/attr")
    name = config.get("name", "train")
    return os.path.join(project, name, "best.pt")


REG_SHIFT = {
    "attr3_internal_volume": 0.0,
    "attr8_year": 2000.0,
}

# Classification attribute index → human-readable string mapping
# CLS_ATTRS = {
#   "attr5_refrigerant": 3,
#   "attr6_door_type": 3,
#   "attr7_cabinet_type": 2,
#   "attr9_insulation_type": 2,
# }
CLS_LABELS = {
    "attr5_refrigerant": {
        0: "R404A",
        1: "R134a",
        2: "R290",
    },
    "attr6_door_type": {
        0: "Sliding",
        1: "Swing",
        2: "Open",
    },
    "attr7_cabinet_type": {
        0: "Vertical",
        1: "Face-to-face",
    },
    "attr9_insulation_type": {
        0: "PU Foam",
        1: "Other",
    },
}


# ============================
# AttrNet Definition (must match training)
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
            out["reg"] = self.reg_head(feat)  # (B, len(reg_attrs))

        cls_out = {}
        for name, head in self.cls_heads.items():
            cls_out[name] = head(feat)  # (B, C)
        out["cls"] = cls_out

        return out


# ============================
# Utility Functions
# ============================

def load_attr_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    reg_attrs: List[str] = ckpt["reg_attrs"]
    cls_attrs: Dict[str, int] = ckpt["cls_attrs"]
    stats_means = ckpt.get("stats_means")
    stats_std = ckpt.get("stats_std")

    model = AttrNet(reg_attrs, cls_attrs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Loaded AttrNet with:")
    print("  reg_attrs:", reg_attrs)
    print("  cls_attrs:", cls_attrs)

    return model, reg_attrs, cls_attrs, stats_means, stats_std


def build_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def preprocess_image(img_path: str, tfm):
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0)  # (1, C, H, W)
    return x, img


def unscale_regression(
    pred_reg: torch.Tensor,
    reg_attrs: List[str],
    stats_means: Dict[str, float] | None = None,
    stats_std: Dict[str, float] | None = None,
):
    """
    pred_reg: (D,) tensor (single sample)
    Returns: {attr_name: value in original units}
    """
    result = {}
    for i, name in enumerate(reg_attrs):
        v = pred_reg[i].item()
        if stats_means is not None and stats_std is not None:
            mu = stats_means.get(name, 0.0)
            sigma = stats_std.get(name, 1.0)
            v = v * sigma + mu
        elif name in REG_SCALE:
            shift = REG_SHIFT.get(name, 0.0)
            scale = REG_SCALE[name]
            v = v * scale + shift
        result[name] = v
    return result


def decode_classification(logits: torch.Tensor, attr_name: str):
    """
    logits: (C,) tensor (single sample)
    Returns: (pred_idx, pred_label, probs)
    """
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    label_map = CLS_LABELS.get(attr_name, {})
    label = label_map.get(idx, f"class_{idx}")
    return idx, label, probs.detach().cpu().numpy()


def infer_on_image(
    img_path: str,
    ckpt_path: str,
    device: torch.device,
    img_size: int,
):
    print(f"\n=== Inference on: {img_path} ===")

    # 1) Load model and transforms
    model, reg_attrs, cls_attrs, stats_means, stats_std = load_attr_model(ckpt_path, device)
    tfm = build_transform(img_size=img_size)

    # 2) Load and preprocess image
    x, raw_img = preprocess_image(img_path, tfm)
    x = x.to(device)

    # 3) Run inference
    with torch.no_grad():
        out = model(x)

    # 4) Process regression results (first sample)
    reg_results = {}
    if "reg" in out:
        pred_reg = out["reg"][0]  # (D,)
        reg_results = unscale_regression(pred_reg, reg_attrs, stats_means, stats_std)

    # 5) Process classification results
    cls_results = {}
    for name, logits in out["cls"].items():
        logits0 = logits[0]  # (C,)
        idx, label, probs = decode_classification(logits0, name)
        cls_results[name] = {
            "index": idx,
            "label": label,
            "probs": probs,
        }

    # 6) Print results
    print("\n[Regression attrs]")
    for k, v in reg_results.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Classification attrs]")
    for k, info in cls_results.items():
        print(f"  {k}: idx={info['index']}, label={info['label']}")

    return reg_results, cls_results


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/fridge_attr.yaml",
        help="Path to YAML config",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    def cfg(key: str, default):
        return config.get(key, default)

    default_weights = default_weights_from_config(config)

    parser = argparse.ArgumentParser(
        description="Inference for fridge attribute predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=cfg("weights", default_weights),
        help="Path to attribute checkpoint (best.pt from a run)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to an input image",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=cfg("img_size", 224),
        help="Input image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(cfg("device", "0")),
        help="Device to use (0, 1, cpu, etc.)",
    )

    args = parser.parse_args(remaining)

    if not args.image:
        print("[ERROR] Please provide --image for inference.")
        sys.exit(1)

    device_str = normalize_device(args.device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = resolve_path(args.weights)
    img_path = resolve_path(args.image)
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    infer_on_image(img_path, ckpt_path, device, args.imgsz)


if __name__ == "__main__":
    main()
