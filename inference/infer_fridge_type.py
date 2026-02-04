"""
Inference script for 6-class refrigerator type classifier.

Type classes:
    0: vertical        - Normal refrigerator (drink freezer)
    1: horizontal      - Horizontal (ice cream freezer)
    2: vertical_open   - Vertical without door
    3: horizontal_open - Horizontal without door
    4: combination     - Two types combined
    5: coldroom        - Walk-in cold room
"""
import argparse
import os
import sys
from pathlib import Path
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

# Type labels (same as training)
TYPE_LABELS = {
    0: "vertical",
    1: "horizontal",
    2: "vertical_open",
    3: "horizontal_open",
    4: "combination",
    5: "coldroom",
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
    project = config.get("project", "runs/type")
    name = config.get("name", "train")
    return os.path.join(project, name, "best.pt")


# ============================
# Model: TypeNet (same as train)
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
# Utility Functions
# ============================

def load_type_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt.get("num_classes", 6)
    type_labels = ckpt.get("type_labels", TYPE_LABELS)

    model = TypeNet(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded TypeNet with {num_classes} classes")

    return model, type_labels


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


def infer_on_image(img_path: str, ckpt_path: str, device: torch.device, img_size: int):
    print(f"\n=== Inference on: {img_path} ===")

    # 1) Load model
    model, type_labels = load_type_model(ckpt_path, device)
    tfm = build_transform(img_size=img_size)

    # 2) Load and preprocess image
    x, raw_img = preprocess_image(img_path, tfm)
    x = x.to(device)

    # 3) Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = logits.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()

    pred_label = type_labels.get(pred_idx, f"class_{pred_idx}")

    # 4) Print results
    print(f"\nPredicted class: {pred_idx}")
    print(f"Predicted label: {pred_label}")
    print(f"Confidence: {pred_prob:.4f}")

    print("\nAll class probabilities:")
    for i, p in enumerate(probs[0].cpu().numpy()):
        label = type_labels.get(i, f"class_{i}")
        print(f"  {i}: {label:15s} - {p:.4f}")

    return pred_idx, pred_label, probs[0].cpu().numpy()


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/fridge_type.yaml",
        help="Path to YAML config",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    def cfg(key: str, default):
        return config.get(key, default)

    default_weights = default_weights_from_config(config)

    parser = argparse.ArgumentParser(
        description="Inference for fridge type classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=cfg("weights", default_weights),
        help="Path to type checkpoint (best.pt from a run)",
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

    global TYPE_LABELS
    if "type_labels" in config:
        TYPE_LABELS = {int(k): v for k, v in config["type_labels"].items()}

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
