import os
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================
# Settings
# ============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Checkpoint path saved during training
ATTR_CKPT = "attr_runs/attrnet_fridge10.pt"

# Regression attr scale/shift (must match training values)
# - attr3_internal_volume: L units -> trained after dividing by 1000
# - attr8_year           : trained with (year - 2000) / 50
REG_SCALE = {
    "attr3_internal_volume": 1000.0,
    "attr8_year": 50.0,
}
REG_SHIFT = {
    "attr3_internal_volume": 0.0,
    "attr8_year": 2000.0,
}

# Classification attr index -> human-readable string mapping
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
        2: "Open type",
    },
    "attr7_cabinet_type": {
        0: "Vertical",
        1: "Island",
    },
    "attr9_insulation_type": {
        0: "PU foam",
        1: "Other",
    },
}


# ============================
# AttrNet definition (must match training)
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
# Utility functions
# ============================

def load_attr_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    reg_attrs: List[str] = ckpt["reg_attrs"]
    cls_attrs: Dict[str, int] = ckpt["cls_attrs"]

    model = AttrNet(reg_attrs, cls_attrs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Loaded AttrNet with:")
    print("  reg_attrs:", reg_attrs)
    print("  cls_attrs:", cls_attrs)

    return model, reg_attrs, cls_attrs


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


def unscale_regression(pred_reg: torch.Tensor, reg_attrs: List[str]):
    """
    pred_reg: (D,) tensor (one sample)
    Return: {attr_name: value in original units}
    """
    result = {}
    for i, name in enumerate(reg_attrs):
        v = pred_reg[i].item()
        if name in REG_SCALE:
            shift = REG_SHIFT.get(name, 0.0)
            scale = REG_SCALE[name]
            v = v * scale + shift
        result[name] = v
    return result


def decode_classification(logits: torch.Tensor, attr_name: str):
    """
    logits: (C,) tensor (one sample)
    Returns: (pred_idx, pred_label, probs)
    """
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    label_map = CLS_LABELS.get(attr_name, {})
    label = label_map.get(idx, f"class_{idx}")
    return idx, label, probs.detach().cpu().numpy()


def infer_on_image(img_path: str):
    print(f"\n=== Inference on: {img_path} ===")

    # 1) Load model and transforms
    model, reg_attrs, cls_attrs = load_attr_model(ATTR_CKPT, DEVICE)
    tfm = build_transform(img_size=224)

    # 2) Load image + preprocess
    x, raw_img = preprocess_image(img_path, tfm)
    x = x.to(DEVICE)

    # 3) Inference
    with torch.no_grad():
        out = model(x)

    # 4) Regression results (first sample)
    reg_results = {}
    if "reg" in out:
        pred_reg = out["reg"][0]  # (D,)
        reg_results = unscale_regression(pred_reg, reg_attrs)

    # 5) Classification results
    cls_results = {}
    for name, logits in out["cls"].items():
        logits0 = logits[0]  # (C,)
        idx, label, probs = decode_classification(logits0, name)
        cls_results[name] = {
            "index": idx,
            "label": label,
            "probs": probs,
        }

    # 6) Pretty print output
    print("\n[Regression attrs]")
    for k, v in reg_results.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Classification attrs]")
    for k, info in cls_results.items():
        print(f"  {k}: idx={info['index']}, label={info['label']}")

    return reg_results, cls_results


if __name__ == "__main__":
    # Example test path (replace with your image file)
    test_img = "yolov12/data/fridge_attr10/images/val/test8.jpg"
    if not os.path.exists(test_img):
        print("⚠ Please change test_img to a path that actually exists.")
    else:
        infer_on_image(test_img)
