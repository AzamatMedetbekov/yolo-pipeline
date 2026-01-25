import os
from typing import Dict, List

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ============================
# Settings
# ============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Checkpoint path saved during training
ATTR_CKPT = "attr_runs/attrnet_fridge10.pt"

# YOLO segmentation weights (replace with your trained best.pt path)
YOLO_SEG_WEIGHTS = "yolov12/runs/fridge_seg_attr102/weights/best.pt"

# Regression attr scale/shift (must match training)
REG_SCALE = {
    "attr3_internal_volume": 1000.0,  # L -> around 0-? range
    "attr8_year": 50.0,               # (year - 2000) / 50
}
REG_SHIFT = {
    "attr3_internal_volume": 0.0,
    "attr8_year": 2000.0,
}

# Human-readable names (regression)
REG_LABEL_KO = {
    "attr1_power_kw": "Power [kW]",
    "attr2_u_value": "U-value [W/m²K]",
    "attr3_internal_volume": "Internal volume [L]",
    "attr4_temp_class": "Temperature class",
    "attr8_year": "Manufacture year",
    "attr10_misc": "Misc (continuous)",
}

# Classification attr index -> human-readable string mapping
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

# Classification attr display names
CLS_LABEL_KO = {
    "attr5_refrigerant": "Refrigerant",
    "attr6_door_type": "Door type",
    "attr7_cabinet_type": "Cabinet type",
    "attr9_insulation_type": "Insulation",
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


def preprocess_pil(img: Image.Image, tfm):
    x = tfm(img).unsqueeze(0)  # (1, C, H, W)
    return x


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
    Returns: (pred_idx, pred_label)
    """
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    label_map = CLS_LABELS.get(attr_name, {})
    label = label_map.get(idx, f"class_{idx}")
    return idx, label


def run_attr_on_pil(pil_img: Image.Image, model, reg_attrs, cls_attrs):
    tfm = build_transform(img_size=224)
    x = preprocess_pil(pil_img, tfm).to(DEVICE)

    with torch.no_grad():
        out = model(x)

    reg_results = {}
    if "reg" in out:
        pred_reg = out["reg"][0]  # (D,)
        reg_results = unscale_regression(pred_reg, reg_attrs)

    cls_results = {}
    for name, logits in out["cls"].items():
        logits0 = logits[0]
        idx, label = decode_classification(logits0, name)
        cls_results[name] = {
            "index": idx,
            "label": label,
        }

    return reg_results, cls_results


def visualize_seg_and_attr(img_path: str,
                           out_path: str = "fridge_seg_attr_vis.png",
                           conf_thres: float = 0.25):
    print(f"\n=== Pipeline inference on: {img_path} ===")

    # 1) Load models (YOLO seg + AttrNet)
    seg_model = YOLO(YOLO_SEG_WEIGHTS)
    attr_model, reg_attrs, cls_attrs = load_attr_model(ATTR_CKPT, DEVICE)

    # 2) Run YOLO segmentation
    results = seg_model(img_path, task="segment", imgsz=640, conf=conf_thres, verbose=False)
    r = results[0]

    # Segmentation visualization image drawn by YOLO (BGR)
    vis_img = r.plot()  # numpy array, BGR

    # 3) Select bbox for fridge crop (single highest-confidence box)
    crop_pil = None
    if r.boxes is not None and len(r.boxes) > 0:
        # Sort by confidence
        boxes = r.boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = int(confs.argmax())
        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        h, w = vis_img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 > x1 and y2 > y1:
            crop = vis_img[y1:y2, x1:x2, :]  # BGR
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

    # 4) AttrNet inference (use crop if available, else full image)
    if crop_pil is None:
        print("⚠ No bbox found; using full image for attr inference.")
        pil_img = Image.open(img_path).convert("RGB")
    else:
        pil_img = crop_pil

    reg_results, cls_results = run_attr_on_pil(pil_img, attr_model, reg_attrs, cls_attrs)

    # 5) Overlay text on the image
    # Keep BGR as-is and only draw text
    lines = []

    # Regression summary (select only what you want)
    reg_order = [
        "attr1_power_kw",
        "attr2_u_value",
        "attr3_internal_volume",
        "attr8_year",
        "attr4_temp_class",
        "attr10_misc",
    ]
    for k in reg_order:
        if k in reg_results:
            v = reg_results[k]
            label_ko = REG_LABEL_KO.get(k, k)
            # Append unit lightly
            lines.append(f"{label_ko}: {v:.2f}")

    # Classification summary
    for k, info in cls_results.items():
        label_name = CLS_LABEL_KO.get(k, k)
        lines.append(f"{label_name}: {info['label']}")

    # Text block at top-left
    y0 = 30
    for i, text in enumerate(lines):
        y = y0 + i * 25
        cv2.putText(
            vis_img,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 6) Save image
    cv2.imwrite(out_path, vis_img)
    print(f"[OK] Saved segmentation + attr result image: {out_path}")

    return reg_results, cls_results, out_path


if __name__ == "__main__":
    # Example test path (replace with your image file)
    test_img = "yolov12/data/fridge_attr10/images/val/test8.jpg"
    if not os.path.exists(test_img):
        print("⚠ Please change test_img to a path that actually exists.")
    else:
        visualize_seg_and_attr(test_img, out_path="fridge_seg_attr_demo.png")
