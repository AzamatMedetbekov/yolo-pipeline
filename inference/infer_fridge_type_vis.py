"""
Combined pipeline: YOLO segmentation + refrigerator type classification.

Workflow:
1. Run YOLO segmentation to detect refrigerators
2. Crop the best detection (highest confidence)
3. Run TypeNet classifier on the crop
4. Overlay type label on visualization
5. Save output image

Type classes:
    0: vertical        - Normal refrigerator (drink freezer)
    1: horizontal      - Horizontal (ice cream freezer)
    2: vertical_open   - Vertical without door
    3: horizontal_open - Horizontal without door
    4: combination     - Two types combined
    5: coldroom        - Walk-in cold room
"""
import os

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ============================
# Configuration
# ============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Model paths
TYPE_CKPT = "runs/type/train/best.pt"
YOLO_SEG_WEIGHTS = "runs/segment/train/weights/best.pt"

# Type labels
TYPE_LABELS = {
    0: "vertical",
    1: "horizontal",
    2: "vertical_open",
    3: "horizontal_open",
    4: "combination",
    5: "coldroom",
}


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


def preprocess_pil(img: Image.Image, tfm):
    x = tfm(img).unsqueeze(0)  # (1, C, H, W)
    return x


def run_type_on_pil(pil_img: Image.Image, model, type_labels):
    tfm = build_transform(img_size=224)
    x = preprocess_pil(pil_img, tfm).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = logits.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()

    pred_label = type_labels.get(pred_idx, f"class_{pred_idx}")

    return pred_idx, pred_label, pred_prob, probs[0].cpu().numpy()


def visualize_seg_and_type(img_path: str,
                           out_path: str = "fridge_seg_type_vis.png",
                           conf_thres: float = 0.25):
    print(f"\n=== Pipeline inference on: {img_path} ===")

    # 1) Load models (YOLO seg + TypeNet)
    seg_model = YOLO(YOLO_SEG_WEIGHTS)
    type_model, type_labels = load_type_model(TYPE_CKPT, DEVICE)

    # 2) Run YOLO segmentation
    results = seg_model(img_path, task="segment", imgsz=640, conf=conf_thres, verbose=False)
    r = results[0]

    # YOLO visualization image (BGR)
    vis_img = r.plot()

    # 3) Crop best detection (highest confidence)
    crop_pil = None
    if r.boxes is not None and len(r.boxes) > 0:
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

    # 4) Run TypeNet on crop (or full image if no detection)
    if crop_pil is None:
        print("No bbox detected, using full image for type classification.")
        pil_img = Image.open(img_path).convert("RGB")
    else:
        pil_img = crop_pil

    pred_idx, pred_label, pred_prob, probs = run_type_on_pil(pil_img, type_model, type_labels)

    # 5) Overlay type label on visualization
    text = f"Type: {pred_label} ({pred_prob:.2f})"

    # Text position (top-left)
    cv2.putText(
        vis_img,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 6) Save image
    cv2.imwrite(out_path, vis_img)
    print(f"[OK] Saved visualization to: {out_path}")
    print(f"Predicted type: {pred_idx} ({pred_label}), confidence: {pred_prob:.4f}")

    return pred_idx, pred_label, pred_prob, out_path


if __name__ == "__main__":
    # Test image path (change to actual image)
    test_img = "data/images/val/test8.jpg"
    if not os.path.exists(test_img):
        print("Test image not found. Please provide a valid image path.")
    else:
        visualize_seg_and_type(test_img, out_path="fridge_seg_type_demo.png")
