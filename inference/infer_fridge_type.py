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
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================
# Configuration
# ============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Checkpoint path
TYPE_CKPT = "type_runs/typenet_fridge.pt"

# Type labels (same as training)
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


def preprocess_image(img_path: str, tfm):
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0)  # (1, C, H, W)
    return x, img


def infer_on_image(img_path: str):
    print(f"\n=== Inference on: {img_path} ===")

    # 1) Load model
    model, type_labels = load_type_model(TYPE_CKPT, DEVICE)
    tfm = build_transform(img_size=224)

    # 2) Load and preprocess image
    x, raw_img = preprocess_image(img_path, tfm)
    x = x.to(DEVICE)

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


if __name__ == "__main__":
    # Test image path (change to actual image)
    test_img = "yolov12/data/fridge_attr10/images/val/test8.jpg"
    if not os.path.exists(test_img):
        print("Test image not found. Please provide a valid image path.")
    else:
        infer_on_image(test_img)
