import os
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================
# 설정
# ============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 학습 때 저장해둔 체크포인트 경로
ATTR_CKPT = "attr_runs/attrnet_fridge10.pt"

# 회귀 attr 스케일/시프트 (학습할 때 썼던 값과 반드시 동일해야 함)
# - attr3_internal_volume: L 단위 → /1000 해서 학습
# - attr8_year           : (year - 2000) / 50 해서 학습
REG_SCALE = {
    "attr3_internal_volume": 1000.0,
    "attr8_year": 50.0,
}
REG_SHIFT = {
    "attr3_internal_volume": 0.0,
    "attr8_year": 2000.0,
}

# 분류 attr 인덱스 → 사람이 읽을 수 있는 문자열 매핑
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
        0: "슬라이딩",
        1: "스윙",
        2: "오픈형",
    },
    "attr7_cabinet_type": {
        0: "수직형",
        1: "대면형",
    },
    "attr9_insulation_type": {
        0: "PU 폼",
        1: "기타",
    },
}


# ============================
# AttrNet 정의 (train 때와 동일해야 함)
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
# 유틸 함수
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
    pred_reg: (D,) 텐서 (한 샘플)
    리턴: {attr_name: 원래 단위의 값}
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
    logits: (C,) 텐서 (한 샘플)
    반환: (pred_idx, pred_label, probs)
    """
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    label_map = CLS_LABELS.get(attr_name, {})
    label = label_map.get(idx, f"class_{idx}")
    return idx, label, probs.detach().cpu().numpy()


def infer_on_image(img_path: str):
    print(f"\n=== Inference on: {img_path} ===")

    # 1) 모델, 변환 로드
    model, reg_attrs, cls_attrs = load_attr_model(ATTR_CKPT, DEVICE)
    tfm = build_transform(img_size=224)

    # 2) 이미지 로드 + 전처리
    x, raw_img = preprocess_image(img_path, tfm)
    x = x.to(DEVICE)

    # 3) 인퍼런스
    with torch.no_grad():
        out = model(x)

    # 4) 회귀 결과 처리 (첫 샘플 기준)
    reg_results = {}
    if "reg" in out:
        pred_reg = out["reg"][0]  # (D,)
        reg_results = unscale_regression(pred_reg, reg_attrs)

    # 5) 분류 결과 처리
    cls_results = {}
    for name, logits in out["cls"].items():
        logits0 = logits[0]  # (C,)
        idx, label, probs = decode_classification(logits0, name)
        cls_results[name] = {
            "index": idx,
            "label": label,
            "probs": probs,
        }

    # 6) 출력 (보기 좋게)
    print("\n[Regression attrs]")
    for k, v in reg_results.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Classification attrs]")
    for k, info in cls_results.items():
        print(f"  {k}: idx={info['index']}, label={info['label']}")

    return reg_results, cls_results


if __name__ == "__main__":
    # 테스트용 예시 경로 (원하는 이미지 파일로 바꿔서 사용)
    test_img = "yolov12/data/fridge_attr10/images/val/test8.jpg"
    if not os.path.exists(test_img):
        print("⚠ test_img 경로를 실제 존재하는 이미지로 바꿔주세요.")
    else:
        infer_on_image(test_img)
