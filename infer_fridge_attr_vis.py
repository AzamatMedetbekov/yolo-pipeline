import os
from typing import Dict, List

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ============================
# 설정
# ============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 학습 때 저장해둔 체크포인트 경로
ATTR_CKPT = "attr_runs/attrnet_fridge10.pt"

# YOLO 세그멘테이션 가중치 (네가 학습한 best.pt 경로로 바꿔)
YOLO_SEG_WEIGHTS = "yolov12/runs/fridge_seg_attr102/weights/best.pt"

# 회귀 attr 스케일/시프트 (학습 때와 동일)
REG_SCALE = {
    "attr3_internal_volume": 1000.0,  # L → 대략 0~몇
    "attr8_year": 50.0,               # (year - 2000) / 50
}
REG_SHIFT = {
    "attr3_internal_volume": 0.0,
    "attr8_year": 2000.0,
}

# 사람이 보기 좋은 한글 이름 (회귀)
REG_LABEL_KO = {
    "attr1_power_kw": "소비전력[kW]",
    "attr2_u_value": "열관류율[W/m²K]",
    "attr3_internal_volume": "내용적[L]",
    "attr4_temp_class": "온도클래스",
    "attr8_year": "제조년",
    "attr10_misc": "기타(연속값)",
}

# 분류 attr 인덱스 → 사람이 읽을 수 있는 문자열 매핑
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

# 분류 attr 한글 이름
CLS_LABEL_KO = {
    "attr5_refrigerant": "냉매",
    "attr6_door_type": "도어타입",
    "attr7_cabinet_type": "캐비닛형태",
    "attr9_insulation_type": "단열재",
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


def preprocess_pil(img: Image.Image, tfm):
    x = tfm(img).unsqueeze(0)  # (1, C, H, W)
    return x


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
    반환: (pred_idx, pred_label)
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

    # 1) 모델 로드 (YOLO seg + AttrNet)
    seg_model = YOLO(YOLO_SEG_WEIGHTS)
    attr_model, reg_attrs, cls_attrs = load_attr_model(ATTR_CKPT, DEVICE)

    # 2) YOLO 세그 돌리기
    results = seg_model(img_path, task="segment", imgsz=640, conf=conf_thres, verbose=False)
    r = results[0]

    # YOLO가 그려준 세그멘테이션 시각화 이미지 (BGR)
    vis_img = r.plot()  # numpy array, BGR

    # 3) 냉장고 영역 crop용 bbox 선택 (가장 신뢰도 높은 박스 1개)
    crop_pil = None
    if r.boxes is not None and len(r.boxes) > 0:
        # conf 기준으로 정렬
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

    # 4) AttrNet 인퍼런스 (crop 있으면 crop, 없으면 원본 전체)
    if crop_pil is None:
        print("⚠ bbox가 없어 전체 이미지를 그대로 attr 인퍼런스에 사용합니다.")
        pil_img = Image.open(img_path).convert("RGB")
    else:
        pil_img = crop_pil

    reg_results, cls_results = run_attr_on_pil(pil_img, attr_model, reg_attrs, cls_attrs)

    # 5) 텍스트를 이미지 위에 overlay
    # BGR → 그대로 두고 글자만 얹자
    lines = []

    # 회귀 요약 (원하면 필요한 것만 골라서)
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
            # 단위 약하게 붙여줌
            lines.append(f"{label_ko}: {v:.2f}")

    # 분류 요약
    for k, info in cls_results.items():
        label_name = CLS_LABEL_KO.get(k, k)
        lines.append(f"{label_name}: {info['label']}")

    # 좌측 상단에 텍스트 블록
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

    # 6) 이미지 저장
    cv2.imwrite(out_path, vis_img)
    print(f"[OK] 세그먼테이션 + attr 결과 이미지를 저장했습니다: {out_path}")

    return reg_results, cls_results, out_path


if __name__ == "__main__":
    # 테스트용 예시 경로 (원하는 이미지 파일로 바꿔서 사용)
    test_img = "yolov12/data/fridge_attr10/images/val/test8.jpg"
    if not os.path.exists(test_img):
        print("⚠ test_img 경로를 실제 존재하는 이미지로 바꿔주세요.")
    else:
        visualize_seg_and_attr(test_img, out_path="fridge_seg_attr_demo.png")
