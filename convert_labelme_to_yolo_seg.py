import os
import json
import glob
import cv2

# 클래스 맵 (지금은 showcase 하나만)
CLASS_MAP = {
    "showcase": 0,
    # 나중에 "upright": 1, "coldroom": 2 이런 식으로 추가
}

BASE_DIR = "yolov12/data/fridge_attr10"
IMG_DIR = os.path.join(BASE_DIR, "images")
JSON_DIR = os.path.join(BASE_DIR, "labelme_json")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

os.makedirs(LABEL_DIR, exist_ok=True)

json_paths = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))

for jp in json_paths:
    with open(jp, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_name = data["imagePath"]
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"[WARN] image not found for {jp}, expected {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] failed to read image: {img_path}")
        continue

    h, w = img.shape[:2]

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in CLASS_MAP:
            print(f"[INFO] skip shape with label '{label}' in {jp}")
            continue

        cls_id = CLASS_MAP[label]

        pts = []
        for x, y in shape["points"]:
            # 0~1로 정규화
            nx = x / w
            ny = y / h
            # YOLO 포맷은 [0,1] 범위를 기대하므로 클램핑
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            pts.append(f"{nx:.6f}")
            pts.append(f"{ny:.6f}")

        if len(pts) < 6:
            # 포인트가 3개 미만이면 폴리곤 안 됨
            print(f"[WARN] too few points in {jp}, label={label}")
            continue

        line = str(cls_id) + " " + " ".join(pts)
        lines.append(line)

    if not lines:
        print(f"[WARN] no valid shapes in {jp}")
        continue

    out_name = os.path.splitext(img_name)[0] + ".txt"
    out_path = os.path.join(LABEL_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] wrote {out_path}")