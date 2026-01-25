import os
import json
import glob
import cv2

# Class map (currently only 'showcase')
CLASS_MAP = {
    "showcase": 0,
    # Add more classes later, e.g., "upright": 1, "coldroom": 2
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
            # Normalize to 0~1
            nx = x / w
            ny = y / h
            # Clamp to [0,1] range as required by YOLO format
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            pts.append(f"{nx:.6f}")
            pts.append(f"{ny:.6f}")

        if len(pts) < 6:
            # Less than 3 points cannot form a valid polygon
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