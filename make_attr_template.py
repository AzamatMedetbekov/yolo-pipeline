import os
import glob
import csv

# 이미지 위치 (이미 YOLO 학습에 사용한 train/val 기준)
BASE_DIR = "yolov12/data/fridge_attr10"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "images", "val")

OUT_CSV = "fridge_attr10_labels.csv"

# 여기서 attr 10개 이름을 먼저 정의해두자.
# 나중에 실제 의미(전력, 열전도율, 냉매종류...)를 이 이름에 매핑해서 쓰면 됨.
ATTR_COLUMNS = [
    "attr1_power_kw",        # 예: 정격 소비전력 (kW)
    "attr2_u_value",         # 예: 열관류율 (W/m2K)
    "attr3_internal_volume", # 예: 내부 용적 (L)
    "attr4_temp_class",      # 예: 온도 클래스 (0~N, 분류)
    "attr5_refrigerant",     # 예: 냉매 종류 (0=R404A, 1=R134a, ...)
    "attr6_door_type",       # 예: 도어 타입 (0=슬라이딩, 1=스윙, ...)
    "attr7_cabinet_type",    # 예: 쇼케이스 타입 (0=수직형, 1=대면형, ...)
    "attr8_year",            # 예: 제조연도 (회귀/분류 둘 다 가능)
    "attr9_insulation_type", # 예: 단열재 타입
    "attr10_misc",           # 예: 기타 임의 속성
]

def collect_images(img_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(paths)

def main():
    train_imgs = collect_images(TRAIN_IMG_DIR)
    val_imgs = collect_images(VAL_IMG_DIR)

    rows = []
    for p in train_imgs:
        rows.append({"split": "train", "image_name": os.path.basename(p)})
    for p in val_imgs:
        rows.append({"split": "val", "image_name": os.path.basename(p)})

    if not rows:
        print("No images found. Check paths.")
        return

    # CSV 헤더: split, image_name, attr1..10
    fieldnames = ["split", "image_name"] + ATTR_COLUMNS

    if os.path.exists(OUT_CSV):
        print(f"{OUT_CSV} already exists. Not overwriting.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            for a in ATTR_COLUMNS:
                r[a] = ""  # 나중에 수동으로 채울 칸
            writer.writerow(r)

    print(f"[OK] Wrote template CSV: {OUT_CSV}")
    print("Fill each attr column for every image (train/val).")

if __name__ == "__main__":
    main()
