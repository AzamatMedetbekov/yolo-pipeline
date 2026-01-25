import os
import glob
import csv

# Image locations (based on train/val used for YOLO training)
BASE_DIR = "yolov12/data/fridge_attr10"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "images", "val")

OUT_CSV = "fridge_attr10_labels.csv"

# Define the 10 attr names here first.
# Later map real meanings (power, U-value, refrigerant type...) to these names.
ATTR_COLUMNS = [
    "attr1_power_kw",        # e.g., rated power consumption (kW)
    "attr2_u_value",         # e.g., U-value (W/m2K)
    "attr3_internal_volume", # e.g., internal volume (L)
    "attr4_temp_class",      # e.g., temperature class (0~N, classification)
    "attr5_refrigerant",     # e.g., refrigerant type (0=R404A, 1=R134a, ...)
    "attr6_door_type",       # e.g., door type (0=sliding, 1=swing, ...)
    "attr7_cabinet_type",    # e.g., showcase type (0=vertical, 1=island, ...)
    "attr8_year",            # e.g., manufacture year (regression or classification)
    "attr9_insulation_type", # e.g., insulation type
    "attr10_misc",           # e.g., other arbitrary attribute
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

    # CSV header: split, image_name, attr1..10
    fieldnames = ["split", "image_name"] + ATTR_COLUMNS

    if os.path.exists(OUT_CSV):
        print(f"{OUT_CSV} already exists. Not overwriting.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            for a in ATTR_COLUMNS:
                r[a] = ""  # Fields to fill manually later
            writer.writerow(r)

    print(f"[OK] Wrote template CSV: {OUT_CSV}")
    print("Fill each attr column for every image (train/val).")

if __name__ == "__main__":
    main()
