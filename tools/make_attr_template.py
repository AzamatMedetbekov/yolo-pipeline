import os
import glob
import csv

# Image directories (based on train/val split used for YOLO training)
BASE_DIR = "yolov12/data/fridge_attr10"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "images", "val")

OUT_CSV = "data/fridge_attr10_labels.csv"

# Define 10 attribute column names here.
# Map actual meanings (power, thermal conductivity, refrigerant type, etc.) to these names later.
ATTR_COLUMNS = [
    "attr1_power_kw",        # e.g., Rated power consumption (kW)
    "attr2_u_value",         # e.g., Thermal transmittance (W/m2K)
    "attr3_internal_volume", # e.g., Internal volume (L)
    "attr4_temp_class",      # e.g., Temperature class (0~N, classification)
    "attr5_refrigerant",     # e.g., Refrigerant type (0=R404A, 1=R134a, ...)
    "attr6_door_type",       # e.g., Door type (0=sliding, 1=swing, ...)
    "attr7_cabinet_type",    # e.g., Cabinet type (0=vertical, 1=face-to-face, ...)
    "attr8_year",            # e.g., Manufacturing year (can be regression or classification)
    "attr9_insulation_type", # e.g., Insulation type
    "attr10_misc",           # e.g., Other miscellaneous attributes
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
                r[a] = ""  # Empty field to be filled manually later
            writer.writerow(r)

    print(f"[OK] Wrote template CSV: {OUT_CSV}")
    print("Fill each attr column for every image (train/val).")

if __name__ == "__main__":
    main()
