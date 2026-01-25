"""
Generate CSV template for 6-class refrigerator type classification.

Columns: split, image_name, type_class

Type classes:
    0: vertical        - Normal refrigerator (drink freezer)
    1: horizontal      - Horizontal (ice cream freezer)
    2: vertical_open   - Vertical without door
    3: horizontal_open - Horizontal without door
    4: combination     - Two types combined
    5: coldroom        - Walk-in cold room
"""
import os
import glob
import csv

# Image directories (same structure as segmentation)
BASE_DIR = "yolov12/data/fridge_attr10"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "images", "val")

OUT_CSV = "data/fridge_type_labels.csv"


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

    fieldnames = ["split", "image_name", "type_class"]

    if os.path.exists(OUT_CSV):
        print(f"{OUT_CSV} already exists. Not overwriting.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r["type_class"] = ""  # User fills this with 0-5
            writer.writerow(r)

    print(f"[OK] Wrote template CSV: {OUT_CSV}")
    print("Fill the type_class column for every image (0-5):")
    print("  0: vertical        - Normal refrigerator (drink freezer)")
    print("  1: horizontal      - Horizontal (ice cream freezer)")
    print("  2: vertical_open   - Vertical without door")
    print("  3: horizontal_open - Horizontal without door")
    print("  4: combination     - Two types combined")
    print("  5: coldroom        - Walk-in cold room")


if __name__ == "__main__":
    main()
