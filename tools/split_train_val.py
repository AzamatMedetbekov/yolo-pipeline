import os
import glob
import shutil

# Paths relative to YoloProject root
BASE_DIR = "yolov12/data/fridge_attr10"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

IMG_TRAIN_DIR = os.path.join(IMG_DIR, "train")
IMG_VAL_DIR = os.path.join(IMG_DIR, "val")
LBL_TRAIN_DIR = os.path.join(LBL_DIR, "train")
LBL_VAL_DIR = os.path.join(LBL_DIR, "val")

os.makedirs(IMG_TRAIN_DIR, exist_ok=True)
os.makedirs(IMG_VAL_DIR, exist_ok=True)
os.makedirs(LBL_TRAIN_DIR, exist_ok=True)
os.makedirs(LBL_VAL_DIR, exist_ok=True)

# Only process images not yet in train/val (directly under images/)
img_paths = sorted(
    glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
    + glob.glob(os.path.join(IMG_DIR, "*.png"))
)

n = len(img_paths)
print(f"Found {n} images in {IMG_DIR}")

if n == 0:
    print("No images found. Check your paths.")
    raise SystemExit

# Simple split: first 80% for train, last 20% for val
split_idx = max(1, int(n * 0.8))
train_imgs = img_paths[:split_idx]
val_imgs = img_paths[split_idx:]

def copy_pair(img_list, img_target_dir, lbl_target_dir):
    for ip in img_list:
        name = os.path.basename(ip)
        stem, _ = os.path.splitext(name)
        lp = os.path.join(LBL_DIR, stem + ".txt")

        if not os.path.exists(lp):
            print(f"[WARN] label not found for {name}, expected {lp}")
            continue

        shutil.copy2(ip, os.path.join(img_target_dir, name))
        shutil.copy2(lp, os.path.join(lbl_target_dir, stem + ".txt"))
        print(f"[OK] copied {name}")

print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")

copy_pair(train_imgs, IMG_TRAIN_DIR, LBL_TRAIN_DIR)
copy_pair(val_imgs, IMG_VAL_DIR, LBL_VAL_DIR)

print("Done.")