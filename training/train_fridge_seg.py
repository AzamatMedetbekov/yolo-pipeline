from ultralytics import YOLO
import os

# ==========================
# Configuration
# ==========================
# YOLOv12 segmentation base model
BASE_MODEL = "yolov12/yolov12s-seg.pt"   # Start from COCO pretrained weights
# To continue fine-tuning, use e.g.: "yolov12/runs/segment/train/weights/best.pt"

DATA_YAML = "configs/fridge_attr10.yaml"        # Dataset config (relative to project root)
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 4
DEVICE = "0"                          # GPU device ID ("0", "1", or "cpu")

PROJECT = "yolov12/runs"               # Base folder for saving results
NAME = "fridge_seg_attr10"             # Experiment name

def main():
    print("=== YOLOv12 Fridge Seg Training ===")
    print(f"Model: {BASE_MODEL}")
    print(f"Data : {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, ImgSize: {IMG_SIZE}, Batch: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    # Load model
    model = YOLO(BASE_MODEL)

    # Start training
    model.train(
        task="segment",
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        amp=False,
        # ========== Augmentation (aggressive for small dataset) ==========
        # Geometric augmentations
        degrees=15.0,       # Rotate image (+/- 15 degrees)
        translate=0.2,      # Translate image (+/- 20%)
        scale=0.5,          # Scale image (+/- 50%)
        shear=5.0,          # Shear angle (+/- 5 degrees)
        perspective=0.001,  # Perspective distortion
        flipud=0.0,         # Flip up-down (0 for fridges - usually upright)
        fliplr=0.5,         # Flip left-right (50% probability)
        # Mosaic & Mixup (very effective for small datasets)
        mosaic=1.0,         # Mosaic augmentation (combines 4 images)
        mixup=0.3,          # Mixup augmentation (blends 2 images) - increased
        copy_paste=0.3,     # Segment copy-paste (very useful for segmentation) - increased
        # Color augmentations (HSV space)
        hsv_h=0.015,        # Hue shift (+/- 1.5%)
        hsv_s=0.7,          # Saturation shift (+/- 70%)
        hsv_v=0.4,          # Value/brightness shift (+/- 40%)
        # Regularization
        erasing=0.4,        # Random erasing (cutout)
    )

    print("=== Training finished ===")
    print(f"Results saved under: {os.path.join(PROJECT, NAME)}")

if __name__ == "__main__":
    main()