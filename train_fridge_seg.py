from ultralytics import YOLO
import os

# ==========================
# Gather settings in one place
# ==========================
# Segmentation base model in YOLOv12 repo
BASE_MODEL = "yolov12/yolov12s-seg.pt"   # Start from COCO pretrained weights
# To continue fine-tuning later, e.g.: "yolov12/runs/segment/train/weights/best.pt"

DATA_YAML = "fridge_attr10.yaml"        # Data yaml just created (relative to YoloProject)
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 4
DEVICE = "0"                          # Use "0" when GPU is set up

PROJECT = "yolov12/runs"               # Default folder for YOLO outputs
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
        amp=False
    )

    print("=== Training finished ===")
    print(f"Results saved under: {os.path.join(PROJECT, NAME)}")

if __name__ == "__main__":
    main()