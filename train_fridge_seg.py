from ultralytics import YOLO
import os

# ==========================
# 설정값 한 군데 모아두기
# ==========================
# YOLOv12 repo 안의 세그멘테이션 base 모델
BASE_MODEL = "yolov12/yolov12s-seg.pt"   # 처음에는 COCO 사전학습 모델에서 시작
# 나중에 fine-tune 이어갈 땐 예: "yolov12/runs/segment/train/weights/best.pt"

DATA_YAML = "fridge_attr10.yaml"        # 방금 만든 data yaml (YoloProject 기준)
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 4
DEVICE = "0"                          # 나중에 GPU 맞춰지면 "0" 같은 걸로

PROJECT = "yolov12/runs"               # YOLO가 결과 저장할 기본 폴더
NAME = "fridge_seg_attr10"             # 이번 실험 이름

def main():
    print("=== YOLOv12 Fridge Seg Training ===")
    print(f"Model: {BASE_MODEL}")
    print(f"Data : {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, ImgSize: {IMG_SIZE}, Batch: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    # 모델 로드
    model = YOLO(BASE_MODEL)

    # 학습 호출
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