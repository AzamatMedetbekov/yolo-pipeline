#!/usr/bin/env python3
"""
YOLOv12 Segmentation Training Script

Features:
- CLI arguments for all training parameters
- Automatic fallback from .pt to .yaml config if weights unavailable
- Dataset validation before training
- Cross-platform path handling
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_model_availability(model_name: str) -> str:
    """
    Check if pretrained weights exist, fallback to YAML config if not.

    Args:
        model_name: Base model name (e.g., 'yolov12s-seg')

    Returns:
        Path to .pt file or .yaml config
    """
    # Possible locations for pretrained weights
    pt_paths = [
        PROJECT_ROOT / "yolov12" / f"{model_name}.pt",
        PROJECT_ROOT / f"{model_name}.pt",
        Path.home() / ".cache" / "ultralytics" / f"{model_name}.pt",
    ]

    for pt_path in pt_paths:
        if pt_path.exists():
            print(f"[INFO] Found pretrained weights: {pt_path}")
            return str(pt_path)

    # Fallback to YAML config (Ultralytics will download weights)
    yaml_name = f"{model_name}.yaml"
    print(f"[INFO] Pretrained weights not found locally, using config: {yaml_name}")
    print(f"[INFO] Ultralytics will download weights automatically if available")
    return yaml_name


def validate_dataset(data_yaml: str) -> bool:
    """
    Validate that dataset directories exist.

    Args:
        data_yaml: Path to dataset YAML config

    Returns:
        True if valid, False otherwise
    """
    import yaml

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        # Try relative to project root
        yaml_path = PROJECT_ROOT / data_yaml

    if not yaml_path.exists():
        print(f"[ERROR] Dataset config not found: {data_yaml}")
        return False

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve base path
    base_path = Path(config.get('path', '.'))
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path

    # Check train/val directories
    train_dir = base_path / config.get('train', 'images/train')
    val_dir = base_path / config.get('val', 'images/val')

    valid = True

    if not train_dir.exists():
        print(f"[ERROR] Training directory not found: {train_dir}")
        valid = False
    else:
        train_images = list(train_dir.glob('*.[jJ][pP][gG]')) + list(train_dir.glob('*.[pP][nN][gG]'))
        print(f"[INFO] Training images: {len(train_images)}")

    if not val_dir.exists():
        print(f"[ERROR] Validation directory not found: {val_dir}")
        valid = False
    else:
        val_images = list(val_dir.glob('*.[jJ][pP][gG]')) + list(val_dir.glob('*.[pP][nN][gG]'))
        print(f"[INFO] Validation images: {len(val_images)}")

    return valid


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv12 segmentation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument('--model', type=str, default='yolov12s-seg',
                        help='Base model name (e.g., yolov12n-seg, yolov12s-seg, yolov12m-seg)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights (overrides --model)')

    # Dataset configuration
    parser.add_argument('--data', type=str, default='configs/fridge_seg.yaml',
                        help='Path to dataset YAML config')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0, 1, cpu, etc.)')

    # Output configuration
    parser.add_argument('--project', type=str, default='runs/segment',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name')

    # Training options
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers')

    args = parser.parse_args()

    # Print configuration
    print("=" * 50)
    print("YOLOv12 Segmentation Training")
    print("=" * 50)
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"ImgSize: {args.imgsz}")
    print(f"Batch:   {args.batch}")
    print(f"Device:  {args.device}")
    print(f"Project: {args.project}")
    print(f"Name:    {args.name}")
    print("=" * 50)

    # Validate dataset
    print("\n[INFO] Validating dataset...")
    if not validate_dataset(args.data):
        print("[ERROR] Dataset validation failed. Exiting.")
        sys.exit(1)

    # Determine model source
    if args.weights:
        model_source = args.weights
        print(f"\n[INFO] Using specified weights: {model_source}")
    else:
        model_source = check_model_availability(args.model)

    # Import and initialize model
    print("\n[INFO] Loading model...")
    from ultralytics import YOLO

    try:
        model = YOLO(model_source)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        # Try with just the model name (let Ultralytics handle it)
        print(f"[INFO] Attempting to load via Ultralytics hub: {args.model}")
        model = YOLO(args.model)

    # Resolve data path
    data_path = args.data
    if not Path(data_path).exists():
        data_path = str(PROJECT_ROOT / args.data)

    # Start training
    print("\n[INFO] Starting training...")
    try:
        results = model.train(
            task='segment',
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            amp=args.amp,
            workers=args.workers,

            # ========== Augmentation (aggressive for small dataset) ==========
            # Geometric augmentations
            degrees=15.0,       # Rotate image (+/- 15 degrees)
            degrees=15.0,       # Rotate image (+/- 15 degrees)
            translate=0.2,      # Translate image (+/- 20%)
            scale=0.5,          # Scale image (+/- 50%)
            shear=5.0,          # Shear angle (+/- 5 degrees)
            perspective=0.001,  # Perspective distortion
            flipud=0.0,         # Flip up-down (0 for fridges - usually upright)
            fliplr=0.5,         # Flip left-right (50% probability
            hsv_s=0.7,          # Saturation shift (+/- 70%)
            hsv_v=0.4,          # Value/brightness shift (+/- 40%)
        )

        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Results saved to: {args.project}/{args.name}")
        print(f"Best weights: {args.project}/{args.name}/weights/best.pt")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
