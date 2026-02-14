#!/usr/bin/env python3
"""
YOLOv12 Segmentation Training Script

Features:
- CLI arguments for all training parameters
- Automatic fallback from .pt to .yaml config if weights unavailable
- Dataset validation before training
- Cross-platform path handling
- Reproducible training with seed setting
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


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
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        # Try relative to project root
        yaml_path = PROJECT_ROOT / data_yaml

    if not yaml_path.exists():
        print(f"[ERROR] Dataset config not found: {data_yaml}")
        return False

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check required YAML keys
    required_keys = ['nc', 'names']
    for key in required_keys:
        if key not in config:
            print(f"[ERROR] Missing required key in dataset YAML: '{key}'")
            return False

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
        
        if len(train_images) == 0:
            print(f"[WARNING] No images found in training directory")
            valid = False

    if not val_dir.exists():
        print(f"[ERROR] Validation directory not found: {val_dir}")
        valid = False
    else:
        val_images = list(val_dir.glob('*.[jJ][pP][gG]')) + list(val_dir.glob('*.[pP][nN][gG]'))
        print(f"[INFO] Validation images: {len(val_images)}")
        
        if len(val_images) == 0:
            print(f"[WARNING] No images found in validation directory")

    # Optional: Check for labels (segmentation expects labels/masks)
    train_label_dir = base_path / 'labels' / 'train'
    if not train_label_dir.exists():
        print(f"[WARNING] Training labels directory not found: {train_label_dir}")
        print(f"[WARNING] Make sure labels exist for segmentation training")

    return valid


def main():
    # Set seeds for reproducibility
    set_seed(42)
    
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
    parser.add_argument('--output', type=str, default='runs/segment',
                        help='Output directory for saving results')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name')

    # Training options
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Validate arguments
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.workers < 0:
        parser.error("--workers must be non-negative")

    # Set user-specified seed
    if args.seed != 42:
        set_seed(args.seed)

    # Print configuration
    print("=" * 60)
    print("YOLOv12 Segmentation Training")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"ImgSize: {args.imgsz}")
    print(f"Batch:   {args.batch}")
    print(f"Device:  {args.device}")
    print(f"Output:  {args.output}")
    print(f"Name:    {args.name}")
    print(f"Seed:    {args.seed}")
    print(f"AMP:     {args.amp}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

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
        try:
            model = YOLO(args.model)
        except Exception as e2:
            print(f"[ERROR] Model loading failed completely: {e2}")
            sys.exit(1)

    # Resolve data path
    data_path = args.data
    if not Path(data_path).exists():
        data_path = str(PROJECT_ROOT / args.data)

    data_yaml = Path(data_path)
    if not data_yaml.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        sys.exit(1)

    # Load augmentations from YAML
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    augmentations = data_cfg.get("augmentations", {})
    if augmentations and not isinstance(augmentations, dict):
        print(f"[ERROR] 'augmentations' in YAML must be a dict/mapping")
        print(f"[ERROR] Got type: {type(augmentations).__name__}")
        print("[INFO] Continuing without custom augmentations...")
        augmentations = {}
    elif augmentations:
        print(f"[INFO] Loaded {len(augmentations)} custom augmentation settings from YAML")

    # Start training
    print("\n[INFO] Starting training...")
    print(f"[INFO] Progress will be shown by Ultralytics...\n")
    
    try:
        results = model.train(
            task='segment',
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.output,
            name=args.name,
            resume=args.resume,
            amp=args.amp,
            workers=args.workers,
            seed=args.seed,
            **augmentations,
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Construct output paths
        output_path = Path(args.output) / args.name
        best_weights = output_path / "weights" / "best.pt"
        last_weights = output_path / "weights" / "last.pt"
        
        print(f"Results saved to: {output_path.resolve()}")
        
        if best_weights.exists():
            print(f"✓ Best weights: {best_weights.resolve()}")
        if last_weights.exists():
            print(f"✓ Last weights: {last_weights.resolve()}")
        
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()