#!/usr/bin/env python3
"""
YOLOv12 Segmentation Inference Script

Outputs for each detected object:
1. Overlay visualization (mask on original image)
2. Binary mask (white=object, black=background)
3. ROI crop (bounding box region)
4. Masked crop (foreground only with transparent background)

Also saves predictions.json with structured output.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_image_files(source: str) -> List[Path]:
    """Get list of image files from source path."""
    source_path = Path(source)

    if source_path.is_file():
        return [source_path]

    if source_path.is_dir():
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in extensions:
            files.extend(source_path.glob(ext))
            files.extend(source_path.glob(ext.upper()))
        return sorted(files)

    print(f"[ERROR] Source not found: {source}")
    return []


def process_single_image(
    model,
    image_path: Path,
    out_dir: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> Optional[Dict[str, Any]]:
    """
    Process a single image and save all outputs.

    Returns:
        Dictionary with prediction info or None if no detections
    """
    # Read original image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None

    h, w = img.shape[:2]
    stem = image_path.stem

    # Run prediction
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    if not results or len(results) == 0:
        print(f"[WARN] No results for: {image_path.name}")
        return None

    result = results[0]

    # Check for masks
    if result.masks is None or len(result.masks) == 0:
        print(f"[INFO] No detections in: {image_path.name}")
        return None

    # Get highest confidence detection
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Find best detection (highest confidence)
    confidences = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confidences))
    best_conf = float(confidences[best_idx])

    # Get class info
    cls_idx = int(boxes.cls[best_idx].cpu().numpy())
    cls_name = model.names.get(cls_idx, f"class_{cls_idx}")

    # Get bounding box [x1, y1, x2, y2]
    bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    x1, y1, x2, y2 = bbox

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Get mask for best detection
    mask_data = result.masks.data[best_idx].cpu().numpy()

    # Resize mask to original image size if needed
    if mask_data.shape != (h, w):
        mask_data = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)

    # Binary mask (0 or 255)
    binary_mask = (mask_data > 0.5).astype(np.uint8) * 255

    # === Output 1: Overlay visualization ===
    overlay = img.copy()
    # Create colored mask
    color = (0, 255, 0)  # Green
    colored_mask = np.zeros_like(img)
    colored_mask[binary_mask > 0] = color
    # Blend
    alpha = 0.4
    overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    # Draw bounding box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    # Add label
    label = f"{cls_name} {best_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(overlay, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    overlay_path = out_dir / f"{stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    # === Output 2: Binary mask ===
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), binary_mask)

    # === Output 3: ROI crop (bounding box region) ===
    crop = img[y1:y2, x1:x2].copy()
    crop_path = out_dir / f"{stem}_crop.jpg"
    cv2.imwrite(str(crop_path), crop)

    # === Output 4: Masked crop (foreground only, transparent background) ===
    # Create RGBA image
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # Set alpha channel based on mask
    rgba[:, :, 3] = binary_mask
    # Crop to bounding box
    masked_crop = rgba[y1:y2, x1:x2].copy()
    masked_crop_path = out_dir / f"{stem}_masked_crop.png"
    cv2.imwrite(str(masked_crop_path), masked_crop)

    print(f"[OK] {image_path.name} -> {cls_name} ({best_conf:.2f})")

    return {
        "image": str(image_path.name),
        "class": cls_name,
        "class_id": cls_idx,
        "confidence": round(best_conf, 4),
        "bbox": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        },
        "outputs": {
            "overlay": str(overlay_path.name),
            "mask": str(mask_path.name),
            "crop": str(crop_path.name),
            "masked_crop": str(masked_crop_path.name),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLOv12 segmentation inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image file or directory')

    # Output configuration
    parser.add_argument('--out-dir', type=str, default='output',
                        help='Output directory for results')

    # Inference parameters
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size')

    # Device
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0, cpu, etc.)')

    args = parser.parse_args()

    # Print configuration
    print("=" * 50)
    print("YOLOv12 Segmentation Inference")
    print("=" * 50)
    print(f"Weights: {args.weights}")
    print(f"Source:  {args.source}")
    print(f"Output:  {args.out_dir}")
    print(f"Conf:    {args.conf}")
    print(f"IoU:     {args.iou}")
    print(f"ImgSize: {args.imgsz}")
    print("=" * 50)

    # Check weights file
    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Try relative to project root
        weights_path = PROJECT_ROOT / args.weights

    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {args.weights}")
        sys.exit(1)

    # Get image files
    image_files = get_image_files(args.source)
    if not image_files:
        print(f"[ERROR] No images found in: {args.source}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(image_files)} image(s)")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("[INFO] Loading model...")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    # Process images
    print("\n[INFO] Processing images...")
    predictions = []

    for image_path in image_files:
        result = process_single_image(
            model=model,
            image_path=image_path,
            out_dir=out_dir,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
        )
        if result:
            predictions.append(result)

    # Save predictions JSON
    json_path = out_dir / "predictions.json"
    with open(json_path, 'w') as f:
        json.dump({
            "model": str(weights_path.name),
            "source": str(args.source),
            "config": {
                "conf_threshold": args.conf,
                "iou_threshold": args.iou,
                "image_size": args.imgsz,
            },
            "predictions": predictions,
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 50)
    print("Inference Complete!")
    print("=" * 50)
    print(f"Processed: {len(image_files)} images")
    print(f"Detections: {len(predictions)}")
    print(f"Output dir: {out_dir}")
    print(f"JSON: {json_path}")


if __name__ == '__main__':
    main()
