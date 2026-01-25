#!/usr/bin/env python3
"""
LabelMe to YOLO Segmentation Format Converter

Converts LabelMe JSON annotations to YOLO polygon segmentation format.

LabelMe format: JSON with shapes containing polygon points
YOLO seg format: class_id x1 y1 x2 y2 ... xn yn (normalized 0-1)

Usage:
    python scripts/convert_labelme.py --json-dir path/to/labelme_json --img-dir path/to/images --out-dir path/to/labels
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Default class mapping
DEFAULT_CLASS_MAP = {
    "showcase": 0,
    "fridge": 0,
    "fridge_equipment": 0,
    "refrigerator": 0,
    # Add more classes as needed:
    # "upright": 1,
    # "coldroom": 2,
}


def convert_labelme_to_yolo(
    json_path: Path,
    img_dir: Path,
    out_dir: Path,
    class_map: Dict[str, int],
    verbose: bool = False,
) -> bool:
    """
    Convert a single LabelMe JSON file to YOLO format.

    Args:
        json_path: Path to LabelMe JSON file
        img_dir: Directory containing images
        out_dir: Output directory for YOLO labels
        class_map: Mapping from label names to class IDs
        verbose: Print detailed output

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {json_path}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to read {json_path}: {e}")
        return False

    # Get image info
    img_name = data.get("imagePath", "")
    if not img_name:
        print(f"[ERROR] No imagePath in {json_path}")
        return False

    # Handle potential path in imagePath (LabelMe sometimes includes full path)
    img_name = Path(img_name).name

    # Find image file
    img_path = img_dir / img_name
    if not img_path.exists():
        # Try with different extensions
        stem = Path(img_name).stem
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                img_name = candidate.name
                break

    if not img_path.exists():
        print(f"[WARN] Image not found for {json_path.name}: {img_path}")
        return False

    # Get image dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Failed to read image: {img_path}")
        return False

    h, w = img.shape[:2]

    # Process shapes
    lines = []
    shapes = data.get("shapes", [])

    for shape in shapes:
        label = shape.get("label", "")

        # Map label to class ID
        if label not in class_map:
            if verbose:
                print(f"[INFO] Skipping unknown label '{label}' in {json_path.name}")
            continue

        cls_id = class_map[label]
        shape_type = shape.get("shape_type", "polygon")

        # Only handle polygons
        if shape_type != "polygon":
            if verbose:
                print(f"[INFO] Skipping shape_type '{shape_type}' in {json_path.name}")
            continue

        points = shape.get("points", [])

        if len(points) < 3:
            print(f"[WARN] Too few points ({len(points)}) in {json_path.name}, label={label}")
            continue

        # Normalize coordinates
        normalized_coords = []
        for x, y in points:
            nx = x / w
            ny = y / h
            # Clamp to [0, 1]
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            normalized_coords.append(f"{nx:.6f}")
            normalized_coords.append(f"{ny:.6f}")

        # Create YOLO line: class_id x1 y1 x2 y2 ... xn yn
        line = str(cls_id) + " " + " ".join(normalized_coords)
        lines.append(line)

    if not lines:
        if verbose:
            print(f"[WARN] No valid shapes in {json_path.name}")
        return False

    # Write output file
    out_name = Path(img_name).stem + ".txt"
    out_path = out_dir / out_name

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    if verbose:
        print(f"[OK] {json_path.name} -> {out_name} ({len(lines)} polygon(s))")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert LabelMe JSON annotations to YOLO segmentation format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--json-dir', type=str, required=True,
                        help='Directory containing LabelMe JSON files')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for YOLO labels')

    parser.add_argument('--class-map', type=str, default=None,
                        help='JSON file with class name to ID mapping')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed output')

    args = parser.parse_args()

    print("=" * 50)
    print("LabelMe to YOLO Segmentation Converter")
    print("=" * 50)
    print(f"JSON dir: {args.json_dir}")
    print(f"Image dir: {args.img_dir}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 50)

    # Resolve paths
    json_dir = Path(args.json_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)

    if not json_dir.exists():
        print(f"[ERROR] JSON directory not found: {json_dir}")
        sys.exit(1)

    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load class map
    if args.class_map:
        try:
            with open(args.class_map, 'r') as f:
                class_map = json.load(f)
            print(f"[INFO] Loaded class map with {len(class_map)} classes")
        except Exception as e:
            print(f"[ERROR] Failed to load class map: {e}")
            sys.exit(1)
    else:
        class_map = DEFAULT_CLASS_MAP
        print(f"[INFO] Using default class map: {list(class_map.keys())}")

    # Find JSON files
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No JSON files found in: {json_dir}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(json_files)} JSON files")

    # Convert each file
    success_count = 0
    fail_count = 0

    for json_path in json_files:
        result = convert_labelme_to_yolo(
            json_path=json_path,
            img_dir=img_dir,
            out_dir=out_dir,
            class_map=class_map,
            verbose=args.verbose,
        )
        if result:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 50)
    print("Conversion Complete")
    print("=" * 50)
    print(f"Successful: {success_count}")
    print(f"Failed:     {fail_count}")
    print(f"Output dir: {out_dir}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
