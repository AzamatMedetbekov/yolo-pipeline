#!/usr/bin/env python3
"""
YOLO Segmentation Label Validator

Validates YOLO polygon segmentation labels:
- Checks image/label file pairs exist
- Validates coordinate ranges [0, 1]
- Ensures minimum polygon points (>= 3 points = 6 values)
- Validates class indices
- Reports detailed errors
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationError:
    """Represents a single validation error."""
    file: str
    line: int
    message: str
    severity: str = "error"  # error, warning


@dataclass
class ValidationResult:
    """Validation results for a dataset."""
    total_images: int = 0
    total_labels: int = 0
    valid_labels: int = 0
    total_polygons: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    missing_labels: List[str] = field(default_factory=list)
    missing_images: List[str] = field(default_factory=list)


def parse_label_line(line: str, line_num: int, file_name: str, num_classes: int) -> Tuple[bool, List[ValidationError]]:
    """
    Parse and validate a single label line.

    YOLO seg format: class_id x1 y1 x2 y2 ... xn yn

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []
    line = line.strip()

    if not line:
        return True, []  # Empty lines are OK

    parts = line.split()

    if len(parts) < 7:  # class_id + at least 3 points (6 coords)
        errors.append(ValidationError(
            file=file_name,
            line=line_num,
            message=f"Too few values: {len(parts)} (need class_id + at least 6 coordinates for 3 points)",
        ))
        return False, errors

    # Validate class ID
    try:
        class_id = int(parts[0])
        if class_id < 0:
            errors.append(ValidationError(
                file=file_name,
                line=line_num,
                message=f"Negative class ID: {class_id}",
            ))
        if class_id >= num_classes:
            errors.append(ValidationError(
                file=file_name,
                line=line_num,
                message=f"Class ID {class_id} >= num_classes ({num_classes})",
            ))
    except ValueError:
        errors.append(ValidationError(
            file=file_name,
            line=line_num,
            message=f"Invalid class ID: '{parts[0]}' (not an integer)",
        ))
        return False, errors

    # Validate coordinates
    coords = parts[1:]

    if len(coords) % 2 != 0:
        errors.append(ValidationError(
            file=file_name,
            line=line_num,
            message=f"Odd number of coordinates: {len(coords)} (should be pairs of x,y)",
        ))
        return False, errors

    num_points = len(coords) // 2
    if num_points < 3:
        errors.append(ValidationError(
            file=file_name,
            line=line_num,
            message=f"Polygon has only {num_points} points (minimum is 3)",
        ))

    # Check each coordinate
    for i, coord in enumerate(coords):
        try:
            val = float(coord)
            if val < 0 or val > 1:
                coord_type = "x" if i % 2 == 0 else "y"
                point_idx = i // 2 + 1
                errors.append(ValidationError(
                    file=file_name,
                    line=line_num,
                    message=f"Coordinate out of range [0,1]: {coord_type}{point_idx}={val:.6f}",
                    severity="warning" if (val < -0.1 or val > 1.1) else "warning",
                ))
        except ValueError:
            errors.append(ValidationError(
                file=file_name,
                line=line_num,
                message=f"Invalid coordinate: '{coord}' (not a number)",
            ))

    return len([e for e in errors if e.severity == "error"]) == 0, errors


def validate_label_file(label_path: Path, num_classes: int) -> Tuple[int, List[ValidationError]]:
    """
    Validate a single label file.

    Returns:
        Tuple of (polygon_count, list of errors)
    """
    errors = []
    polygon_count = 0

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return 0, [ValidationError(
            file=label_path.name,
            line=0,
            message=f"Failed to read file: {e}",
        )]

    for line_num, line in enumerate(lines, start=1):
        if line.strip():
            is_valid, line_errors = parse_label_line(
                line, line_num, label_path.name, num_classes
            )
            errors.extend(line_errors)
            if is_valid:
                polygon_count += 1

    return polygon_count, errors


def validate_dataset(data_path: str, num_classes: int = 1, verbose: bool = False) -> ValidationResult:
    """
    Validate entire dataset.

    Args:
        data_path: Path to dataset root (containing images/ and labels/)
        num_classes: Number of classes in dataset
        verbose: Print detailed output

    Returns:
        ValidationResult with all findings
    """
    result = ValidationResult()
    base_path = Path(data_path)

    # Find image and label directories
    splits = ['train', 'val', 'test']
    image_dirs = []
    label_dirs = []

    for split in splits:
        img_dir = base_path / 'images' / split
        lbl_dir = base_path / 'labels' / split

        if img_dir.exists():
            image_dirs.append((split, img_dir, lbl_dir))

    if not image_dirs:
        # Try flat structure
        img_dir = base_path / 'images'
        lbl_dir = base_path / 'labels'
        if img_dir.exists():
            image_dirs.append(('all', img_dir, lbl_dir))

    if not image_dirs:
        print(f"[ERROR] No image directories found in: {base_path}")
        return result

    # Process each split
    for split_name, img_dir, lbl_dir in image_dirs:
        if verbose:
            print(f"\n[INFO] Validating split: {split_name}")
            print(f"  Images: {img_dir}")
            print(f"  Labels: {lbl_dir}")

        # Get all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        images = []
        for ext in extensions:
            images.extend(img_dir.glob(ext))
            images.extend(img_dir.glob(ext.upper()))

        result.total_images += len(images)

        if not lbl_dir.exists():
            print(f"[WARN] Label directory not found: {lbl_dir}")
            for img in images:
                result.missing_labels.append(str(img.name))
            continue

        # Check each image has corresponding label
        for img_path in images:
            label_name = img_path.stem + '.txt'
            label_path = lbl_dir / label_name

            if not label_path.exists():
                result.missing_labels.append(str(img_path.name))
                if verbose:
                    print(f"  [MISS] No label for: {img_path.name}")
            else:
                result.total_labels += 1
                polygon_count, errors = validate_label_file(label_path, num_classes)

                if errors:
                    for err in errors:
                        if err.severity == "error":
                            result.errors.append(err)
                        else:
                            result.warnings.append(err)

                        if verbose:
                            prefix = "[ERROR]" if err.severity == "error" else "[WARN]"
                            print(f"  {prefix} {err.file}:{err.line} - {err.message}")
                else:
                    result.valid_labels += 1

                result.total_polygons += polygon_count

        # Check for orphan labels (labels without images)
        labels = list(lbl_dir.glob('*.txt'))
        for label_path in labels:
            img_name_base = label_path.stem
            found_img = False
            for ext in extensions:
                pattern = ext.replace('*', img_name_base)
                if list(img_dir.glob(pattern)):
                    found_img = True
                    break
            if not found_img:
                result.missing_images.append(str(label_path.name))
                if verbose:
                    print(f"  [ORPHAN] No image for label: {label_path.name}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLO segmentation labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset root (containing images/ and labels/)')
    parser.add_argument('--num-classes', type=int, default=1,
                        help='Number of classes in dataset')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed output')

    args = parser.parse_args()

    print("=" * 50)
    print("YOLO Segmentation Label Validator")
    print("=" * 50)
    print(f"Dataset: {args.data}")
    print(f"Classes: {args.num_classes}")
    print("=" * 50)

    # Validate
    result = validate_dataset(args.data, args.num_classes, args.verbose)

    # Summary
    print("\n" + "=" * 50)
    print("Validation Summary")
    print("=" * 50)
    print(f"Total images:     {result.total_images}")
    print(f"Total labels:     {result.total_labels}")
    print(f"Valid labels:     {result.valid_labels}")
    print(f"Total polygons:   {result.total_polygons}")
    print(f"Missing labels:   {len(result.missing_labels)}")
    print(f"Orphan labels:    {len(result.missing_images)}")
    print(f"Errors:           {len(result.errors)}")
    print(f"Warnings:         {len(result.warnings)}")

    # Print errors if not verbose (verbose already printed them)
    if not args.verbose:
        if result.errors:
            print("\nErrors:")
            for err in result.errors[:10]:  # Show first 10
                print(f"  {err.file}:{err.line} - {err.message}")
            if len(result.errors) > 10:
                print(f"  ... and {len(result.errors) - 10} more errors")

        if result.missing_labels:
            print(f"\nMissing labels (first 5):")
            for name in result.missing_labels[:5]:
                print(f"  {name}")
            if len(result.missing_labels) > 5:
                print(f"  ... and {len(result.missing_labels) - 5} more")

    # Exit code
    if result.errors or len(result.missing_labels) > 0:
        print("\n[FAIL] Validation failed")
        sys.exit(1)
    else:
        print("\n[PASS] Validation passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
