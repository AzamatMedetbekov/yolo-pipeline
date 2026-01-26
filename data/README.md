DATA DIRECTORY TEMPLATE

This folder is the single source of truth for training, validation,
and inference data used by the segmentation pipeline.

Expected structure:

data/
  images/
    train/          # training images
    val/            # validation images
  labels/
    train/          # YOLO segmentation labels for training images
    val/            # YOLO segmentation labels for validation images
  labelme_json/     # raw LabelMe JSON annotations (optional)

Label format (segmentation):
  class_id x1 y1 x2 y2 x3 y3 ... xn yn
  - coordinates are normalized to [0, 1]
  - each line is one instance (polygon)

Quick check:
  python scripts/validate_labels.py --data data --num-classes 1 --verbose
