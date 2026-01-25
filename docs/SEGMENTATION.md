# Refrigerator ROI Segmentation Pipeline

This document describes the YOLOv12 instance segmentation pipeline for detecting and segmenting refrigerator equipment in images.

## Overview

The pipeline uses YOLOv12 for instance segmentation to:
1. Detect refrigerator equipment in images
2. Generate precise segmentation masks
3. Extract ROI (Region of Interest) crops for downstream processing

## Dataset Structure

```
yolov12/data/fridge_attr10/
├── images/
│   ├── train/          # Training images (8 images)
│   │   ├── test1.jpg
│   │   ├── test2.jpg
│   │   └── ...
│   └── val/            # Validation images (2 images)
│       ├── test8.jpg
│       └── test10.jpg
├── labels/
│   ├── train/          # YOLO polygon labels
│   │   ├── test1.txt
│   │   └── ...
│   └── val/
│       ├── test8.txt
│       └── test10.txt
└── labelme_json/       # Original LabelMe annotations
    ├── test1.json
    └── ...
```

## Label Format

YOLO segmentation labels use normalized polygon coordinates:

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

- `class_id`: Integer class index (0 for fridge_equipment)
- `x, y`: Normalized coordinates in range [0, 1]
- Each line represents one polygon (object instance)

Example:
```
0 0.123456 0.234567 0.345678 0.456789 0.567890 0.678901
```

## Configuration

### Dataset Config (`configs/fridge_seg.yaml`)

```yaml
path: yolov12/data/fridge_attr10
train: images/train
val: images/val

names:
  0: fridge_equipment

nc: 1
```

## Scripts

### 1. Validate Labels

Before training, validate your labels:

```bash
python scripts/validate_labels.py --data yolov12/data/fridge_attr10 --verbose
```

Checks performed:
- Image/label file pairs exist
- All coordinates in [0, 1]
- Polygon has >= 3 points
- Class index is valid

### 2. Convert LabelMe Annotations

If you have new LabelMe annotations:

```bash
python scripts/convert_labelme.py \
    --json-dir yolov12/data/fridge_attr10/labelme_json \
    --img-dir yolov12/data/fridge_attr10/images \
    --out-dir yolov12/data/fridge_attr10/labels \
    --verbose
```

### 3. Training

Train the segmentation model:

```bash
# Basic training (50 epochs)
python scripts/train_seg.py --data configs/fridge_seg.yaml

# Short test run
python scripts/train_seg.py --data configs/fridge_seg.yaml --epochs 5 --batch 2

# Full training with custom settings
python scripts/train_seg.py \
    --data configs/fridge_seg.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 4 \
    --device 0 \
    --project runs/segment \
    --name fridge_v1
```

Training arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `yolov12s-seg` | Base model (n/s/m/l/x variants) |
| `--weights` | None | Path to pretrained weights |
| `--data` | `configs/fridge_seg.yaml` | Dataset config |
| `--epochs` | 50 | Training epochs |
| `--imgsz` | 640 | Input image size |
| `--batch` | 4 | Batch size |
| `--device` | `0` | GPU device (0, 1, cpu) |
| `--project` | `runs/segment` | Output directory |
| `--name` | `train` | Experiment name |

### 4. Inference

Run inference on images:

```bash
# Single image
python scripts/predict_seg.py \
    --weights runs/segment/train/weights/best.pt \
    --source path/to/image.jpg \
    --out-dir output/

# Directory of images
python scripts/predict_seg.py \
    --weights runs/segment/train/weights/best.pt \
    --source yolov12/data/fridge_attr10/images/val \
    --out-dir output/ \
    --conf 0.25
```

Inference outputs (per image):
| File | Description |
|------|-------------|
| `{name}_overlay.jpg` | Visualization with mask overlay |
| `{name}_mask.png` | Binary mask (white=object, black=bg) |
| `{name}_crop.jpg` | ROI crop (bounding box region) |
| `{name}_masked_crop.png` | Foreground only (transparent bg) |
| `predictions.json` | Structured output with paths + coords |

## Output Directory Structure

After training:
```
runs/segment/train/
├── weights/
│   ├── best.pt         # Best checkpoint
│   └── last.pt         # Last checkpoint
├── results.csv         # Training metrics
├── results.png         # Training plots
├── confusion_matrix.png
└── ...
```

After inference:
```
output/
├── test8_overlay.jpg
├── test8_mask.png
├── test8_crop.jpg
├── test8_masked_crop.png
├── test10_overlay.jpg
├── test10_mask.png
├── test10_crop.jpg
├── test10_masked_crop.png
└── predictions.json
```

## predictions.json Format

```json
{
  "model": "best.pt",
  "source": "yolov12/data/fridge_attr10/images/val",
  "config": {
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "image_size": 640
  },
  "predictions": [
    {
      "image": "test8.jpg",
      "class": "fridge_equipment",
      "class_id": 0,
      "confidence": 0.9234,
      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 580,
        "y2": 720,
        "width": 460,
        "height": 640
      },
      "outputs": {
        "overlay": "test8_overlay.jpg",
        "mask": "test8_mask.png",
        "crop": "test8_crop.jpg",
        "masked_crop": "test8_masked_crop.png"
      }
    }
  ]
}
```

## Troubleshooting

### Model not found
If YOLOv12 pretrained weights are not available locally, the training script will attempt to use the YAML config and Ultralytics will download weights automatically.

### CUDA out of memory
Reduce batch size: `--batch 2` or `--batch 1`

### Label validation errors
Run `validate_labels.py` with `--verbose` to see detailed errors. Common issues:
- Coordinates outside [0, 1] range
- Fewer than 3 polygon points
- Missing label files

### Slow training on CPU
Add `--device cpu` explicitly and consider reducing image size: `--imgsz 320`

## Integration with Attribute Network

After segmentation, use the ROI crops as input to the attribute prediction network:

```python
from PIL import Image
import json

# Load predictions
with open('output/predictions.json') as f:
    preds = json.load(f)

# Use masked crops for attribute prediction
for pred in preds['predictions']:
    crop_path = f"output/{pred['outputs']['masked_crop']}"
    # Process with attribute network...
```

## Class Mapping

Current class mapping:
| Class ID | Name | Original Label |
|----------|------|----------------|
| 0 | fridge_equipment | showcase |

To add more classes, edit the `DEFAULT_CLASS_MAP` in `scripts/convert_labelme.py`.
