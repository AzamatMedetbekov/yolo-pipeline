# Fridge Analysis Pipeline

A two-stage computer vision pipeline for detecting, segmenting, and analyzing refrigerators in images.

## Overview

The pipeline consists of two main stages:
1.  **Stage 1: Segmentation (YOLOv12)**
    -   Detects refrigerator equipment.
    -   Segments the object mask.
    -   Outputs: ROI crops and masks.
2.  **Stage 2: Analysis (ResNet-18)**
    -   **Type Classification**: Identifies fridge type (Vertical, Horizontal, etc.).
    -   **Attribute Prediction**: Extracts specs (Volume, Power, Refrigerant, etc.).

## Project Structure

```
YoloProject/
├── yolov12/            # Core Object Detection Library (Vendored Ultralytics)
├── scripts/            # Training & Processing Scripts
│   ├── train_seg.py    # Stage 1 Training
│   ├── predict_seg.py  # Stage 1 Inference
│   └── ...
├── inference/          # Stage 2 Inference Scripts
│   ├── infer_fridge_type.py
│   └── infer_fridge_attr.py
├── configs/            # YAML Configurations
├── data/               # Dataset (Images, Labels, CSVs)
└── docs/               # Documentation
```

## Documentation

Detailed guides for each part of the pipeline:

-   **[Stage 1: Segmentation Guide](docs/SEGMENTATION.md)** - Training and running YOLOv12 segmentation.
-   **[Stage 2: Class & Attributes Guide](docs/CLASSIFICATION_AND_ATTRIBUTES.md)** - Training and running Type/Attribute analysis.
-   **[Data Preparation Guide](docs/DATA_PREPARATION.md)** - Tools for labeling, converting, and splitting data.

## Quick Start

### 1. Installation

```bash
# Create environment
conda create -n yolov12 python=3.10
conda activate yolov12

# Install dependencies
pip install -r requirements.txt
pip install -e yolov12/  # Install local YOLOv12 lib
```

### 2. Run Inference (Example)

**Segmentation:**
```bash
python scripts/predict_seg.py \
    --weights runs/segment/train/weights/best.pt \
    --source data/images/test1.jpg
```

**Type Classification:**
```bash
python inference/infer_fridge_type.py \
    --weights runs/type/train/best.pt \
    --image output/test1_crop.jpg
```

### 3. Training

**Train Segmentation:**
```bash
python scripts/train_seg.py --data configs/fridge_seg.yaml
```

**Train Attributes:**
```bash
python scripts/train_fridge_attr+type_.py --mode attr --config configs/fridge_attr.yaml
```

## Data Pipeline

1.  **Collect Data**: Place images in `data/images`.
2.  **Label**:
    -   Use **LabelMe** for segmentation polygons.
    -   Convert to YOLO format: `python scripts/convert_labelme.py ...`
    -   Generate CSV templates: `python tools/make_attr_template.py ...`
    -   Fill CSVs with attribute data.
3.  **Train**: Run training scripts for respective stages.
4.  **Deploy**: Use inference scripts to process new images.
