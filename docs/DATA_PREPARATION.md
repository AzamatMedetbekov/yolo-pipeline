# Data Preparation Guide

This guide covers the tools available for preparing the dataset for the Fridge Analysis pipeline.

## Overview

The data preparation workflow generally involves:
1.  **Annotation**: Labeling images using LabelMe (creating JSONs).
2.  **Conversion**: Converting LabelMe JSONs to YOLO format (for Segmentation).
3.  **Template Generation**: Creating CSV templates for Type/Attribute labeling.
4.  **Splitting**: Organizing data into Train/Val sets.
5.  **Validation**: Checking label integrity.

## 1. Converting LabelMe to YOLO

If you annotated your images with LabelMe, use `scripts/convert_labelme.py` to generate YOLO segmentation labels.

```bash
python scripts/convert_labelme.py \
    --json-dir data/labelme_json \
    --img-dir data/images \
    --out-dir data/labels \
    --verbose
```

**Arguments:**
-   `--json-dir`: Directory containing `.json` files from LabelMe.
-   `--img-dir`: Directory containing corresponding images.
-   `--out-dir`: Where to save the generated `.txt` YOLO labels.
-   `--class-map`: (Optional) Path to a JSON file mapping label names to class IDs. Default maps `showcase` -> `0`.

## 2. Validating Labels

Ensure your segmentation labels are valid before training.

```bash
python scripts/validate_labels.py \
    --data data/fridge_attr10 \
    --verbose
```

This script checks:
-   Existence of image/label pairs.
-   Coordinate normalization (0-1).
-   Polygon validity (>= 3 points).

## 3. Creating CSV Templates

For Stage 2 (Classification/Attributes), labels are stored in CSV files. Use the generator tools to create blank templates based on your image directories.

**For Type Classification:**
```bash
python tools/make_type_template.py
```
*Generates: `data/fridge_type_labels.csv`*

**For Attribute Prediction:**
```bash
python tools/make_attr_template.py
```
*Generates: `data/fridge_attr10_labels.csv`*

**Workflow:**
1.  Run the tool.
2.  Open the generated CSV.
3.  Fill in the `type_class` or attribute columns for each image.
4.  Save and use for training.

## 4. Splitting Train/Val

To split a unified dataset into training and validation folders:

```bash
python tools/split_train_val.py
```

*Note: This script currently uses hardcoded paths in `tools/split_train_val.py`. You may need to edit `BASE_DIR` in the script to match your project location.*

## 5. PDF to Images

If your source data is in PDF format (catalogs), convert them to images:

```bash
python tools/pdf_to_images.py \
    --input documents/catalog.pdf \
    --out-dir data/extracted_images \
    --dpi 300
```
