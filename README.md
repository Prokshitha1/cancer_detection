# Skin Cancer Detection App (HAM10000)

This repository provides an end-to-end starter pipeline for skin lesion analysis using the HAM10000 dataset:

- **Data preparation** for HAM10000 metadata and train/val/test splits
- **Transfer learning classifier** (EfficientNetB0) for:
  - Binary classification (`benign` vs `malignant`)
  - Multi-class classification (`dx` labels)
- **Grad-CAM** visualization for interpretability
- **Streamlit app** for image upload + prediction + heatmap overlay

> Note: Training deep models can be compute intensive. The app supports loading a trained model if available and otherwise gives setup guidance.

## 1) Download dataset

Use your commands:

```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ./data
unzip -q ./data/skin-cancer-mnist-ham10000.zip -d ./data
```

Expected files after unzip:
- `data/HAM10000_metadata.csv`
- `data/HAM10000_images_part_1/`
- `data/HAM10000_images_part_2/`

## 2) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Prepare splits

```bash
python scripts/prepare_ham10000.py --data-dir data --output-dir data/processed --test-size 0.15 --val-size 0.15
```

This generates:
- `data/processed/metadata_with_paths.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## 4) Train classifier

### Binary model (benign vs malignant)

```bash
python scripts/train_classifier.py \
  --processed-dir data/processed \
  --output-dir models/binary \
  --task binary \
  --img-size 224 \
  --batch-size 32 \
  --epochs 10
```

### Multi-class model (HAM10000 classes)

```bash
python scripts/train_classifier.py \
  --processed-dir data/processed \
  --output-dir models/multiclass \
  --task multiclass \
  --img-size 224 \
  --batch-size 32 \
  --epochs 10
```

Outputs include:
- `best_model.keras`
- `label_map.json`
- `metrics.json`
- `confusion_matrix.csv`
- `classification_report.json`

## 5) Run app

```bash
streamlit run app.py
```

In the UI you can:
- Upload a dermoscopic image
- Choose binary or multiclass model path
- View predicted class/probability
- Generate **Grad-CAM** overlay

## Project structure

- `app.py` – Streamlit interface
- `scripts/prepare_ham10000.py` – metadata + splits
- `scripts/train_classifier.py` – training + evaluation
- `src/cancer_detection/gradcam.py` – Grad-CAM utility
- `src/cancer_detection/data_utils.py` – data utilities

## Suggested next steps

- Add U-Net segmentation stage and chain segmented ROI into classifier
- Add class weighting or focal loss for imbalance
- Add model registry/versioning and experiment tracking
- Containerize with Docker for easier deployment
