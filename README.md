# FoodCalorieEstimator
This repo implements a food calorie estimator from image, to help people measure their calorie intake, give them insight about what they eat!
# 🍽️ Food Calorie Estimation Pipeline

A Google Colab image processing pipeline that takes a photo of a plate of food, segments each food item, estimates its volume, and calculates total calorie content.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Branch Structure](#branch-structure)
- [Phases](#phases)
  - [Phase 0 — Setup](#phase-0--setup)
  - [Phase 1 — Input Pipeline](#phase-1--input-pipeline)
  - [Phase 2 — Segmentation](#phase-2--segmentation)
  - [Phase 3 — Classification](#phase-3--classification)
  - [Phase 4 — Volume Estimation](#phase-4--volume-estimation)
  - [Phase 5 — Calorie Database](#phase-5--calorie-database)
  - [Phase 6 — Evaluation & Debug UI](#phase-6--evaluation--debug-ui)
- [Metrics](#metrics)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

---

## Overview

Given a single image of a food plate, this pipeline:

1. Segments the image into per-food-item masks
2. Classifies each segment (what food is it?)
3. Estimates the volume of each segment
4. Looks up calorie density from a swappable dataset
5. Outputs a per-item and total calorie estimate with a debug overlay

Each module is designed to be swappable — segmentation model, classifier, volume strategy, and calorie database can all be changed independently via a common interface.

---

## Pipeline Architecture

```
Image Input
    │
    ▼
Preprocessing (resize, normalize, plate crop)
    │
    ▼
Segmentation ──► per-mask binary masks
    │
    ▼
Classification ──► per-mask food label
    │
    ▼
Volume Estimation ──► per-mask volume (cm³)
    │
    ▼
Calorie Lookup ──► per-mask kcal estimate
    │
    ▼
Output: annotated image + per-item table + total kcal
```

---

## Branch Structure

This project uses a **`main` / `dev` / feature branches** workflow:

- `main` — stable only; tagged releases after each phase is validated
- `dev` — integration branch; all feature branches PR into `dev` first
- `feature/*` — one branch per phase or sub-experiment

For phases with multiple competing implementations (segmentation, classification, volume), a `-base` branch defines the shared interface and test fixtures. Sub-branches implement each approach and merge back into `-base` after benchmarking.

```
main
 └── dev
      ├── feature/setup
      ├── feature/input-pipeline
      ├── feature/segmentation-base
      │    ├── seg/sam
      │    ├── seg/yolov8
      │    └── seg/maskrcnn
      ├── feature/classification-base
      │    ├── clf/clip
      │    ├── clf/food101
      │    └── clf/vlm
      ├── feature/volume-base
      │    ├── vol/plate-prior
      │    ├── vol/reference-obj
      │    └── vol/midas
      ├── feature/calorie-db
      └── feature/eval-ui
```

Each `-base` branch contains a `COMPARISON.md` file that is filled in during benchmarking to document model choice rationale before moving to the next phase.

---

## Phases

### Phase 0 — Setup

**Branch:** `feature/setup`  
**Depends on:** —

Sets up the Colab environment, installs all dependencies, and establishes the project folder structure.

**Deliverables:**
- `requirements.txt` / `setup notebook cell`
- Folder structure: `data/`, `models/`, `outputs/`, `src/`
- GPU availability check
- Sample test images (5–10 labelled plates for benchmarking)

---

### Phase 1 — Input Pipeline

**Branch:** `feature/input-pipeline`  
**Depends on:** `feature/setup`

Handles all image ingestion and preprocessing before segmentation.

**Deliverables:**
- Colab file upload widget + URL input option
- Resize to model input resolution (configurable)
- Normalize pixel values (ImageNet stats by default)
- Optional: plate bounding box crop (YOLO or U2-Net) to remove background noise before segmentation
- Unit test: input → preprocessed tensor, assert shape and dtype

---

### Phase 2 — Segmentation

**Base branch:** `feature/segmentation-base`  
**Depends on:** `feature/input-pipeline`

Defines the shared `Segmentor` interface and benchmarking harness. Three implementations are developed in parallel sub-branches and compared before one is carried forward.

```python
class Segmentor:
    def segment(self, image: np.ndarray) -> List[Mask]:
        """Returns a list of binary masks, one per detected food region."""
        ...
```

**Shared (base branch):**
- `Mask` dataclass: binary mask, bounding box, confidence
- Mask overlay visualizer (matplotlib)
- IoU metric vs hand-labelled ground truth masks
- `COMPARISON.md` template

**Sub-branches:**

| Branch | Model | Notes |
|---|---|---|
| `seg/sam` | Segment Anything Model (Meta) | Zero-shot, auto-mask mode; no food-specific training needed; returns category-agnostic masks |
| `seg/yolov8` | YOLOv8-seg | Fast; needs a food-domain checkpoint (e.g. fine-tuned on Food-101 or UECFOOD); gives labels alongside masks |
| `seg/maskrcnn` | Mask R-CNN (torchvision) | Classic; well-documented; slower than YOLO; good baseline |

**Benchmark criteria:** mask IoU on test set, inference time on Colab T4, false positive rate on plate background.

---

### Phase 3 — Classification

**Base branch:** `feature/classification-base`  
**Depends on:** `feature/segmentation-base` (winning model merged)

Classifies each segmented region into a food label. Takes a cropped-and-masked image patch per segment as input.

```python
class Classifier:
    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """Returns (food_label, confidence)."""
        ...
```

**Shared (base branch):**
- Crop-per-mask utility (mask → tight bounding box crop with background zeroed)
- Label normalizer (maps model output to canonical names for DB lookup)
- Accuracy metric vs ground truth labels
- `COMPARISON.md` template

**Sub-branches:**

| Branch | Model | Notes |
|---|---|---|
| `clf/clip` | CLIP (OpenAI, via `open_clip`) | Zero-shot; provide a text prompt list of food names; no training; flexible label set |
| `clf/food101` | ResNet/ViT fine-tuned on Food-101 | Fast inference; fixed 101-class label set; good accuracy on common foods |
| `clf/vlm` | LLaVA (local) or GPT-4V (API) | Highest flexibility; can describe arbitrary foods; optional — requires API key or larger GPU |

**Benchmark criteria:** top-1 accuracy on test set labels, inference time, handling of unseen/mixed foods.

---

### Phase 4 — Volume Estimation

**Base branch:** `feature/volume-base`  
**Depends on:** `feature/classification-base` (winning classifier merged)

Estimates the physical volume (cm³) of each food segment. This is the hardest module — all three strategies are available and selectable at runtime.

```python
class VolumeEstimator:
    def estimate(self, mask: Mask, image: np.ndarray) -> float:
        """Returns estimated volume in cm³."""
        ...
```

**Shared (base branch):**
- `VolumeEstimator` abstract base class
- Volume error metric (vs known portion sizes in test set)
- `COMPARISON.md` template

**Sub-branches:**

| Branch | Strategy | Notes |
|---|---|---|
| `vol/plate-prior` | Fixed plate diameter assumption (26 cm standard) | Simplest; ~10 lines; start here; gives pixel-per-cm scale factor |
| `vol/reference-obj` | Detect known object (fork, coin) via contour detection | More accurate scale estimation; requires reference object in frame |
| `vol/midas` | MiDaS / DepthAnything depth model | Most accurate; runs on T4; gives per-pixel depth map for volume integration |

> **Recommended starting point:** `vol/plate-prior` for the PoC. Upgrade to `vol/midas` once the end-to-end pipeline is validated.

---

### Phase 5 — Calorie Database

**Branch:** `feature/calorie-db`  
**Depends on:** `feature/volume-base` (winning strategy merged)

Provides a swappable calorie lookup layer. Converts `(food_label, volume_cm3)` → `kcal`.

```python
class CalorieDB:
    def lookup(self, label: str, volume_cm3: float) -> float:
        """Returns estimated kcal for a given food and volume."""
        ...
```

**Implementations (in same branch, switchable via config):**

| Backend | Description |
|---|---|
| `CSVCalorieDB` | Hand-crafted CSV of ~50 common foods with density (g/cm³) and kcal/100g. Default for PoC. |
| `USDACalorieDB` | Queries USDA FoodData Central API. Requires API key. Falls back to CSV on miss. |

**Label matching:** fuzzy string matching (`rapidfuzz`) between classifier output and DB keys to handle label variations (e.g. `"fried rice"` → `"Rice, fried"`).

**CSV schema:**

```
label, density_g_per_cm3, kcal_per_100g
rice, 0.85, 130
chicken breast, 1.05, 165
broccoli, 0.37, 34
...
```

---

### Phase 6 — Evaluation & Debug UI

**Branch:** `feature/eval-ui`  
**Depends on:** `feature/calorie-db`

Ties the full pipeline together with a visual debug overlay and metrics dashboard.

**Deliverables:**
- Annotated output image: segmentation mask overlay + food label + kcal per region
- Per-item results table (label, volume cm³, kcal)
- Total calorie summary
- Metrics report cell: IoU, volume error (MAE), end-to-end kcal error (MAE, MAPE) on test set
- `run_pipeline(image_path, seg_model, clf_model, vol_strategy, db_backend)` single entry point

---

## Metrics

| Metric | Module | Description |
|---|---|---|
| Mask IoU | Segmentation | Intersection-over-union vs hand-labelled masks on test set |
| Volume MAE | Volume estimation | Mean absolute error (cm³) vs known portion sizes |
| Calorie MAE | End-to-end | Mean absolute error (kcal) vs ground truth calorie counts |
| Calorie MAPE | End-to-end | Mean absolute percentage error — more interpretable than MAE |

Test set: 10–20 labelled plate images with known foods, portion weights, and calorie counts (sourced manually or from a public dataset such as ECOC or UECFOOD256).

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/food-calorie-pipeline.git
cd food-calorie-pipeline

# Switch to dev to see the latest integrated work
git checkout dev

# Or start from a specific phase
git checkout feature/segmentation-base
```

Open the corresponding `.ipynb` notebook in Google Colab, connect to a T4 GPU runtime, and run all cells.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Model inference |
| `ultralytics` | YOLOv8-seg |
| `segment-anything` | SAM |
| `open_clip_torch` | CLIP classification |
| `transformers` | VLM / depth models (MiDaS, LLaVA) |
| `opencv-python` | Image processing, contour detection |
| `matplotlib` | Visualization and debug overlay |
| `rapidfuzz` | Fuzzy label matching for calorie DB |
| `pandas` | CSV calorie dataset handling |
| `requests` | USDA API calls |

---
