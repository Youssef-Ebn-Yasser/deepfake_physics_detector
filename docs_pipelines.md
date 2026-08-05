# Pipelines Documentation

This document explains every script inside the `pipelines/` directory. These scripts are the executable entry points for training, evaluating, and extracting features from videos.

---

## 1. `extract_features.py` (Baseline Feature Extraction)

**Purpose**: The first step before any training can begin. Processes every raw `.mp4` video in the `DataSets/` directory tree and saves extracted feature vectors as compressed `.npz` files into `DataSets/features/`.

**How it works**:
1. **Video Discovery**: Scans all configured source paths (e.g., `original_sequences/youtube`, `manipulated_sequences/Deepfakes`) for `.mp4` files at the specified compression level (`c23` or `c40`).
2. **Frame Sampling**: For each video, it samples a fixed number of frames evenly distributed across its duration.
3. **Preprocessing (v1)**: Applies the baseline preprocessing pipeline (face detection, cropping).
4. **Feature Extraction**: Calls all four extractor modules sequentially on each frame batch:
   - `extract_lens_distortion_features()`
   - `extract_motion_blur_features()`
   - `extract_biomechanics_features()`
   - `extract_lighting_sh_features()`
5. **Saving**: Saves the final feature dictionary as a `.npz` archive. Each `.npz` file represents a single video and contains arrays for keys `f1`, `f2`, `f3`, `f4`, and `f5`.

**Run command**:
```bash
python pipelines/extract_features.py
```

---

## 2. `extract_features_v2.py` (Advanced Feature Extraction)

**Purpose**: The advanced version of the extraction pipeline. Identical in logic to `extract_features.py`, but uses the **v2 Preprocessing Pipeline** (Eye Alignment, CLAHE, and Bilateral Filtering) before running the extractors. Saves output to `DataSets/features_v2/`.

The v2 pipeline produces higher-quality, more standardized face crops, which significantly improves the signal-to-noise ratio for the biological extractors (rPPG, Spherical Harmonics).

**Run command**:
```bash
python pipelines/extract_features_v2.py
```

---

## 3. `train_subsystem1.py` (Train Hardware Optics Encoder)

**Purpose**: Independently pre-trains only the **Sub-System 1 (Hardware Optics) Encoder** on the hardware physics features ($f_1$, $f_2$, $f_5$) before it is integrated into the full fusion model.

**Why pre-train independently?**
Pre-training each sub-system in isolation ensures the encoder learns the task-specific discriminative features of its domain *before* it gets pulled in multiple directions by the shared fusion loss. This leads to a much more stable fusion training.

**Training Details**:
- **Inputs**: Loads only the `f1_lens_distortion`, `f2_motion_blur`, and `f5_frequency_fft` keys from the dataset.
- **Loss**: Binary Cross-Entropy on its own output head (`sub1_logits`).
- **Callbacks**:
  - `ModelCheckpoint`: Saves to `checkpoints/subsystem1_best.keras` when validation accuracy improves.
  - `EarlyStopping` (patience=5): Stops training if the model stops improving.
  - `ReduceLROnPlateau`: Halves the learning rate when validation loss plateaus for 3 epochs.
- **Output**: `checkpoints/subsystem1_best.keras` and `checkpoints/subsystem1_final.keras`.

**Run command**:
```bash
python pipelines/train_subsystem1.py
```

---

## 4. `train_subsystem2.py` (Train Biological Encoder)

**Purpose**: Independently pre-trains only the **Sub-System 2 (Biological) Encoder** on biological features ($f_3$, $f_4$).

**Special Feature — Balanced Sampling Mode**:
This script includes an optional `--balanced` flag that creates a custom subset of exactly 1,000 original, 1,000 Deepfakes, and 1,000 Face2Face samples for training, preventing the biological domain from being biased by class imbalance.

**Training Details**:
- **Inputs**: Loads only `f3_biomechanics` and `f4_lighting_sh` keys.
- **Loss**: Binary Cross-Entropy on `sub2_logits`.
- **Callbacks**: Same as Sub-System 1 (Checkpoint, EarlyStopping, ReduceLR).
- **Output**: `checkpoints/subsystem2_best.keras` and `checkpoints/subsystem2_final.keras`.

**Run commands**:
```bash
# Standard mode
python pipelines/train_subsystem2.py

# Balanced subset mode
python pipelines/train_subsystem2.py --balanced
```

---

## 5. `train_fusion_1_2_mid.py` (Train Mid-Level Fusion Model)

**Purpose**: The main training script for the full **Mid-Level Bidirectional Cross-Attention Fusion** model. This is the primary model for deepfake detection.

**Two-Stage Training Strategy**:
This is the most important concept of this training script. It uses curriculum learning:

- **Stage 1 — Fusion Block Warm-Up (Encoders Frozen)**: The pre-trained Sub-System 1 and Sub-System 2 encoder weights are loaded from `checkpoints/`. The encoder layers are then **frozen** (non-trainable), so only the Cross-Attention Fusion Block and the final Classification Head are trained. This teaches the fusion mechanism how to combine the two domains without corrupting the pre-learned features.

- **Stage 2 — End-to-End Fine-Tuning (All Layers Unfrozen)**: All layers are unfrozen and the entire model is fine-tuned jointly with a very low learning rate. This allows the sub-system encoders to adapt to the fusion objective.

**Loss Function**:
Uses a multi-output loss setup with **Auxiliary Losses**:
- `fusion_logits`: **Focal Loss** (gamma=2.0) — Penalizes hard-to-classify samples more, great for imbalanced data.
- `sub1_logits` + `sub2_logits`: Standard BCE as **auxiliary losses** (weight=0.2). This forces the individual encoders to remain meaningful classifiers even after fusion, preventing representation collapse.

**Run command**:
```bash
python pipelines/train_fusion_1_2_mid.py

# With explicit checkpoint paths
python pipelines/train_fusion_1_2_mid.py --sub1 checkpoints/subsystem1_best.keras --sub2 checkpoints/subsystem2_best.keras
```

---

## 6. `train_master.py` (Train End-to-End Master Model)

**Purpose**: An alternative training path that trains the complete **Master Physics Detector** end-to-end. Uses the same two-stage strategy as the fusion trainer but operates through the `MasterModel` wrapper class.

**Difference from `train_fusion_1_2_mid.py`**:
Architecturally very similar, but uses the `build_master_physics_detector()` function. The output head is labeled `master_logits` instead of `fusion_logits`.

**Run command**:
```bash
python pipelines/train_master.py
```

---

## 7. `plot_features.py` (Feature Visualization)

**Purpose**: A diagnostic and analysis script that generates visualizations of the extracted features to help understand the data quality and feature distributions.

**What it generates**:
- **t-SNE Plots**: Reduces the high-dimensional feature vectors ($f_1$ through $f_5$) to 2D using t-SNE and plots real vs. fake clusters. If the clusters are well-separated, the feature is discriminative.
- **Feature Health Reports**: Checks for NaN values, zero vectors, and extreme outliers in the `.npz` files.
- Saves outputs to `results/`.

**Run command**:
```bash
python pipelines/plot_features.py
```

---

## Recommended Execution Order

```
1. python pipelines/extract_features.py       # (or extract_features_v2.py)
2. python pipelines/train_subsystem1.py       # Optional pre-training
3. python pipelines/train_subsystem2.py       # Optional pre-training
4. python pipelines/train_fusion_1_2_mid.py   # Main model
5. python pipelines/plot_features.py          # Diagnostics
```

Or use the automation script to run all of the above in sequence for both feature sets:
```bash
python run_all_experiments.py
```
