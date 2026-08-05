# Preprocessing and Dataset Loading Documentation

This document explains the data pipeline for the Deepfake Physics Detector, focusing on how raw videos are processed and loaded into the training pipeline.

## 🛠️ Preprocessing Pipeline (v2)
The preprocessing step is critical. Since this project focuses on *physical* and *biological* features (rather than raw pixels for a CNN), the faces must be perfectly normalized. We use the **v2 Preprocessing Pipeline**.

### 1. Face Detection and Tracking
We use **MediaPipe Face Mesh** to detect the face and extract dense 3D landmarks. 

### 2. Eye Alignment & Rotation
To ensure biological features (like lighting and rPPG) are consistent, the face is rotated so that the eyes are perfectly horizontal. This removes variance caused by head-tilt (roll).

### 3. Face Cropping
The aligned face is cropped to a strict **256x256** pixel bounding box.

### 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
We apply CLAHE to the cropped face. Standard histogram equalization can over-amplify noise in relatively uniform regions (like skin). CLAHE prevents this by dividing the image into tiles and applying equalization locally. This helps reveal hidden textures and subtle deepfake blending artifacts.

### 5. Bilateral Filtering
Finally, we apply a Bilateral Filter. Unlike a standard Gaussian blur that blurs everything, a bilateral filter preserves sharp edges (like the nose, eyes, and jawline) while smoothing out background noise. This is critical for our Frequency (FFT) and Motion Blur extractors to work accurately without being confused by camera grain.

---

## 🗃️ Dataset Loader (`data/dataset_loader.py`)
Once videos are preprocessed and features are extracted into `.npz` files, `dataset_loader.py` handles feeding this data into TensorFlow.

### The "Strict Split" Logic
To guarantee a fair evaluation and prevent data leakage, we implemented a custom subset sampling strategy (`get_strict_datasets()`):

1. **Quotas**: The script grabs exactly **1,000 Real**, **750 Deepfakes**, and **750 Face2Face** videos.
2. **Train/Val Split**: This pool of 2,500 videos is vigorously shuffled and split **80/20**:
   - **Training Set**: ~2,000 videos
   - **Validation Set**: ~500 videos
3. **50-50 Class Balancing**: The `tf.data.Dataset.sample_from_datasets` API is used to pull exactly 50% Real and 50% Fake samples during training batches, preventing class imbalance.
4. **Test Set (Zero Leakage)**: The script takes **every single remaining video** from the dataset that was NOT used in the Train/Val pool and places it into the Test set. This guarantees the model is tested on unseen data.

### TensorFlow Data Pipeline
The `build_tf_dataset` function uses `tf.data.Dataset.from_generator` to asynchronously load the `.npz` arrays from disk during training, which prevents RAM exhaustion.
