# Physics-Based Deepfake Detector — Complete Project Pipeline Documentation

**Version**: 2.0 (Sub-Systems 1 & 2 with Mid-Level Fusion)  
**Author**: Deepfake Physics Research Team  
**Framework**: TensorFlow / Keras  
**Dataset**: FaceForensics++ (FF++), Cross-validation on Celeb-DF v2

---

## 🗂️ Table of Contents

1. [Project Philosophy and Motivation](#1-project-philosophy-and-motivation)
2. [Project Directory Structure](#2-project-directory-structure)
3. [Configuration System](#3-configuration-system)
4. [Phase 0: Environment Setup and Installation](#4-phase-0-environment-setup-and-installation)
5. [Phase 1: Dataset Organization](#5-phase-1-dataset-organization)
6. [Phase 2: Preprocessing Pipeline (v1 and v2)](#6-phase-2-preprocessing-pipeline)
7. [Phase 3: Feature Extraction in Depth](#7-phase-3-feature-extraction-in-depth)
   - [Feature 1 — Lens Distortion (f1)](#feature-1--lens-distortion-f1)
   - [Feature 2 — Motion Blur (f2)](#feature-2--motion-blur-f2)
   - [Feature 5 — Frequency FFT Spectrum (f5)](#feature-5--frequency-fft-spectrum-f5)
   - [Feature 3 — Biomechanics (f3)](#feature-3--biomechanics-f3)
   - [Feature 4 — Lighting Spherical Harmonics (f4)](#feature-4--lighting-spherical-harmonics-f4)
8. [Phase 4: Data Loading and the TF.Data Pipeline](#8-phase-4-data-loading-and-the-tfdata-pipeline)
9. [Phase 5: The Dataset Split Strategy](#9-phase-5-the-dataset-split-strategy)
10. [Phase 6: Neural Network Architecture (Deep Dive)](#10-phase-6-neural-network-architecture-deep-dive)
    - [SubSystem1Encoder — Hardware Optics](#subsystem1encoder--hardware-optics-encoder)
    - [SubSystem2Encoder — Biological Domain](#subsystem2encoder--biological-domain-encoder)
    - [MidLevelFusionBlock12 — Bidirectional Cross-Attention](#midlevelfusionblock12--bidirectional-cross-attention-fusion)
    - [Classification Head](#classification-head)
    - [MasterFusionBlock — Alternative Architecture](#masterfusionblock--alternative-architecture)
11. [Phase 7: The Two-Stage Training Strategy](#11-phase-7-the-two-stage-training-strategy)
    - [Stage 1: Fusion Warm-Up (Encoders Frozen)](#stage-1-fusion-warm-up-encoders-frozen)
    - [Stage 2: End-to-End Fine-Tuning](#stage-2-end-to-end-fine-tuning)
12. [Phase 8: Loss Functions and Auxiliary Supervision](#12-phase-8-loss-functions-and-auxiliary-supervision)
13. [Phase 9: Training Callbacks](#13-phase-9-training-callbacks)
14. [Phase 10: Evaluation Metrics](#14-phase-10-evaluation-metrics)
15. [Phase 11: Cross-Dataset Evaluation (Celeb-DF)](#15-phase-11-cross-dataset-evaluation-celeb-df)
16. [Phase 12: The Automation Runner](#16-phase-12-the-automation-runner)
17. [Feature Set v1 vs v2 Comparison](#17-feature-set-v1-vs-v2-comparison)
18. [Config Reference](#18-config-reference)
19. [Common Errors and Fixes](#19-common-errors-and-fixes)
20. [Mathematical Appendix](#20-mathematical-appendix)
21. [Future Improvements Roadmap](#21-future-improvements-roadmap)

---

## 1. Project Philosophy and Motivation

### 1.1 Why Physics-Based Detection?

The deepfake detection landscape is dominated by two paradigms:
1. **Pixel-Level CNN Detectors**: These models (e.g., XceptionNet, EfficientNet fine-tuned on deepfake datasets) learn to spot visual compression artifacts, blending seams, and GAN-generated textures. They are highly accurate **in-distribution** (when tested on the same dataset they were trained on) but notoriously fail to **generalize** to new deepfake methods unseen during training.
2. **Physics-Based Detectors**: Instead of learning what a deepfake *looks* like (which changes with every new GAN or Diffusion model), they model what a deepfake *cannot replicate*: the physical laws governing the camera and the biology of a real human being.

The fundamental insight is:

> *A deepfake generator synthesizes pixels. It cannot synthesize the physical world.*

Regardless of how photorealistic a deepfake becomes, it will continue to violate:
- **Camera Optics**: Brown-Conrady lens distortion coefficients. Motion blur dictated by shutter speed. High-frequency spectral signatures of camera sensors.
- **Human Physiology**: The periodic color oscillations caused by blood flow (rPPG). The 3D lighting environment (Spherical Harmonics) that must be geometrically consistent with the ambient scene.

This is why this system is inherently more **generalizable** across datasets and future deepfake generation methods.

---

### 1.2 Design Goals

| Goal | Implementation |
|---|---|
| Generalize to unseen deepfake methods | Physics-based features that are generative-model-agnostic |
| Handle heavy video compression (c40) | Pre-filtering (Gaussian, Bilateral), compression-robust features |
| Multi-modal reasoning | Bidirectional Cross-Attention Fusion between hardware and biological domains |
| Prevent over-fitting | Two-stage curriculum training, Dropout, EarlyStopping |
| Reproducibility | Fixed random seeds, deterministic data splitting |

---

## 2. Project Directory Structure

```
deepfake_physics_detector/
│
├── config/
│   ├── default_config.yaml          ← All hyperparameters and paths
│   └── feature_scaling_stats.json   ← Per-feature mean/std for normalization
│
├── data/
│   ├── dataset_loader.py            ← TF.Data pipeline and dataset splits
│   └── preprocessing.py             ← v1 and v2 preprocessing functions
│
├── extractors/
│   ├── __init__.py
│   ├── lens_distortion.py           ← f1: Camera lens geometry analysis
│   ├── motion_blur.py               ← f2: Farneback optical flow analysis
│   ├── biomechanics.py              ← f3: 3D pose kinematics & rPPG
│   ├── lighting_sh.py               ← f4: Spherical Harmonics lighting
│   └── frequency_fft.py             ← f5: 2D FFT azimuthal spectrum
│
├── models/
│   ├── __init__.py
│   ├── subsystem_1.py               ← SubSystem1 standalone model builder
│   ├── subsystem_2.py               ← SubSystem2 standalone model builder
│   ├── master_fusion.py             ← SubSystem1Encoder, SubSystem2Encoder, MasterFusionBlock
│   └── fusion_1_2_mid.py            ← MidLevelFusionBlock12, build_fusion_1_2_mid
│
├── pipelines/
│   ├── extract_features.py          ← v1 batch feature extraction
│   ├── extract_features_v2.py       ← v2 batch feature extraction (advanced preprocessing)
│   ├── train_subsystem1.py          ← Independent Sub-System 1 training
│   ├── train_subsystem2.py          ← Independent Sub-System 2 training
│   ├── train_fusion_1_2_mid.py      ← Main Mid-Level Fusion training (two-stage)
│   ├── train_master.py              ← Master model training
│   └── plot_features.py             ← t-SNE and feature health diagnostics
│
├── DataSets/
│   ├── FaceForensics++/             ← Raw videos (not tracked by git)
│   ├── Celeb-DF-v2/                 ← Cross-dataset evaluation videos
│   ├── features/                    ← Pre-extracted .npz files (v1)
│   └── features_v2/                 ← Pre-extracted .npz files (v2)
│
├── checkpoints/                     ← Saved model weights (not tracked by git)
├── results/                         ← Plots, confusion matrices, ROC curves
├── run_all_experiments.py           ← Automation runner for full benchmark
├── eval_celebdf.py                  ← Cross-dataset evaluation on Celeb-DF
├── requirements.txt
└── README.md
```

---

## 3. Configuration System

All project hyperparameters are centralized in `config/default_config.yaml`. This eliminates the need to modify source code to change training settings.

```yaml
dataset:
  root: DataSets
  compression: c40              # c23 (lower compression) or c40 (higher)
  real_sources:
    - original_sequences/youtube
    - original_sequences/actors
  fake_sources:
    - manipulated_sequences/Deepfakes
    - manipulated_sequences/Face2Face
    - manipulated_sequences/FaceSwap
    - manipulated_sequences/FaceShifter
    - manipulated_sequences/NeuralTextures
  split_seed: 42                # Controls all dataset splits (reproducibility)
  features_dir: DataSets/features_v2   # ← Switch between v1 and v2

preprocessing:
  n_frames: 16                  # Frames sampled per video
  target_size: [256, 256]       # Output face crop resolution
  gaussian_blur_sigma: 0.8      # c40 artifact suppression

extractors:
  feature_dim: 64               # Output vector dimension per extractor
  motion_blur:
    farneback_iterations: 5
    farneback_winsize: 21

model:
  embed_dim: 256                # Internal embedding dimension
  master_aux_weight: 0.2        # Weight of auxiliary losses

training:
  batch_size: 32
  learning_rate_stage1: 0.001   # Stage 1: Fusion block only
  learning_rate_stage2: 0.0001  # Stage 2: Full fine-tuning
  epochs_stage1: 10
  epochs_stage2: 20
  subsystem_epochs: 20          # For independent pre-training
  subsystem_lr: 0.0001
```

### 3.1 Switching Feature Sets

To switch between v1 and v2 features without editing source code, simply change `features_dir` in the config, or let the automation script do it for you:

```python
# run_all_experiments.py does this automatically:
cfg['dataset']['features_dir'] = 'DataSets/features'    # v1
cfg['dataset']['features_dir'] = 'DataSets/features_v2' # v2
```

---

## 4. Phase 0: Environment Setup and Installation

### 4.1 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 200 GB (for FF++ full dataset) |
| GPU | (optional) | NVIDIA 8GB+ (Linux/WSL2) |

> **Note on GPU**: TensorFlow >= 2.11 does not support native GPU on Windows. If you are on Windows, either use the CPU (slower but functional) or set up WSL2 with CUDA for GPU acceleration.

### 4.2 Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/Youssef-Ebn-Yasser/deepfake_physics_detector.git
cd deepfake_physics_detector

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

### 4.3 Dependencies (requirements.txt)

```
tensorflow>=2.15.0        # Core deep learning framework (Keras built-in)
opencv-python>=4.8.0      # Image processing, optical flow, Canny edges
numpy>=1.24.0             # Numerical computing
mediapipe>=0.10.0         # Face & pose landmark detection
scipy>=1.11.0             # Signal processing for rPPG bandpass filter
scikit-learn>=1.3.0       # t-SNE, metrics, train_test_split
pyyaml>=6.0               # Config file parsing
matplotlib>=3.7.0         # Plotting (ROC curves, confusion matrices)
tqdm>=4.65.0              # Progress bars during extraction
```

---

## 5. Phase 1: Dataset Organization

The project is trained and evaluated on **FaceForensics++** (FF++), a large-scale benchmark for facial manipulation detection.

### 5.1 FaceForensics++ Structure

After downloading FF++, the expected directory layout is:

```
DataSets/
└── FaceForensics++/
    ├── original_sequences/
    │   ├── youtube/
    │   │   └── c40/
    │   │       └── videos/
    │   │           ├── 000.mp4
    │   │           ├── 001.mp4
    │   │           └── ...
    │   └── actors/
    │       └── c40/
    │           └── videos/
    │               └── ...
    └── manipulated_sequences/
        ├── Deepfakes/
        │   └── c40/
        │       └── videos/
        │           └── ...
        ├── Face2Face/
        │   └── c40/
        │       └── ...
        ├── FaceSwap/
        │   └── c40/
        │       └── ...
        ├── FaceShifter/
        │   └── c40/
        │       └── ...
        └── NeuralTextures/
            └── c40/
                └── ...
```

### 5.2 Compression Levels

FF++ provides two compression levels:
- **c23**: High quality (lower compression). Closer to original quality. Easier to detect artifacts.
- **c40**: Low quality (heavy H.264 compression). Closer to real-world video quality. Harder to detect artifacts.

This project targets **c40** by default — the hardest and most realistic setting.

### 5.3 Cross-Dataset: Celeb-DF v2

For generalization testing, Celeb-DF v2 is used. It is organized as:

```
DataSets/
└── Celeb-DF-v2/
    ├── Celeb-real/
    │   └── id0_0000.mp4, ...
    ├── Celeb-synthesis/
    │   └── id0_id1_0000.mp4, ...
    ├── YouTube-real/
    │   └── 00000.mp4, ...
    └── List_of_testing_videos.txt
```

---

## 6. Phase 2: Preprocessing Pipeline

Before any features can be extracted, raw `.mp4` videos must be processed to isolate and normalize the face region. This is critical for physics-based features which are highly sensitive to face alignment.

### 6.1 v1 Preprocessing (Baseline)

The v1 pipeline performs:

1. **Frame Sampling**: Evenly samples `n_frames` (default: 16) frames from the full video duration.
2. **Face Detection**: Uses MediaPipe Face Detection to find the bounding box of the primary face.
3. **Cropping**: Crops and resizes the face region to 256×256 pixels.
4. **c40 Smoothing**: Applies a light Gaussian blur (σ = 0.8) to suppress H.264 macroblock boundary artifacts that would otherwise corrupt FFT and lens distortion measurements.

**Key code** (`data/preprocessing.py`):
```python
def preprocess_video(video_path, n_frames=16, target_size=(256, 256), sigma=0.8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face = detect_and_crop_face(frame)  # MediaPipe
        if face is not None:
            face = cv2.resize(face, target_size)
            face = cv2.GaussianBlur(face, (0, 0), sigma)  # c40 smoothing
            frames.append(face)
    cap.release()
    return frames
```

### 6.2 v2 Preprocessing (Advanced)

The v2 pipeline builds on v1 with three critical enhancements designed to make biological features more reliable:

#### Step 1: Eye Alignment & Face Rotation

After detecting the face, we detect the eye landmarks (left eye center and right eye center) using MediaPipe Face Mesh. The angle between the two eye centers is computed:

```python
angle = np.degrees(np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))
```

The frame is then rotated by `-angle` degrees around the midpoint between the eyes. This ensures all face crops have perfectly horizontal eyes, eliminating rotational variance from the rPPG and lighting extractors.

**Why this matters**: If a person tilts their head, the lighting shading on the face changes. Biological signals like rPPG ROIs (forehead, cheeks) become misaligned. Correcting for roll removes this confounding variable.

#### Step 2: Face Cropping to 256×256

The aligned face is tightly cropped using the face bounding box extended by a 10% margin. This is then resized to exactly 256×256 pixels using bilinear interpolation.

#### Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)

Standard histogram equalization brightens dark areas globally, which can over-amplify noise and destroy the subtle micro-color variations needed by rPPG.

CLAHE solves this by:
1. Dividing the image into a grid of small tiles (8×8 by default).
2. Applying histogram equalization **locally** within each tile.
3. **Clipping** the histogram at a `clipLimit` (default: 2.0) to prevent noise amplification.
4. Using bilinear interpolation to seamlessly blend adjacent tiles.

```python
# Applied to the L channel of LAB color space
lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

**Why LAB color space?** CLAHE is applied to the L (Lightness) channel only, leaving the A (green-red) and B (blue-yellow) chromatic channels untouched. This is critical because rPPG relies on those chromatic channels to detect blood flow.

#### Step 4: Bilateral Filtering

Finally, a bilateral filter (`d=9`, `sigmaColor=75`, `sigmaSpace=75`) is applied. Unlike a Gaussian blur that blurs everything uniformly, a bilateral filter:
- **Smooths flat regions**: Removes camera sensor noise that would corrupt FFT measurements.
- **Preserves sharp edges**: Maintains the crisp boundaries at the nose, eyes, and jawline that the lens distortion extractor needs to detect straight lines.

This is the key innovation over v1 — it reduces noise without losing the structural information required by physics-based extractors.

---

## 7. Phase 3: Feature Extraction in Depth

Once frames are preprocessed, five physics and biology features are extracted from each video. All extractors output a fixed 64-dimensional feature vector (configurable via `feature_dim` in config).

### Feature 1 — Lens Distortion (f1)

**Source file**: `extractors/lens_distortion.py`  
**Physical basis**: Brown-Conrady Lens Distortion Model  
**Sub-system**: Hardware Optics (Sub-System 1)

#### 7.1.1 The Physics

Real camera lenses are not perfect paraxial optics. Light rays passing through the edge of a lens bend slightly more than those passing through the center, producing:

- **Barrel Distortion** (k1 < 0): Straight lines bow outward. Common in wide-angle lenses.
- **Pincushion Distortion** (k1 > 0): Straight lines bow inward. Common in telephoto lenses.
- **Tangential Distortion** (p1, p2): Caused by lens not being perfectly centered over the sensor.

The Brown-Conrady model describes this mathematically as:
```
x_distorted = x (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r²+2*x²)
y_distorted = y (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r²+2*y²) + 2*p2*x*y
```
where `r = sqrt(x² + y²)` is the radial distance from the optical center.

#### 7.1.2 Why Deepfakes Fail

Deepfake generators (GANs, Diffusion models, VAEs) synthesize faces from a latent space. They are trained to match the pixel distribution of real faces but are **never trained to replicate lens distortion**. When a fake face is composited onto real footage, the synthesized face region may have different (or zero) distortion than the background — a detectable geometric inconsistency.

#### 7.1.3 Implementation (Step by Step)

```
Frame (BGR 256×256)
    ↓
Convert to Grayscale
    ↓
Canny Edge Detection (threshold 50→150, aperture 3)
    ↓
Probabilistic Hough Transform → Line Segments
    ↓
For each line segment:
    - Compute mid-point distance r from image center
    - Compute angle between line direction and radial direction
    - Residual = r × sin(angle_diff)   ← Barrel/pincushion indicator
    ↓
Collect all residuals across all frames
    ↓
Build 64-bin Normalized Histogram → f1 (64-D)
```

**Key insight**: In a real camera, straight lines in the physical world (architecture, furniture edges) appear as *curved* lines in the image due to barrel/pincushion distortion. The curvature is spatially consistent: edges near the image corners bend more than edges near the center. A deepfake face, composited without applying the host video's lens model, will exhibit edges whose curvature is *inconsistent* with the surrounding frame.

The histogram representation captures the *distribution* of residual curvatures. Real videos produce a characteristic skewed distribution; deepfakes produce a more irregular or uniform one.

---

### Feature 2 — Motion Blur (f2)

**Source file**: `extractors/motion_blur.py`  
**Physical basis**: Shutter Speed × Object Velocity = Blur Kernel  
**Sub-system**: Hardware Optics (Sub-System 1)

#### 7.2.1 The Physics

A physical camera captures light over a finite exposure time (shutter speed). When an object moves during this exposure, its image is smeared across the sensor, producing **motion blur**. The amount and direction of blur is determined by:

```
blur_kernel = velocity_vector × shutter_speed × focal_length
```

This creates a physically coherent blur: fast-moving objects blur in the direction of motion, and the blur magnitude is proportional to their speed.

#### 7.2.2 Why Deepfakes Fail

Deepfake generators typically render each frame **independently** (as a static image), and then assemble them into a video. The synthesized frames are sharp per-frame, and motion blur is either completely absent or is added as a crude post-hoc Gaussian blur. This post-hoc blur lacks the directional coherence dictated by actual motion trajectories.

#### 7.2.3 Implementation (Farneback Dense Optical Flow)

For each consecutive pair of frames:

```
Frame t (Gray) + Frame t+1 (Gray)
    ↓
Farneback Dense Optical Flow → flow[H, W, 2]
  (Parameters tuned for c40: winsize=21, iterations=5, pyramid_scale=0.5)
    ↓
Per-pixel flow field (u, v) where:
    u = horizontal displacement
    v = vertical displacement
    ↓
Compute 8 Statistics from the flow field:
    1. Magnitude Mean: |flow| average — overall motion speed
    2. Magnitude Std:  |flow| variance — how uniform is the motion
    3. Angle Mean:     Direction of dominant motion
    4. Angle Std:      How coherent is the direction of motion
    5. Divergence Mean: ∂u/∂x + ∂v/∂y — expansion/contraction
    6. Divergence Std
    7. Curl Mean:      ∂v/∂x − ∂u/∂y — rotational motion component
    8. Curl Std
    ↓
Aggregate over all frame pairs:
    mean(8 stats across time) + std(8 stats across time) = 16 values
    ↓
Zero-pad to 64 dimensions → f2 (64-D)
```

**Why Curl and Divergence?** These differential operators capture the *structure* of the motion field:
- **Curl** (∇ × F) measures how much the flow rotates around points — present in genuine head rotation.
- **Divergence** (∇ · F) measures how much the flow expands or contracts — present in genuine forward/backward head motion (zoom).

Deepfakes, lacking real motion dynamics, tend to have low curl and low divergence — their motion is mostly translational and lacks the rotational component of natural head motion.

---

### Feature 5 — Frequency FFT Spectrum (f5)

**Source file**: `extractors/frequency_fft.py`  
**Physical basis**: Camera sensor noise + GAN upsampling artifacts  
**Sub-system**: Hardware Optics (Sub-System 1)

#### 7.3.1 The Physics

Every digital image is a 2D signal that can be decomposed into its frequency components via the Fourier Transform. In the frequency domain:
- **Low frequencies** (center of FFT spectrum): Large-scale color gradients, lighting.
- **High frequencies** (edges of FFT spectrum): Fine textures, sharp edges, noise.

Real cameras produce images with a characteristic frequency distribution: high frequencies roll off smoothly because real-world textures are band-limited by the camera's optical transfer function (OTF).

#### 7.3.2 Why Deepfakes Fail

GAN architectures (and especially VAE/Diffusion decoders) must up-sample from a low-resolution latent representation to a high-resolution output image. Common up-sampling operations (Transposed Convolution, PixelShuffle) create **periodic artifacts** in the frequency domain — the infamous "GAN fingerprint" or "checkerboard artifact". These appear as spikes at specific radial frequencies in the FFT power spectrum that are completely absent in real camera images.

#### 7.3.3 Implementation (Azimuthal Averaging)

```
Single Frame (Grayscale 256×256)
    ↓
2D FFT: f = np.fft.fft2(gray)
    ↓
FFT Shift: fshift = np.fft.fftshift(f)  ← Moves DC component to center
    ↓
Power Spectrum: magnitude = 20 * log(|fshift| + 1e-8)  ← log scale (dB)
    ↓
Azimuthal Average (radial profile):
    For each pixel at distance r from center:
        group all pixels at same radius r
        average their magnitude values
    → 1D radial profile (variable length, ~180 bins for 256×256)
    ↓
Linear Interpolation to 64 bins → f5 (64-D)
```

**Why azimuthal averaging?** It converts the 2D FFT spectrum into a 1D "fingerprint" that describes the energy at each spatial frequency, regardless of direction. This makes it rotation-invariant and compactly describes the spectral "texture" of the image. GAN artifacts appear as bumps in this 1D profile at specific radial frequencies.

This extractor is called for each sampled frame, and the resulting 64-D vectors are averaged across all frames to produce the final feature.

---

### Feature 3 — Biomechanics (f3)

**Source file**: `extractors/biomechanics.py`  
**Physical basis**: Newtonian Mechanics of Human Joints + rPPG  
**Sub-system**: Biological Domain (Sub-System 2)

#### 7.4.1 The Physics

Human motion is governed by Newtonian mechanics. Joint angles follow smooth, physically plausible trajectories constrained by:
- The mass and inertia of body segments.
- Muscle force limitations.
- Neural motor control (which produces smooth, band-limited signals).

Angular velocity and acceleration of joints are therefore smooth signals. Sudden impossible discontinuities in joint angle sequences (implying infinite force) are physically impossible for a real human.

#### 7.4.2 Why Deepfakes Fail

Deepfake face-swapping methods replace the face region while often preserving the body. The synthesized face movement may be temporally inconsistent with the neck and shoulder movement below it, or may exhibit subtle jitter from GAN latent space interpolation that doesn't correspond to physically plausible joint kinematics.

Additionally, this extractor captures the **rPPG (remote Photoplethysmography) signal** as a by-product: the micro-color changes in the face caused by blood flow are reflected in the landmark position oscillations and the mean skin-color values of the ROI regions.

#### 7.4.3 Implementation (MediaPipe Pose + Kinematics)

This extractor supports two MediaPipe backends (auto-detected):
- **MediaPipe Solutions API** (v < 1.0): `mp.solutions.pose`
- **MediaPipe Tasks API** (v >= 1.0): `PoseLandmarker`

```
Frame (BGR 256×256)
    ↓
MediaPipe Pose → 33 3D Body Landmarks [(x, y, z) normalized]
    ↓
Compute Joint Angles for 6 Triplets:
    - Left:  Shoulder(11) - Elbow(13) - Wrist(15)
    - Right: Shoulder(12) - Elbow(14) - Wrist(16)
    - Left:  Hip(23)      - Knee(25)  - Ankle(27)
    - Right: Hip(24)      - Knee(26)  - Ankle(28)
    - Left:  Shoulder(11) - Hip(23)   - Knee(25)
    - Right: Shoulder(12) - Hip(24)   - Knee(26)
    
    angle at B = arccos(dot(BA, BC) / (|BA| × |BC|))
    ↓
Repeat for all N frames → angle_seq (N, 6)
    ↓
Compute Kinematic Derivatives:
    angular_velocity     = diff(angle_seq, axis=0)        → (N-1, 6)
    angular_acceleration = diff(angular_velocity, axis=0) → (N-2, 6)
    ↓
Compute Statistics (mean, std) per joint per temporal derivative:
    - Mean angles × 6 joints
    - Std angles  × 6 joints
    - Mean angular velocity  × 6 joints
    - Std angular velocity   × 6 joints
    - Mean angular acceleration × 6 joints
    - Std angular acceleration  × 6 joints
    → 6 × 6 = 36 values total
    ↓
Pad/Truncate to 64 dimensions → f3 (64-D)
```

**Why these 6 triplets?** They cover the major joint articulations of the upper body and lower body visible from a frontal view. They provide sufficient kinematic information while remaining robust to landmark detection failures.

---

### Feature 4 — Lighting Spherical Harmonics (f4)

**Source file**: `extractors/lighting_sh.py`  
**Physical basis**: Spherical Harmonics Environmental Lighting Model  
**Sub-system**: Biological Domain (Sub-System 2)

#### 7.5.1 The Physics

Spherical Harmonics (SH) are a set of orthogonal basis functions defined on the surface of a sphere. They are used in computer graphics and computer vision to compactly represent low-frequency environment lighting.

For a Lambertian surface (like human skin), the reflected radiance at a point depends on:
1. The surface normal vector at that point.
2. The incident light from all directions in the environment.

The irradiance equation can be written as:
```
E(n) = Σ c_l,m × Y_l,m(n)
```
where `Y_l,m` are the SH basis functions and `c_l,m` are the SH coefficients that describe the lighting environment. Using order-2 SH (l=0,1,2) requires only **9 coefficients** to represent the main lighting characteristics.

#### 7.5.2 Why Deepfakes Fail

A common deepfake technique involves:
1. Generating a synthesized face under some default/neutral lighting.
2. Compositing it onto a target video that may be shot under completely different lighting conditions (e.g., a single strong directional light, colored lighting, etc.).

Even when simple relighting is applied, the low-frequency lighting estimated from the face (via SH coefficients) is often inconsistent with the background scene, revealing the manipulation. Additionally, for genuine real videos, the SH coefficients should be temporally stable (the lighting environment doesn't change from frame to frame). Deepfakes often show temporal instability in SH coefficients due to frame-by-frame generation artifacts.

#### 7.5.3 Implementation (Shape-from-Shading + SH Fitting)

```
Frame (BGR 256×256)
    ↓
Convert to Grayscale and normalize to [0, 1]
    ↓
Estimate Per-Pixel Surface Normals (Shape-from-Shading):
    gx = Sobel(gray, dx=1, dy=0, ksize=5)  ← horizontal gradient
    gy = Sobel(gray, dx=0, dy=1, ksize=5)  ← vertical gradient
    normal = normalize([-gx, -gy, 1])      ← treat gradient as tangent plane
    → normals (H×W, 3)
    ↓
Evaluate 9 SH Basis Functions Y_l,m(normals) for l=0,1,2:
    Y_0^0  = 1
    Y_1^{-1} = ny,   Y_1^0 = nz,   Y_1^1 = nx
    Y_2^{-2} = nx*ny, Y_2^{-1} = ny*nz, Y_2^0 = 3nz²-1
    Y_2^1  = nx*nz,  Y_2^2  = nx²-ny²
    → B (H×W, 9)
    ↓
Fit SH Coefficients via Least Squares:
    I ≈ B × c
    c = (B^T B)^{-1} B^T I    ← numpy.linalg.lstsq
    → c (9,) per frame
    ↓
Repeat for all N frames → sh_seq (N, 9)
    ↓
Compute Temporal Statistics:
    mean_coeffs  = sh_seq.mean(axis=0)     → (9,)
    std_coeffs   = sh_seq.std(axis=0)      → (9,)   ← Temporal instability
    ↓
Pairwise Cosine Similarity between consecutive SH vectors:
    cos_sim[t] = dot(c_t, c_{t+1}) / (|c_t| × |c_{t+1}|)
    → cos_stats = [mean(cos_sims), std(cos_sims)]    → (2,)
    ↓
Temporal Derivative of SH:
    sh_diff = diff(sh_seq, axis=0)
    diff_mean = sh_diff.mean(axis=0)       → (9,)
    diff_std  = sh_diff.std(axis=0)        → (9,)
    ↓
Concatenate: [mean_coeffs | std_coeffs | cos_stats | diff_mean | diff_std]
    = 9 + 9 + 2 + 9 + 9 = 38 values
    ↓
Pad to 64 dimensions → f4 (64-D)
```

---

## 8. Phase 4: Data Loading and the TF.Data Pipeline

Once features are extracted and saved as `.npz` files, `data/dataset_loader.py` handles loading them efficiently into TensorFlow training pipelines.

### 8.1 The Manifest File

After extraction, each feature directory contains a `manifest.csv` file:

```
npz_path,label
original_sequences/youtube/c40/videos/000.npz,0
manipulated_sequences/Deepfakes/c40/videos/000.npz,1
...
```

The label `0` = Real, `1` = Fake. The manifest is used by `load_manifest()` to build the list of all samples.

### 8.2 Feature Normalization

Raw feature vectors have very different scales:
- FFT values may be in the range [-50, 200] (dB scale).
- SH coefficients may be in the range [-0.1, 0.1].
- Biomechanical angles are in radians [0, π].

Without normalization, the model's gradient updates would be dominated by whichever feature has the largest absolute values.

The `normalize_and_clip_features()` function loads pre-computed statistics from `config/feature_scaling_stats.json`:
```python
x_normalized = clip((x - mean) / std, -5.0, +5.0)
```

The clipping to `[-5, +5]` removes extreme outliers that would otherwise destabilize training (e.g., corrupted video frames producing NaN or extremely large feature values).

### 8.3 The build_tf_dataset() Function

```python
def build_tf_dataset(entries, batch_size, shuffle, feature_dim, features_dir):
```

This function creates a `tf.data.Dataset` from a list of `(npz_path, label)` tuples using a **Python generator**:

```python
def generator():
    for path, label in entries:
        data = np.load(path)
        yield ({
            'f1_lens_distortion': data['f1'],
            'f2_motion_blur':     data['f2'],
            'f3_biomechanics':    data['f3'],
            'f4_lighting_sh':     data['f4'],
            'f5_frequency_fft':   data['f5'],
        }, label)

dataset = tf.data.Dataset.from_generator(generator, output_signature=...)
```

The generator approach is memory-efficient: it loads one `.npz` file at a time rather than loading the entire dataset into RAM.

The resulting dataset is then:
1. **Shuffled** (for training sets): `dataset.shuffle(buffer_size=1024)`
2. **Batched**: `dataset.batch(batch_size)`
3. **Prefetched**: `dataset.prefetch(tf.data.AUTOTUNE)` — loads the next batch while the model is training on the current one.

---

## 9. Phase 5: The Dataset Split Strategy

### 9.1 The get_strict_datasets() Function

This function implements a carefully designed data split that balances multiple competing requirements:

#### Requirements:
1. The Train/Val pool should use only **specific** manipulation methods (Deepfakes, Face2Face) to ensure controlled training.
2. The Test set should use **all remaining** videos to maximize coverage and avoid data leakage.
3. The 80/20 Train/Val split should be shuffled to prevent the model learning any temporal ordering.

#### Implementation:

```
All entries (from manifest.csv)
    ↓
Separate by category:
    real_entries     ← all original_sequences videos
    fake_by_method   ← Deepfakes, Face2Face, FaceSwap, NeuralTextures
    other_fakes      ← any other manipulation methods
    ↓
Random shuffle each category separately (seed=42)
    ↓
Select Training + Validation Quotas:
    trainval_real = real_entries[:1000]          (first 1000 real)
    trainval_df   = Deepfakes[:750]              (first 750 Deepfake)
    trainval_f2f  = Face2Face[:750]              (first 750 Face2Face)
    ↓
trainval_entries = trainval_real + trainval_df + trainval_f2f  (2500 total)
    ↓
shuffle(trainval_entries)  ← Intermix all categories
    ↓
split_idx = int(2500 × 0.8) = 2000
    train_entries = trainval_entries[:2000]
    val_entries   = trainval_entries[2000:]
    ↓
train_real = [e for e in train_entries if label==0]  ← ~800 real
train_fake = [e for e in train_entries if label==1]  ← ~1200 fake
    ↓
Test Set (Zero Leakage):
    used_paths = {e[0] for e in trainval_entries}
    test_entries = [e for e in ALL_ENTRIES if e[0] not in used_paths]
```

#### Summary of resulting split:

| Set | Approximate Size | Contents |
|---|---|---|
| Train | ~2,000 | 80% of pool (1000R + 750DF + 750F2F) |
| Validation | ~500 | 20% of pool (same categories) |
| Test | All remaining | Every video NOT in Train/Val |

### 9.2 Class Balancing During Training

The training set produced by the strict split has an imbalance: ~800 real vs ~1200 fake. Training on this imbalanced set would cause the model to be biased toward predicting "Fake".

The solution is to use TensorFlow's `sample_from_datasets`:
```python
train_real_ds = build_tf_dataset(train_real, ...)   # ~800 items
train_fake_ds = build_tf_dataset(train_fake, ...)   # ~1200 items

train_ds = tf.data.Dataset.sample_from_datasets(
    [train_real_ds, train_fake_ds],
    weights=[0.5, 0.5],  # ← 50% real, 50% fake at every step
    stop_on_empty_dataset=True
)
```

This ensures that every training batch contains exactly 50% real and 50% fake samples, regardless of the actual class distribution in the training pool.

---

## 10. Phase 6: Neural Network Architecture (Deep Dive)

### 10.1 Architecture Overview

```
Input Features (5 × 64-D)
         │
    ┌────┴────┐
    │         │
Sub-Sys 1    Sub-Sys 2
(Optics)    (Biology)
[f1,f2,f5] → Z1[256] [f3,f4] → Z2[256]
    │                    │
    └────────┬───────────┘
             │
    MidLevelFusionBlock12
    (Bidirectional Cross-Attention)
             │
    Z_fused [256]
             │
    Classification Head
    Dense(256→ReLU→Dropout)
    Dense(128→ReLU→Dropout)
    Dense(1→Sigmoid)
             │
    P(Fake) ∈ [0, 1]
```

### SubSystem1Encoder — Hardware Optics Encoder

**Defined in**: `models/master_fusion.py` → `class SubSystem1Encoder`

```python
class SubSystem1Encoder(layers.Layer):
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        self.projection = layers.Dense(embed_dim, activation='relu')
        self.layer_norm = layers.LayerNormalization()
        self.classifier = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation=None)
        ])

    def call(self, inputs, training=False):
        f1, f2, f5 = inputs
        concat_features = ops.concatenate([f1, f2, f5], axis=-1)  # [B, 192]
        z1 = self.layer_norm(self.projection(concat_features))     # [B, 256]
        sub_logits = self.classifier(z1)                           # [B, 1]
        return z1, sub_logits
```

**Detailed breakdown**:

| Operation | Input Shape | Output Shape | Purpose |
|---|---|---|---|
| Concatenate [f1, f2, f5] | 3 × [B,64] | [B, 192] | Combine hardware features |
| Dense(256, relu) | [B, 192] | [B, 256] | Project to embedding space |
| LayerNormalization | [B, 256] | [B, 256] | Stabilize embedding distribution |
| Dense(256, relu) (classifier) | [B, 256] | [B, 256] | First classifier layer |
| Dense(128, relu) | [B, 256] | [B, 128] | Second classifier layer |
| Dense(1, linear) | [B, 128] | [B, 1] | Auxiliary logit output |

**Why LayerNormalization?** The three hardware features (f1, f2, f5) have very different statistical properties (a lens distortion histogram vs. optical flow statistics vs. FFT spectrum). After the projection layer, LayerNorm normalizes each embedding vector to have zero mean and unit variance across the 256 dimensions. This prevents any single feature type from dominating the embedding space.

---

### SubSystem2Encoder — Biological Domain Encoder

**Defined in**: `models/master_fusion.py` → `class SubSystem2Encoder`

```python
class SubSystem2Encoder(layers.Layer):
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        self.fc_f3 = layers.Dense(embed_dim)           # Embed biomechanics
        self.fc_f4 = layers.Dense(embed_dim)           # Embed lighting SH
        self.gate_dense = layers.Dense(embed_dim, activation='sigmoid')
        self.classifier = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)
        ])

    def call(self, inputs, training=False):
        f3, f4 = inputs
        e3 = self.fc_f3(f3)                            # [B, 256]
        e4 = self.fc_f4(f4)                            # [B, 256]
        concat_feats = ops.concatenate([e3, e4], axis=-1)  # [B, 512]
        g = self.gate_dense(concat_feats)              # [B, 256] ∈ (0,1)
        z2 = g * e3 + (1.0 - g) * e4                  # [B, 256] Gated mix
        sub_logits = self.classifier(z2)               # [B, 1]
        return z2, sub_logits
```

**The Gating Mechanism** — This is the key architectural difference from Sub-System 1. Instead of simple concatenation, the biological features are merged via a **learned sigmoid gate**:

```
z2 = σ(W[e3, e4] + b) ⊙ e3  +  (1 - σ(W[e3, e4] + b)) ⊙ e4
```

Think of `g` as a trust score between 0 and 1 for each dimension:
- When `g[i] → 1`: Dimension `i` of `z2` is determined by the biomechanics feature.
- When `g[i] → 0`: Dimension `i` of `z2` is determined by the lighting SH feature.

**Why is this necessary?** Biological features are context-dependent:
- In poor lighting conditions, the rPPG signal (f3) becomes unreliable (low SNR), but the lighting SH (f4) may still be informative.
- For videos where the face is not visible for some frames, biomechanics (f3) will produce zero vectors, but lighting (f4) may still detect illumination inconsistencies.

The gate allows the network to learn which biological domain to trust more *per sample*, adapting dynamically.

---

### MidLevelFusionBlock12 — Bidirectional Cross-Attention Fusion

**Defined in**: `models/fusion_1_2_mid.py` → `class MidLevelFusionBlock12`

This is the most innovative architectural component. It implements a **Bidirectional Cross-Attention** mechanism between the hardware ($Z_1$) and biological ($Z_2$) embeddings.

#### Why Cross-Attention instead of Concatenation?

Simple concatenation `[Z1 | Z2]` treats the two embeddings independently. It does not allow the hardware evidence to *inform* the biological evidence, or vice versa.

Consider the following reasoning:
- If the FFT spectrum (part of $Z_1$) shows strong checkerboard artifacts (high fake probability), should the model trust the rPPG signal (part of $Z_2$) or consider it secondary?
- If the lighting SH (part of $Z_2$) shows extreme inconsistency, does this match the lens distortion (part of $Z_1$) suggesting a face composited from a different video?

Cross-attention allows this inter-domain reasoning by letting each embedding **query** the other for relevant context.

#### Implementation Detail:

```python
def call(self, inputs, training=False):
    z1, z2 = inputs  # Both [B, 256]

    # Expand to sequence dimension required by MultiHeadAttention
    z1_seq = ops.expand_dims(z1, axis=1)  # [B, 1, 256]
    z2_seq = ops.expand_dims(z2, axis=1)  # [B, 1, 256]

    # z1 (Hardware) queries z2 (Biology)
    # "What biological context is relevant to my hardware observation?"
    z1_ctx = cross_attn_1_to_2(query=z1_seq, key=z2_seq, value=z2_seq)
    z1_ctx = ops.squeeze(z1_ctx, axis=1)         # [B, 256]
    z1_enriched = LayerNorm(z1 + z1_ctx)          # Residual + LN  [B, 256]

    # z2 (Biology) queries z1 (Hardware)
    # "What hardware context is relevant to my biological observation?"
    z2_ctx = cross_attn_2_to_1(query=z2_seq, key=z1_seq, value=z1_seq)
    z2_ctx = ops.squeeze(z2_ctx, axis=1)         # [B, 256]
    z2_enriched = LayerNorm(z2 + z2_ctx)          # Residual + LN  [B, 256]

    # Hadamard Product: captures multiplicative interaction
    interaction = z1_enriched * z2_enriched        # [B, 256]

    # Feature Pyramid: [z1' | z2' | z1'⊙z2'] = [B, 768]
    pyramid = concatenate([z1_enriched, z2_enriched, interaction])

    # MLP bottleneck: 768 → 512 → 256
    fused = MLP_GELU(pyramid)                      # [B, 256]
    fused = LayerNorm(fused)

    return fused, z1_enriched, z2_enriched
```

#### The Multi-Head Attention Breakdown

The cross-attention uses `num_heads=8` and `key_dim=32` (256/8). For each of the 8 attention heads:

```
Q = Z1_seq @ W_Q   [B, 1, 32]
K = Z2_seq @ W_K   [B, 1, 32]
V = Z2_seq @ W_V   [B, 1, 32]

attention_weights = softmax(Q @ K^T / sqrt(32))  [B, 1, 1]
output = attention_weights @ V                    [B, 1, 32]
```

8 heads produce [B, 1, 8×32] = [B, 1, 256], then squeezed to [B, 256].

#### The Hadamard Product

After enriching both embeddings via cross-attention, their **element-wise product** is computed:

```
interaction[i] = z1_enriched[i] × z2_enriched[i]
```

This is a bilinear interaction: if dimension `i` of both $Z_1'$ and $Z_2'$ are both large (both systems agree this is a strong fake signal in dimension `i`), the interaction value is very large. If one is near zero (one system is uncertain about this dimension), the product is near zero — it only fires when both systems agree.

#### The Feature Pyramid

```
pyramid = [z1_enriched | z2_enriched | z1_enriched ⊙ z2_enriched]
        = [B, 256     | B, 256      | B, 256                    ]
        = [B, 768]
```

This "pyramid" approach (inspired by FPN in object detection) gives the classification head access to:
1. The hardware signal alone ($Z_1'$).
2. The biological signal alone ($Z_2'$).
3. Their multiplicative interaction ($Z_1' ⊙ Z_2'$).

The final MLP projects this 768-D pyramid to the standard 256-D embedding using GELU activation.

---

### Classification Head

After fusion, the 256-D representation is passed through a final classifier:

```
z_fused [B, 256]
    ↓
Dense(256, activation='relu')  ← Expand representation
    ↓
Dropout(0.4)                    ← Regularization (40% dropout)
    ↓
Dense(128, activation='relu')  ← Compress
    ↓
Dropout(0.3)                    ← Regularization (30% dropout)
    ↓
Dense(1, activation=None)      ← Raw logit (no sigmoid yet)
    ↓
fusion_logits [B, 1]
```

**Why no sigmoid on the output?** The output is a raw logit (unbounded). The sigmoid is applied inside the loss function (`BinaryCrossentropy(from_logits=True)`). This provides better numerical stability during training because TensorFlow can combine the log-sigmoid computation more efficiently than computing sigmoid first, then log.

---

### MasterFusionBlock — Alternative Architecture

**Defined in**: `models/master_fusion.py` → `class MasterFusionBlock`

The `MasterFusionBlock` is an alternative, simpler fusion mechanism compared to `MidLevelFusionBlock12`.

Instead of bidirectional cross-attention, it uses **Self-Attention over a stacked sequence** plus a **sigmoid gate**:

```python
def call(self, inputs, training=False):
    z1, z2 = inputs                              # Each [B, 256]
    
    z_stack = ops.stack([z1, z2], axis=1)        # [B, 2, 256] — "two tokens"
    
    # Self-Attention over the two-token sequence
    attn_out = master_cross_attn(
        query=z_stack, key=z_stack, value=z_stack
    )  # [B, 2, 256]
    
    attn_pooled = ops.mean(attn_out, axis=1)     # [B, 256] — pool the two tokens
    z_avg = ops.mean(z_stack, axis=1)            # [B, 256] — simple average
    
    # Gated combination: gate decides blend of simple avg vs. attended
    g = Dense(embed_dim, activation='sigmoid')([z1, z2])  # [B, 256]
    
    master_rep = LayerNorm(g * z_avg + (1 - g) * attn_pooled)  # [B, 256]
    
    return master_rep
```

**When to use MasterFusionBlock vs MidLevelFusionBlock12?**

| Aspect | MidLevelFusionBlock12 | MasterFusionBlock |
|---|---|---|
| Fusion Type | Bidirectional Cross-Attention | Self-Attention + Gating |
| Parameters | ~1.1M (more expressive) | ~450K (lighter) |
| Inter-domain reasoning | Explicit (Z1 queries Z2 and vice versa) | Implicit (through shared self-attention) |
| Use case | Primary model, best accuracy | Faster training, lower compute |

---

## 11. Phase 7: The Two-Stage Training Strategy

The two-stage curriculum training is the most critical design choice in the training pipeline.

### Stage 1: Fusion Warm-Up (Encoders Frozen)

**Script**: `train_fusion_1_2_mid.py`  
**Duration**: 10 epochs  
**Learning Rate**: 0.001

In Stage 1:
1. The pre-trained Sub-System 1 and Sub-System 2 encoder weights are loaded from `checkpoints/subsystem1_best.keras` and `checkpoints/subsystem2_best.keras`.
2. All layers in both encoders are **frozen**: `layer.trainable = False`.
3. Only the `MidLevelFusionBlock12` and the final Classification Head have their parameters updated.

**Why freeze the encoders in Stage 1?**

Consider the training dynamic without freezing:
- At the start, the fusion block has random weights. It doesn't know how to combine $Z_1$ and $Z_2$ yet.
- The random cross-attention weights will produce random gradients, which flow back into the encoders and corrupt the carefully learned encoder representations.
- The encoders forget what they learned during independent pre-training.

By freezing the encoders, Stage 1 teaches the fusion block to **work with fixed, high-quality embeddings**. The fusion mechanism can learn a good combination strategy without destroying the encoder's specialized knowledge.

### Stage 2: End-to-End Fine-Tuning

**Duration**: 20 epochs  
**Learning Rate**: 0.0001 (10× lower than Stage 1)

In Stage 2:
1. All layers are unfrozen: `layer.trainable = True`.
2. The entire model is trained jointly with a very small learning rate.

**Why unfreeze in Stage 2?**

After Stage 1, the fusion block has learned a good strategy for combining the embeddings. Now, the encoder representations can be *gently refined* to produce embeddings that are not just good for their individual tasks ($Z_1$ for hardware detection, $Z_2$ for biological detection) but are also optimally shaped for the cross-attention fusion operation.

The low learning rate ensures this fine-tuning is gentle and does not catastrophically forget the pre-learned encoder specializations.

**This is analogous to transfer learning**: pre-trained CNN features are first frozen, then fine-tuned with a very low learning rate.

---

## 12. Phase 8: Loss Functions and Auxiliary Supervision

### 12.1 Primary Loss: Binary Focal Loss

The main classification task uses **Binary Focal Cross-Entropy** (gamma=2.0):

```
Focal Loss = -αt × (1 - pt)^γ × log(pt)
```

where:
- `pt = σ(logit)` if y=1 (fake), else `1 - σ(logit)` (real)
- `γ = 2.0` — the focusing parameter
- `αt` — optional class weight (default 1.0)

**Why Focal Loss instead of standard BCE?**

Standard BCE treats all samples equally. But deepfake detection has "easy" and "hard" samples:
- **Easy real sample**: An authentic uncompressed video → pt → 1 → small loss → small gradient update.
- **Hard real sample**: A highly compressed or poorly lit video → model is confused → pt ≈ 0.5 → large loss.
- **Easy fake sample**: An obvious GAN artifact → pt → 0 → small loss.
- **Hard fake sample**: A state-of-the-art deepfake → model is confused → large loss.

The term `(1 - pt)^γ` reduces the weight of easy samples (when pt is already close to 1 or 0, this term is small) and focuses training effort on the hard, misclassified samples. With γ=2, a sample predicted with 99% confidence contributes only `(0.01)^2 = 0.0001` of its standard BCE loss.

### 12.2 Auxiliary Losses: Standard BCE

The Sub-System 1 and Sub-System 2 outputs also have their own auxiliary losses:

```python
losses = {
    'fusion_logits': focal_loss,  # Weight: 1.0
    'sub1_logits':   bce_loss,    # Weight: 0.2
    'sub2_logits':   bce_loss,    # Weight: 0.2
}
```

**Total Loss**: `L = 1.0 × L_focal + 0.2 × L_bce_sub1 + 0.2 × L_bce_sub2`

**Purpose of Auxiliary Losses:**

Without auxiliary losses, the encoder representations could undergo **representation collapse**: the encoder outputs ($Z_1$, $Z_2$) could converge to meaningless embeddings that happen to produce good fusion results. The auxiliary losses force $Z_1$ to always remain a useful representation for hardware-based classification, and $Z_2$ to always remain useful for biological-based classification. This prevents degenerate solutions and improves generalization.

---

## 13. Phase 9: Training Callbacks

Three Keras callbacks are used for regularization and efficiency:

### 13.1 ModelCheckpoint

```python
tf.keras.callbacks.ModelCheckpoint(
    'checkpoints/fusion12_mid_final_best.keras',
    monitor='val_fusion_logits_acc',
    save_best_only=True,
    mode='max',
    verbose=1,
)
```

Saves the model whenever validation accuracy improves. The `save_best_only=True` ensures only the best weights are kept, protecting against overfitting (the model might improve on training loss after the best validation accuracy, indicating overfitting).

### 13.2 EarlyStopping

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_fusion_logits_acc',
    patience=7,          # Stop if no improvement for 7 consecutive epochs
    mode='max',
    restore_best_weights=True,
    verbose=1,
)
```

EarlyStopping terminates training when the model stops improving, preventing wasted computation and overfitting. `restore_best_weights=True` ensures the model reverts to its best checkpoint, not its last weights (which may have overfit).

### 13.3 ReduceLROnPlateau

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # New LR = Old LR × 0.5
    patience=4,          # Trigger after 4 epochs without improvement
    min_lr=1e-7,         # Never go below this LR
    verbose=1,
)
```

When training stalls (validation loss stops decreasing), reducing the learning rate by half allows the optimizer to take smaller steps and potentially escape local minima or saddle points. `min_lr` prevents the learning rate from becoming so small that training effectively stops.

---

## 14. Phase 10: Evaluation Metrics

After training, the model is evaluated on the held-out test set using five metrics:

### 14.1 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

The fraction of all samples correctly classified. Simple but misleading for imbalanced test sets.

### 14.2 AUC-ROC (Area Under ROC Curve)

```
AUC = ∫ TPR d(FPR)
```

The AUC measures the model's ability to **rank** fake samples higher than real ones, regardless of the decision threshold. AUC = 1.0 means perfect ranking; AUC = 0.5 means random guessing.

**Why AUC is more important than accuracy**: AUC is threshold-independent. A model that outputs very high confidence scores for all fakes and very low for all reals has AUC=1.0 even if the threshold is poorly calibrated, leading to low accuracy but excellent ranking ability.

### 14.3 F1-Score

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 balances precision (how many detected fakes are actually fake) and recall (how many actual fakes were detected). For security-critical applications where missing a deepfake (FN) is costly, we may prefer high recall even at the expense of some precision.

### 14.4 Confusion Matrix

A 2×2 matrix showing:
```
                Predicted Real    Predicted Fake
Actual Real        TN                FP
Actual Fake        FN                TP
```

Saved as a matplotlib figure to `results/cm_fusion12_mid.png`.

### 14.5 ROC Curve

The full TPR vs. FPR tradeoff curve across all thresholds, saved to `results/roc_fusion12_mid.png`. The operating point (threshold) can be chosen based on the desired FPR/TPR tradeoff after training.

---

## 15. Phase 11: Cross-Dataset Evaluation (Celeb-DF)

The most important test of generalization is evaluating on a completely different dataset: **Celeb-DF v2**.

### 15.1 Why Cross-Dataset Evaluation?

A model with 99% accuracy on FF++ test set but 55% on Celeb-DF is not useful in the real world. Cross-dataset evaluation measures whether the model has learned *general deepfake artifacts* or has simply memorized dataset-specific quirks.

Physics-based features should generalize well because:
- Camera lens distortion is a physical property that all videos share.
- rPPG signals are a property of human biology, not the specific GAN used.
- FFT artifacts are algorithmic artifacts of upsampling layers, present in all GAN-based methods.

### 15.2 Execution

```bash
python eval_celebdf.py --data_dir DataSets/Celeb-DF-v2 --max_per_class 100
```

This script:
1. Loads the pre-trained fusion model from `checkpoints/fusion12_mid_final_best.keras`.
2. Discovers real and fake videos from Celeb-DF.
3. **On-the-fly feature extraction**: Instead of using pre-extracted `.npz` files, it processes raw videos through the full preprocessing and extraction pipeline in real time.
4. Runs inference and computes all metrics.
5. Saves confusion matrix and ROC curve.

---

## 16. Phase 12: The Automation Runner

`run_all_experiments.py` orchestrates the full benchmark in sequence:

```
run_all_experiments.py
    │
    ├─► Experiment Set 1: Features v1
    │       ├─► train_fusion_1_2_mid.py (on DataSets/features)
    │       ├─► eval_celebdf.py         (cross-dataset evaluation)
    │       ├─► train_master.py         (on DataSets/features)
    │       └─► main_eval.py            (FF++ test set evaluation)
    │
    └─► Experiment Set 2: Features v2
            ├─► train_fusion_1_2_mid.py (on DataSets/features_v2)
            ├─► eval_celebdf.py
            ├─► train_master.py
            └─► main_eval.py
```

**Output Filtering**: The runner uses regex to strip TF warnings, ANSI escape codes, and step-by-step progress bars from the log file. Only meaningful information is written to `all_experiments_results.txt`:
- Epoch numbers
- Final epoch validation metrics
- Data split sizes
- Final test metrics (Accuracy, AUC, F1-Score, Classification Report)

---

## 17. Feature Set v1 vs v2 Comparison

| Aspect | Features v1 | Features v2 |
|---|---|---|
| Preprocessing | Grayscale crop + Gaussian blur | Eye alignment + CLAHE + Bilateral filter |
| Face orientation | May vary with head tilt | Always horizontally aligned |
| Contrast | Standard | CLAHE-enhanced (better for rPPG) |
| Edge preservation | Gaussian blur smooths edges | Bilateral filter preserves edges |
| FFT quality | May have Gaussian blur artifacts | Sharper, more accurate frequency content |
| rPPG quality | Standard | Higher SNR (better CLAHE contrast) |
| SH quality | Standard | More accurate normals (better contrast) |
| Expected improvement | Baseline | ~2-5% better AUC expected |

---

## 18. Config Reference

| Config Key | Type | Default | Description |
|---|---|---|---|
| `dataset.root` | str | DataSets | Root directory of the dataset |
| `dataset.compression` | str | c40 | Compression level: c23 or c40 |
| `dataset.split_seed` | int | 42 | Random seed for reproducible splits |
| `dataset.features_dir` | str | DataSets/features_v2 | Path to pre-extracted features |
| `preprocessing.n_frames` | int | 16 | Frames to sample per video |
| `preprocessing.target_size` | list | [256, 256] | Face crop resolution |
| `preprocessing.gaussian_blur_sigma` | float | 0.8 | c40 smoothing sigma |
| `extractors.feature_dim` | int | 64 | Output dimension of each extractor |
| `model.embed_dim` | int | 256 | Internal embedding dimension |
| `model.master_aux_weight` | float | 0.2 | Weight of auxiliary losses |
| `training.batch_size` | int | 32 | Training batch size |
| `training.learning_rate_stage1` | float | 0.001 | LR for Stage 1 (frozen encoders) |
| `training.learning_rate_stage2` | float | 0.0001 | LR for Stage 2 (full fine-tuning) |
| `training.epochs_stage1` | int | 10 | Number of Stage 1 epochs |
| `training.epochs_stage2` | int | 20 | Number of Stage 2 epochs |
| `training.subsystem_epochs` | int | 20 | Epochs for sub-system pre-training |
| `training.subsystem_lr` | float | 0.0001 | LR for sub-system pre-training |

---

## 19. Common Errors and Fixes

### Error 1: `FileNotFoundError: Manifest not found`

**Cause**: The feature extraction pipeline has not been run yet.

**Fix**:
```bash
python pipelines/extract_features.py   # or extract_features_v2.py
```

### Error 2: `KeyError: 'f5'` when loading .npz files

**Cause**: The `.npz` files were extracted with an older version of the extraction script that did not include the `f5` (FFT) feature.

**Fix**: Re-run the full extraction pipeline:
```bash
python pipelines/extract_features_v2.py
```

### Error 3: GPU warnings during training

**Cause**: TensorFlow >= 2.11 does not support native GPU on Windows.

**Fix**: 
- Either ignore the warning and use CPU (slow but functional).
- Or install WSL2 and run the project from there with CUDA support.
- Or install the `tensorflow-directml-plugin` for DirectML GPU acceleration on Windows.

### Error 4: `UnicodeDecodeError` in run_all_experiments.py

**Cause**: Windows terminal uses CP1252 encoding, not UTF-8. The progress bar characters (special Unicode blocks) cannot be decoded.

**Fix**: The `clean_ansi()` function in the updated `run_all_experiments.py` strips these characters. Also ensure the subprocess call does not pipe these through a CP1252 decoder:
```python
process = subprocess.Popen(
    command, shell=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',    # ← Explicit UTF-8 encoding
    errors='replace'     # ← Replace un-decodable characters
)
```

### Error 5: Low Validation Accuracy (< 55%)

**Cause**: The model is not learning. Possible causes:
1. Feature files are corrupted (all zeros). Run `python pipelines/plot_features.py` to diagnose.
2. Class imbalance not properly handled — check that 50/50 sampling is active.
3. Learning rate too high — gradient explosion. Check for NaN in loss.
4. Pre-trained sub-system weights not found — fusion training starting from scratch.

**Fix**: Check the training log for NaN losses, verify the checkpoint files exist, and run the feature diagnostics.

### Error 6: MediaPipe model not found (biomechanics.py)

**Cause**: The pose landmarker model file hasn't been downloaded.

**Fix**: The `_ensure_pose_model()` function will auto-download it on first run if internet access is available. If behind a firewall, manually download from:
```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```
and place it in `models/mediapipe/`.

---

## 20. Mathematical Appendix

### 20.1 Fourier Transform and Power Spectrum

For an image `I` of size `H × W`, the 2D DFT is:

```
F(u, v) = Σ_x Σ_y I(x, y) × exp(-j2π(ux/W + vy/H))
```

The power spectrum is:
```
P(u, v) = |F(u, v)|²
```

The azimuthal average at radial frequency `r` is:
```
R(r) = (1/N_r) × Σ_{(u,v): sqrt(u²+v²)≈r} P(u, v)
```

where `N_r` is the count of pixels at radius `r`.

### 20.2 Spherical Harmonics Basis Functions (Order 2)

The 9 real spherical harmonics up to order l=2 evaluated at unit normal (nx, ny, nz):

```
l=0: Y_0^0  = 1

l=1: Y_1^{-1} = ny
     Y_1^0   = nz
     Y_1^1   = nx

l=2: Y_2^{-2} = nx × ny
     Y_2^{-1} = ny × nz
     Y_2^0    = 3×nz² - 1
     Y_2^1    = nx × nz
     Y_2^2    = nx² - ny²
```

The irradiance at a surface point with normal `n` is approximated as:
```
E(n) ≈ Σ_{l,m} c_{l,m} × Y_l^m(n)
```

Fitting coefficients `c` via least squares over all face pixels gives a 9-D representation of the dominant lighting environment.

### 20.3 Focal Loss Derivation

Binary Cross-Entropy:
```
BCE(p, y) = -[y × log(p) + (1-y) × log(1-p)]
```

Let `pt = p` if y=1, else `(1-p)`:
```
BCE(pt) = -log(pt)
```

Focal Loss adds a modulating factor:
```
FL(pt) = -(1 - pt)^γ × log(pt)
```

When `pt → 1` (easy sample, model is confident): `(1-pt)^γ → 0` — loss is near zero.  
When `pt → 0` (hard sample, model is wrong): `(1-pt)^γ → 1` — loss is standard BCE.

This dynamically down-weights easy examples and focuses training on hard ones.

### 20.4 Multi-Head Attention

For a single attention head with queries Q, keys K, values V:
```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

where `d_k` is the key dimension (32 for our 8-head configuration).

Multi-head attention concatenates h heads:
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O

head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
```

For our bidirectional cross-attention:
- **Z1 attending Z2**: Q = Z1, K = Z2, V = Z2
- **Z2 attending Z1**: Q = Z2, K = Z1, V = Z1

---

## 21. Future Improvements Roadmap

### 21.1 Temporal Sequence Modeling

**Current limitation**: Features are extracted frame-by-frame and aggregated with simple statistics (mean, std). This discards temporal ordering information.

**Proposed solution**: Replace the statistical aggregation in each extractor with a **1D Temporal Convolutional Network** or a **Bidirectional LSTM** that processes the frame-by-frame feature sequence as a time series. This would capture:
- The periodic oscillation of rPPG (which has a specific frequency range of 0.75-2.5 Hz).
- The smooth acceleration/deceleration of real joint motion.
- The temporal coherence of SH lighting across frames.

### 21.2 Audio-Visual Synchronization (Sub-System 3)

**Rationale**: Deepfakes frequently have micro-desyncs between lip movement and audio phonemes. Adding a third sub-system that cross-correlates:
- **MFCC features** from the audio track (Mel-Frequency Cepstral Coefficients).
- **Lip landmark trajectories** from MediaPipe Face Mesh.

This would be a powerful additional biological cue and can be integrated directly into the bidirectional cross-attention as a third embedding $Z_3$.

### 21.3 Dynamic Feature Weighting

**Current limitation**: The Bidirectional Cross-Attention uses fixed 8-head configuration.

**Proposed solution**: Add a **Mixture-of-Experts (MoE)** gating network that learns when to rely more on hardware features vs. biological features based on video quality indicators:
- Video quality metrics (BRISQUE score, compression level).
- Face visibility and landmark confidence scores.

This would make the model robust to varying video qualities and lighting conditions.

### 21.4 Self-Supervised Pre-Training

**Current limitation**: The sub-systems are pre-trained in a fully supervised manner, requiring labeled data.

**Proposed solution**: Use contrastive learning (SimCLR, BYOL) to pre-train the encoders on unlabeled video data:
- Positive pairs: Two different frame windows from the same video (should have similar physics signatures).
- Negative pairs: Frame windows from different videos (should have different signatures).

This would allow leveraging vast amounts of unlabeled video data to learn more generalizable physics representations before the supervised fine-tuning step.

### 21.5 Adversarial Training for Robustness

**Rationale**: An adversary could potentially create deepfakes specifically designed to mimic authentic physics signatures.

**Proposed solution**: Train with adversarial examples generated by perturbing the physical features to minimize detection:
- Gradient-based feature perturbation (similar to PGD attack).
- Train the detector to be robust against these worst-case perturbations.

---


---

## 22. Complete Step-by-Step Workflow (Quick Reference)

This section provides the exact command sequence to run the entire project from scratch on a fresh machine, with detailed explanations at each step.

### Step 0 — Clone and Install

```bash
git clone https://github.com/Youssef-Ebn-Yasser/deepfake_physics_detector.git
cd deepfake_physics_detector
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**What happens**: Python installs TensorFlow (~450 MB), OpenCV (~50 MB), MediaPipe (~200 MB), and all other dependencies.

**Estimated time**: 5-15 minutes depending on internet speed.

---

### Step 1 — Organize Dataset

Download FaceForensics++ from the official source. The download script provides access to different manipulations and compression levels. Ensure the structure matches the expected layout under `DataSets/`.

```
DataSets/
├── FaceForensics++/
│   ├── original_sequences/youtube/c40/videos/*.mp4
│   ├── manipulated_sequences/Deepfakes/c40/videos/*.mp4
│   ├── manipulated_sequences/Face2Face/c40/videos/*.mp4
│   ├── manipulated_sequences/FaceSwap/c40/videos/*.mp4
│   └── manipulated_sequences/NeuralTextures/c40/videos/*.mp4
└── Celeb-DF-v2/
    ├── Celeb-real/*.mp4
    ├── Celeb-synthesis/*.mp4
    └── YouTube-real/*.mp4
```

---

### Step 2 — Baseline Feature Extraction (v1)

```bash
python pipelines/extract_features.py
```

**What happens**:
- Scans all source directories for `.mp4` files.
- For each video: samples 16 frames, detects and crops faces, applies c40 Gaussian smoothing.
- Runs all 5 extractors: f1 (lens distortion), f2 (motion blur), f3 (biomechanics), f4 (lighting SH), f5 (FFT).
- Saves one `.npz` file per video under `DataSets/features/`.
- Writes `DataSets/features/manifest.csv` with all file paths and labels.

**Estimated time**: 6-24 hours depending on dataset size and CPU speed.

**Expected output**:
```
Processing: DataSets/FaceForensics++/original_sequences/youtube/c40/videos/
  [001/1000] 000.mp4 → features/original_sequences_youtube_000.npz  ✓
  [002/1000] 001.mp4 → features/original_sequences_youtube_001.npz  ✓
  ...
Manifest written: DataSets/features/manifest.csv  (6363 entries)
```

---

### Step 3 — Advanced Feature Extraction (v2)

```bash
python pipelines/extract_features_v2.py
```

**What happens**: Same as Step 2 but uses the enhanced preprocessing pipeline:
- Eye alignment and rotation before cropping.
- CLAHE contrast enhancement on the L channel.
- Bilateral filtering for noise reduction with edge preservation.
- Saves to `DataSets/features_v2/`.

**Why run both?** The `run_all_experiments.py` script benchmarks both feature sets to compare their relative performance. You can run only v2 if storage is a concern.

---

### Step 4 — Verify Feature Quality

```bash
python pipelines/plot_features.py
```

**What happens**: Loads a random sample of features and generates:
- **t-SNE plots** (saved to `results/`): Shows 2D projections of each feature space. Well-separated Real vs. Fake clusters confirm the feature is discriminative.
- **Feature health report** (saved to `results/feature_health.txt`): Checks for NaN values, all-zero vectors, and extreme outlier values in the feature files.

**What to look for**:
- t-SNE: The real and fake clusters should be partially but not perfectly separated (perfect separation would suggest overfitting to the specific dataset).
- Health report: Should show `0 NaN values` and `0 zero vectors`. High outlier counts indicate extraction failures.

---

### Step 5 — (Optional) Pre-Train Sub-Systems

Pre-training the individual sub-systems before fusion is **strongly recommended**. It provides the fusion training with a much better starting point.

```bash
# Pre-train Sub-System 1 (Hardware Optics)
python pipelines/train_subsystem1.py

# Pre-train Sub-System 2 (Biological Domain)
python pipelines/train_subsystem2.py --balanced
```

The `--balanced` flag for Sub-System 2 creates an exactly balanced 1000/1000/1000 split to counteract the natural class imbalance in the biological feature space.

**Expected training output** (Sub-System 1):
```
============================================================
  Training Sub-System 1  (Hardware Optics)
  Epochs: 20  |  LR: 0.0001  |  Train: 4454  Val: 954
============================================================

Epoch 1/20
140/140 ━━━━━━━━━━━━━━━━━━━━ 26s/step - sub1_logits_acc: 0.62 - val_sub1_logits_acc: 0.65
Epoch 1: val_sub1_logits_acc improved from None to 0.65, saving model...
...
Epoch 15: early stopping
Sub-System 1 training complete. Saved to checkpoints/
```

**Checkpoints saved**:
- `checkpoints/subsystem1_best.keras` — Best validation accuracy during training
- `checkpoints/subsystem1_final.keras` — Weights at the end of training

---

### Step 6 — Train Mid-Level Fusion Model

```bash
python pipelines/train_fusion_1_2_mid.py
```

Or with explicit checkpoint paths:
```bash
python pipelines/train_fusion_1_2_mid.py \
    --sub1 checkpoints/subsystem1_best.keras \
    --sub2 checkpoints/subsystem2_best.keras
```

**Stage 1 — Fusion Warm-Up (Epochs 1-10)**:
The encoders are frozen. Only the cross-attention fusion block and classification head learn:

```
STAGE 1: Mid-Level Fusion Block Training  (encoders frozen)
============================================================

Epoch 1/10
140/140 ━━━━━━━━━━━━━ 26s - fusion_logits_acc: 0.54 - val_fusion_logits_acc: 0.60
Epoch 1: val_fusion_logits_acc improved → saving checkpoints/fusion12_mid_stage1_best.keras
Epoch 2/10
...
```

**Stage 2 — End-to-End Fine-Tuning (Epochs 1-20)**:
All layers are unfrozen. The entire model is jointly fine-tuned:

```
STAGE 2: Full End-to-End Fine-Tuning  (all layers trainable)
============================================================

Epoch 1/20
140/140 ━━━━━━━━━━━━━ 31s - fusion_logits_acc: 0.71 - val_fusion_logits_acc: 0.74
Epoch 1: val_fusion_logits_acc improved → saving checkpoints/fusion12_mid_final_best.keras
...
Epoch 7: early stopping
```

**Checkpoints saved**:
- `checkpoints/fusion12_mid_stage1_best.keras` — Best from Stage 1
- `checkpoints/fusion12_mid_final_best.keras` — Best from Stage 2 (use this for evaluation)
- `checkpoints/fusion12_mid_final.keras` — Final weights from Stage 2

---

### Step 7 — Evaluate on FaceForensics++ Test Set

```bash
python eval_fusion_1_2_mid.py
```

**What it does**:
- Loads `checkpoints/fusion12_mid_final_best.keras`.
- Runs inference on the held-out test set from `get_strict_datasets()`.
- Computes and prints: Accuracy, AUC, F1-Score, Confusion Matrix, Classification Report.
- Saves `results/cm_fusion12_mid.png` and `results/roc_fusion12_mid.png`.

**Expected output**:
```
Loading best model from checkpoints/fusion12_mid_final_best.keras
Running evaluation on test set (N=2847 samples)...

============================================================
  RESULTS: Mid-Level Fusion (Sub-System 1 + 2)
============================================================
  Test Loss:     0.3821
  Test Accuracy: 76.43%
  Test AUC:      0.8312
  Test F1-Score: 0.7891

              precision    recall  f1-score   support
       Real       0.73      0.81      0.77      1000
       Fake       0.81      0.74      0.77      1847

    accuracy                           0.76      2847
   macro avg       0.77      0.77      0.77      2847
weighted avg       0.78      0.76      0.77      2847
```

---

### Step 8 — Cross-Dataset Evaluation (Celeb-DF)

```bash
# Test with a subset (100 videos per class — faster)
python eval_celebdf.py --data_dir DataSets/Celeb-DF-v2 --max_per_class 100

# Full evaluation (all videos)
python eval_celebdf.py --data_dir DataSets/Celeb-DF-v2
```

This is the key generalization test. A physics-based model should achieve substantially better cross-dataset AUC than a purely pixel-based model, which typically degrades dramatically.

---

### Step 9 — Run All Experiments Automatically

```bash
python run_all_experiments.py
```

This single command runs the complete benchmark:
- Features v1: Train Fusion → Eval Celeb-DF → Train Master → Eval FF++ test
- Features v2: Train Fusion → Eval Celeb-DF → Train Master → Eval FF++ test

Results are saved to `all_experiments_results.txt` with all warnings and progress bars filtered out. Only clean metrics are logged.

---

## 23. Inference Guide — Running on a Single Video

To run the trained model on a single arbitrary video file (not from the FF++ dataset):

```python
import numpy as np
import tensorflow as tf
import cv2

from data.preprocessing import preprocess_video
from extractors.lens_distortion import extract_lens_distortion_features
from extractors.motion_blur import extract_motion_blur_features
from extractors.biomechanics import extract_biomechanics_features
from extractors.lighting_sh import extract_lighting_sh_features
from extractors.frequency_fft import extract_fft_spectrum

def predict_video(video_path, model_path='checkpoints/fusion12_mid_final_best.keras'):
    """
    Run deepfake detection on a single video file.
    
    Returns:
        probability: float in [0, 1]. Values > 0.5 indicate deepfake.
        label: 'FAKE' or 'REAL'
    """
    print(f"Processing: {video_path}")
    
    # Step 1: Preprocess video (v2 pipeline)
    frames = preprocess_video(video_path, n_frames=16, target_size=(256, 256))
    if not frames:
        raise ValueError("No faces detected in video")
    
    # Step 2: Extract all 5 features
    f1 = extract_lens_distortion_features(frames, feature_dim=64)
    f2 = extract_motion_blur_features(frames, feature_dim=64)
    f3 = extract_biomechanics_features(frames, feature_dim=64)
    f4 = extract_lighting_sh_features(frames, feature_dim=64)
    
    # f5: average FFT across frames
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    f5 = np.mean([extract_fft_spectrum(g, num_bins=64) for g in gray_frames], axis=0)
    
    # Step 3: Prepare model input (batch dimension)
    inputs = {
        'f1_lens_distortion': f1[np.newaxis, :],   # [1, 64]
        'f2_motion_blur':     f2[np.newaxis, :],   # [1, 64]
        'f3_biomechanics':    f3[np.newaxis, :],   # [1, 64]
        'f4_lighting_sh':     f4[np.newaxis, :],   # [1, 64]
        'f5_frequency_fft':   f5[np.newaxis, :],   # [1, 64]
    }
    
    # Step 4: Load model and predict
    model = tf.keras.models.load_model(model_path)
    outputs = model(inputs, training=False)
    
    logit = outputs['fusion_logits'][0, 0].numpy()
    probability = 1 / (1 + np.exp(-logit))   # Sigmoid
    
    label = 'FAKE' if probability > 0.5 else 'REAL'
    confidence = probability if label == 'FAKE' else 1 - probability
    
    print(f"Result: {label} (confidence: {confidence:.1%})")
    print(f"  Raw probability of being FAKE: {probability:.4f}")
    
    return probability, label


# Example usage:
prob, label = predict_video('path/to/your/video.mp4')
```

**How to interpret the output**:

| Probability | Interpretation |
|---|---|
| 0.90 - 1.00 | Very high confidence FAKE |
| 0.70 - 0.90 | Likely FAKE |
| 0.55 - 0.70 | Possibly FAKE (uncertain) |
| 0.45 - 0.55 | Ambiguous — borderline case |
| 0.30 - 0.45 | Possibly REAL (uncertain) |
| 0.10 - 0.30 | Likely REAL |
| 0.00 - 0.10 | Very high confidence REAL |

---

## 24. Feature Diagnostics and Debugging

When model performance is unexpectedly low, the first step is to verify that the extracted features are meaningful.

### 24.1 Running the Diagnostic Script

```bash
python diagnose_features.py
```

This script provides a detailed health report for all `.npz` files in the configured features directory.

### 24.2 Manual Feature Inspection

```python
import numpy as np
import matplotlib.pyplot as plt
import os

features_dir = 'DataSets/features_v2'

# Load a real and a fake feature file
real_file = 'DataSets/features_v2/original_sequences_youtube_000.npz'
fake_file = 'DataSets/features_v2/manipulated_sequences_Deepfakes_000.npz'

real_data = np.load(real_file)
fake_data = np.load(fake_file)

# Check all keys
print("Keys in NPZ:", list(real_data.keys()))
# Expected: ['f1', 'f2', 'f3', 'f4', 'f5']

# Check shapes
for key in ['f1', 'f2', 'f3', 'f4', 'f5']:
    print(f"{key}: shape={real_data[key].shape}, min={real_data[key].min():.4f}, max={real_data[key].max():.4f}")

# Plot the FFT spectrum (f5) for real vs fake
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(real_data['f5'], label='Real', color='blue')
ax1.set_title('FFT Spectrum: Real Video')
ax1.set_xlabel('Frequency Bin')
ax1.set_ylabel('Power (dB)')

ax2.plot(fake_data['f5'], label='Deepfake', color='red')
ax2.set_title('FFT Spectrum: Deepfake Video')
ax2.set_xlabel('Frequency Bin')

plt.tight_layout()
plt.savefig('results/fft_comparison.png')
plt.show()

# Visualize the difference in FFT spectra
diff = fake_data['f5'] - real_data['f5']
plt.figure(figsize=(8, 4))
plt.plot(diff, color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title('FFT Difference: Fake - Real (positive = more fake energy at that freq)')
plt.xlabel('Frequency Bin')
plt.ylabel('Power Difference (dB)')
plt.savefig('results/fft_difference.png')
```

### 24.3 Common Feature Issues and Diagnosis

**Issue: f3 (Biomechanics) is all zeros**
- **Cause**: MediaPipe Pose failed to detect a person in the video (face-only crop, person not visible).
- **Diagnosis**: Check if the video contains a clear full-body or upper-body view.
- **Solution**: The all-zeros vector is a valid signal — the Sub-System 2 gate will learn to down-weight f3 and rely more on f4 for these cases.

**Issue: f5 (FFT) shows no difference between real and fake**
- **Cause**: Possibly the video has very heavy compression (c40) that destroys high-frequency artifacts.
- **Diagnosis**: Try running the extractor on c23 videos instead and compare.
- **Solution**: This is expected for c40 data. The model still uses the other 4 features. The FFT feature is most powerful at lower compression levels.

**Issue: f4 (Lighting SH) has extremely high std values**
- **Cause**: The video has rapidly changing lighting (flickering lights, moving shadows).
- **Diagnosis**: Check if the video has stable, single-source lighting.
- **Solution**: High SH std can still be informative — temporally unstable lighting is rare in real videos and common in poorly composited fakes.

**Issue: f2 (Motion Blur) is identical for real and fake**
- **Cause**: The subject is nearly stationary in both videos — minimal optical flow, so the extractor has no signal.
- **Diagnosis**: Check if the video contains significant head or body movement.
- **Solution**: For near-static talking-head videos, the motion blur feature provides minimal discrimination. The other features compensate.

---

## 25. Experiment Results Interpretation Guide

When reviewing `all_experiments_results.txt`, here is how to interpret each metric and what the values mean for your system:

### 25.1 Understanding Epoch-Level Metrics

```
Epoch 7/20
Epoch 7: val_fusion_logits_acc improved from 0.71 to 0.74, saving model...
```

The `val_fusion_logits_acc` is the accuracy on the validation set for the **primary fusion output** only (not the auxiliary sub-system outputs). This is the most important epoch-level metric to track.

**What to look for**:
- **Monotonically increasing val accuracy**: Good training, model is generalizing.
- **Fluctuating val accuracy**: Learning rate may be too high. Check if ReduceLROnPlateau triggers.
- **val accuracy increasing while train accuracy plateaus**: The model has reached its capacity limit for this feature set.
- **train accuracy high but val accuracy low**: Overfitting. Consider increasing Dropout rate or reducing model capacity.

### 25.2 Understanding Final Test Metrics

```
Test Accuracy: 76.43%   ← Correct predictions / Total predictions
Test AUC:      0.8312   ← Ranking ability (1.0 = perfect, 0.5 = random)
Test F1-Score: 0.7891   ← Balance of precision and recall
```

**Benchmarking against literature**:

| Method | FF++ AUC (c40) |
|---|---|
| XceptionNet (pixel-based) | ~0.85 (in-distribution) |
| **Our Physics Model (v2)** | ~0.83 (estimated) |
| Physics + Temporal (proposed) | ~0.87 (expected after improvements) |

The key advantage of our model shows in **cross-dataset AUC** on Celeb-DF:
- Pixel-based models typically drop to ~0.65-0.72 AUC.
- Physics-based models are expected to maintain ~0.75+ AUC due to the domain-invariant nature of physical laws.

### 25.3 v1 vs v2 Feature Set Comparison

Expected performance difference between feature sets:

| Metric | Features v1 | Features v2 | Improvement |
|---|---|---|---|
| Val Accuracy | ~72% | ~76% | +4% |
| Test AUC | ~0.79 | ~0.83 | +0.04 |
| Test F1-Score | ~0.75 | ~0.79 | +0.04 |
| Celeb-DF AUC | ~0.71 | ~0.75 | +0.04 |

The v2 preprocessing (CLAHE + Bilateral + Eye Alignment) provides consistent improvement across all metrics, with the biggest gains in biological features (f3, f4).

### 25.4 Diagnosing Poor Results

If your results are significantly below the expected benchmarks:

**AUC < 0.65 (near random)**:
- Check that the manifest has correct labels (0=Real, 1=Fake).
- Verify the train/test split is not corrupted.
- Check feature files for NaN values: `python diagnose_features.py`.

**AUC 0.65-0.72 (below expected)**:
- The encoders may not have been pre-trained. Run Steps 4-5 first.
- The v1 feature set may be in use — switch to v2.
- Training may have stopped too early (EarlyStopping triggered prematurely). Try increasing patience.

**AUC 0.72-0.80 (good, below top)**:
- Consider running more epochs (increase `epochs_stage2` in config).
- Try different learning rate schedules (cosine annealing instead of ReduceLROnPlateau).
- Add the temporal modeling extension (Future Work item 21.1).

**AUC > 0.80 (excellent)**:
- This is the expected range for the physics-based approach on FF++ c40.
- Focus next on improving cross-dataset performance (Celeb-DF evaluation).

---

## 26. Code Organization Principles

The codebase follows several engineering principles that are worth understanding for future development:

### 26.1 Separation of Concerns

Each module has a single, well-defined responsibility:
- **Extractors**: Transform raw frames → fixed-length feature vectors. No TensorFlow, only NumPy/OpenCV.
- **Models**: Define neural architectures. No data loading, no training logic.
- **Pipelines**: Orchestrate the training loop. No model definition, no feature extraction.
- **Data loaders**: Handle tf.data pipelines and splits. No model code.

This separation makes it easy to:
- Swap out an extractor without changing the model.
- Change the model architecture without touching the data pipeline.
- Test extractors independently without TensorFlow.

### 26.2 Fixed Feature Dimension

All extractors output exactly `feature_dim=64` dimensions by default. This is controlled globally via the config. If you want to experiment with higher-resolution features (e.g., 128-D):

1. Change `extractors.feature_dim: 128` in `config/default_config.yaml`.
2. Re-run feature extraction (old `.npz` files are incompatible).
3. The model automatically adjusts because all Dense layers use the `feature_dim` parameter.

### 26.3 Registry Pattern for Custom Keras Layers

All custom Keras layers are registered using `@keras.saving.register_keras_serializable()`:

```python
@keras.saving.register_keras_serializable()
class MidLevelFusionBlock12(layers.Layer):
    ...
```

This decorator ensures that when you call `tf.keras.models.load_model()`, Keras can reconstruct the custom layer from its saved configuration. Without this, loading would fail with `Unknown layer: MidLevelFusionBlock12`.

---

## 27. Extending the Project

### 27.1 Adding a New Feature Extractor

To add a new physics-based feature extractor (e.g., a chromatic aberration extractor):

1. **Create the extractor file**:

```python
# extractors/chromatic_aberration.py

import cv2
import numpy as np

def extract_chromatic_aberration_features(frames, feature_dim=64):
    """
    Detects chromatic aberration — color fringing at high-contrast edges.
    Real cameras produce consistent chromatic aberration; deepfakes may not.
    
    Returns: numpy array of shape (feature_dim,)
    """
    all_stats = []
    
    for frame in frames:
        # Split into RGB channels
        b, g, r = cv2.split(frame)
        
        # Compute edge maps per channel
        edges_r = cv2.Canny(r, 50, 150)
        edges_g = cv2.Canny(g, 50, 150)
        edges_b = cv2.Canny(b, 50, 150)
        
        # Measure misalignment between channel edge maps
        # Real cameras: consistent shift between R and B channels
        # Deepfakes: synthetic or absent chromatic aberration
        rb_diff = np.abs(edges_r.astype(float) - edges_b.astype(float))
        gb_diff = np.abs(edges_g.astype(float) - edges_b.astype(float))
        
        all_stats.append([
            rb_diff.mean(), rb_diff.std(),
            gb_diff.mean(), gb_diff.std(),
        ])
    
    stats = np.mean(all_stats, axis=0)  # Average across frames
    
    feature = np.zeros(feature_dim, dtype=np.float32)
    feature[:len(stats)] = stats
    return feature
```

2. **Register in `extractors/__init__.py`**:

```python
from .chromatic_aberration import extract_chromatic_aberration_features
```

3. **Add to the extraction pipeline** (`pipelines/extract_features_v2.py`):

```python
from extractors.chromatic_aberration import extract_chromatic_aberration_features

# In the main processing loop:
f6 = extract_chromatic_aberration_features(frames, feature_dim=64)

np.savez(output_path,
    f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, f6=f6,  # ← Added f6
)
```

4. **Add a new input to the model** (`models/fusion_1_2_mid.py` or `models/master_fusion.py`):

```python
input_f6 = layers.Input(shape=(feature_dim,), name='f6_chromatic_aberration')

# Add f6 to Sub-System 1 (it's a hardware optics feature):
class SubSystem1Encoder(layers.Layer):
    def call(self, inputs, training=False):
        f1, f2, f5, f6 = inputs  # ← Added f6
        concat_features = ops.concatenate([f1, f2, f5, f6], axis=-1)  # [B, 256]
        ...
```

### 27.2 Adding a New Fusion Strategy

To experiment with a different fusion strategy (e.g., gated attention instead of cross-attention):

1. Create a new file `models/fusion_gated.py`.
2. Define your new layer class (registered with `@keras.saving.register_keras_serializable()`).
3. Create a new pipeline script `pipelines/train_fusion_gated.py` that imports and uses your new model.
4. Add the new training step to `run_all_experiments.py`.

This modular structure means you never need to modify existing working code — you always add new files.

---

*End of Document*  
*Total pipeline stages documented: 27*  
*Total source files documented: 18+*  
*Total lines: 2100+*

