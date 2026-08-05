# Models Architecture Documentation

This document explains the core deep learning architecture of the project located inside the `models/` directory. The architecture is written in Keras (TensorFlow) and is divided into individual sub-systems and a master fusion controller.

---

## 1. `subsystem1.py` (Hardware Optics Encoder)
This file defines the `SubSystem1Encoder` class. 
**Input**: Hardware-based features:
- $f_1$ (Lens Distortion)
- $f_2$ (Motion Blur)
- $f_5$ (Frequency FFT)

**Architecture Details**:
- **Feature Embedding**: Each input feature is passed through its own independent Multi-Layer Perceptron (MLP) mapping layer (Dense 128 -> ReLU -> Dropout). This brings the drastically different feature types into the same dimensional space.
- **Concatenation & Normalization**: The three embedded features are concatenated together. A `LayerNormalization` block is applied to stabilize the hardware domain representation.
- **Output ($Z_1$)**: A final Dense bottleneck layer outputs a unified 256-dimensional vector $Z_1$, which represents the entire hardware/optical signature of the video.

---

## 2. `subsystem2.py` (Biological Encoder)
This file defines the `SubSystem2Encoder` class.
**Input**: Biological features:
- $f_3$ (rPPG Pulse Signal)
- $f_4$ (Lighting Spherical Harmonics)

**Architecture Details**:
- **Feature Embedding**: Similar to Sub-System 1, $f_3$ and $f_4$ are independently embedded via Dense layers into a shared dimensional space.
- **Sigmoid Gating Mechanism**: Because biological features are highly sensitive to noise (e.g., poor lighting ruining the rPPG signal), they are merged using a learned Gating Mechanism rather than simple concatenation. A sigmoid layer acts as a gate, deciding how much weight to give the pulse vs. the lighting environment dynamically for each frame.
- **Output ($Z_2$)**: Outputs a unified 256-dimensional vector $Z_2$, representing the biological signature of the face.

---

## 3. `master_fusion.py` (Cross-Attention & Master Model)
This file is the brain of the project. It contains two main components that combine $Z_1$ and $Z_2$ together.

### `MidLevelFusionBlock12`
This is a custom Keras Layer that implements **Bidirectional Cross-Attention**.
Instead of simply adding or concatenating $Z_1$ and $Z_2$ (which is basic Late Fusion), this block lets the two domains "talk" to each other:
1. **$Z_1$ attends to $Z_2$**: Using $Z_1$ as the Query and $Z_2$ as the Key/Value. (e.g. "Do the hardware lens flares align with the biological lighting map?"). Output is $Z_1'$.
2. **$Z_2$ attends to $Z_1$**: Using $Z_2$ as the Query and $Z_1$ as the Key/Value. Output is $Z_2'$.
3. **Merging**: $Z_1'$ and $Z_2'$ are combined using a Hadamard (Element-wise) Product.
4. **Classification Head**: The merged representation goes through Dense(256) -> Dropout -> Dense(128) -> Dense(1, Sigmoid) to output the final Fake (1) or Real (0) prediction.

### `MasterModel`
This is the end-to-end wrapper class. It instantiates `SubSystem1Encoder`, `SubSystem2Encoder`, and `MidLevelFusionBlock12`. It takes all 5 raw features simultaneously and passes them through the respective encoders and the fusion block to generate the final prediction.
