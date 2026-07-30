"""
preprocessing3.py
-----------------
Physics-Aware Consistency preprocessing — pure OpenCV, zero external model files.

Because the input frames are already 256x256 face-centred crops produced by the
existing preprocessing pipeline, we do NOT need a secondary face detector.
The entire frame IS the face region.

Pipeline per frame:
  A. sRGB  -> Linear RGB
  B. Skin mask via YCrCb colour segmentation on the full frame
  C. 6-point geometric landmark approximation (eye / nose / mouth positions
     derived from the image centre + colour-guided eye finding)
  D. Head-pose estimation (PnP with the 6 approximate points)
  E. Surface normal estimation from image gradients
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PhysicsFrame:
    image_linear: np.ndarray             # (H, W, 3) float32 linear RGB
    skin_mask: np.ndarray                # (H, W) uint8 binary mask
    landmarks: np.ndarray                # (6, 2) pixel coords
    pose: Tuple[np.ndarray, np.ndarray]  # (rvec, tvec)
    normals: np.ndarray                  # (H, W, 3) float32


class PhysicsPreprocessor:
    """
    Fully model-free physics preprocessor that works on pre-cropped 256x256
    face frames (the output of data.preprocessing.preprocess_video).
    """

    # 3D model points for PnP head-pose estimation (standard frontal face model)
    _MODEL_3D = np.array([
        ( 0.0,    0.0,    0.0),   # Nose tip
        ( 0.0, -330.0,  -65.0),   # Chin
        (-225.0,  170.0, -135.0), # Left eye outer corner
        ( 225.0,  170.0, -135.0), # Right eye outer corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        ( 150.0, -150.0, -125.0), # Right mouth corner
    ], dtype=np.float64)

    # ------------------------------------------------------------------
    # sRGB -> linear
    # ------------------------------------------------------------------

    def srgb_to_linear(self, image: np.ndarray) -> np.ndarray:
        """Standard sRGB gamma -> linear light conversion."""
        img = image.astype(np.float32) / 255.0
        lo = img <= 0.04045
        linear = np.empty_like(img)
        linear[lo]  = img[lo]  / 12.92
        linear[~lo] = np.power((img[~lo] + 0.055) / 1.055, 2.4)
        return linear

    # ------------------------------------------------------------------
    # Skin mask
    # ------------------------------------------------------------------

    def get_skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        YCrCb skin-colour segmentation on the full (pre-cropped) face frame.
        Cr ∈ [133, 173]  Cb ∈ [77, 127]  (standard skin range).
        """
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0,   133,  77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        return mask

    # ------------------------------------------------------------------
    # Landmark approximation (geometry only, no model)
    # ------------------------------------------------------------------

    def get_approx_landmarks(self, h: int, w: int) -> np.ndarray:
        """
        Approximate 6 key landmark positions from image dimensions.
        The face-crop is assumed to be centred and front-facing.
        Returns (6, 2) array in pixel coordinates [x, y].

        Point order matches _MODEL_3D:
          0 = nose tip, 1 = chin, 2 = left eye, 3 = right eye,
          4 = left mouth, 5 = right mouth
        """
        pts = np.array([
            [w * 0.50, h * 0.55],   # nose tip
            [w * 0.50, h * 0.88],   # chin
            [w * 0.30, h * 0.38],   # left eye
            [w * 0.70, h * 0.38],   # right eye
            [w * 0.38, h * 0.72],   # left mouth
            [w * 0.62, h * 0.72],   # right mouth
        ], dtype=np.float32)
        return pts

    # ------------------------------------------------------------------
    # Head pose
    # ------------------------------------------------------------------

    def estimate_pose(self, landmarks: np.ndarray,
                      h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        camera_matrix = np.array([
            [w,   0, w / 2],
            [0,   w, h / 2],
            [0,   0,     1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        _, rvec, tvec = cv2.solvePnP(
            self._MODEL_3D,
            landmarks.astype(np.float64),
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        return rvec, tvec

    # ------------------------------------------------------------------
    # Normal estimation
    # ------------------------------------------------------------------

    def estimate_normals(self, image_linear: np.ndarray) -> np.ndarray:
        """Surface normals approximated via image gradients (Shape-from-Shading)."""
        gray = cv2.cvtColor(image_linear, cv2.COLOR_RGB2GRAY)
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        normals = np.stack([-gx, -gy, np.ones_like(gx)], axis=2)
        norms   = np.linalg.norm(normals, axis=2, keepdims=True)
        return normals / (norms + 1e-8)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[PhysicsFrame]:
        """
        Process a single pre-cropped BGR face frame.
        Returns None if the frame is empty / all-black.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]

        # Skip near-black frames (invalid / padded)
        if float(frame_bgr.mean()) < 2.0:
            return None

        rgb          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_linear = self.srgb_to_linear(rgb)
        skin_mask    = self.get_skin_mask(frame_bgr)
        landmarks    = self.get_approx_landmarks(h, w)
        pose         = self.estimate_pose(landmarks, h, w)
        normals      = self.estimate_normals(image_linear)

        return PhysicsFrame(
            image_linear=image_linear,
            skin_mask=skin_mask,
            landmarks=landmarks,
            pose=pose,
            normals=normals,
        )
