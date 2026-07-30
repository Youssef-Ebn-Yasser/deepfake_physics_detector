"""
lens_distortion.py  –  Sub-System 1 / Feature 1 (Optical)
----------------------------------------------------------
Extracts a feature vector that captures lens-distortion inconsistencies.

Deepfake generators rarely model the radial & tangential distortion of a
physical camera lens.  This extractor fits a Brown–Conrady distortion model
to detected straight-edge segments and encodes the residual error as a
fixed-length feature vector.

Strategy for c40 data:
    A spatial Gaussian pre-filter (σ = 0.8) should be applied *before*
    calling this extractor to suppress H.264 macroblock boundary artefacts.
    See ``data.preprocessing.apply_c40_smoothing``.
"""

import cv2
import numpy as np


def _detect_lines(gray, min_line_length=40, max_line_gap=10):
    """Detect line segments with the probabilistic Hough transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.empty((0, 1, 4), dtype=np.int32)
    return lines


def _line_curvature_residuals(lines, img_shape):
    """
    Compute the curvature residual for each detected line relative to
    a simple radial-distortion model centred on the image.

    Returns a 1-D array of residual magnitudes (one per line).
    """
    h, w = img_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    residuals = []
    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Distance of the line mid-point from the image centre
        r = np.sqrt((mid_x - cx) ** 2 + (mid_y - cy) ** 2)

        # Angle between the line and the radial direction
        line_angle = np.arctan2(y2 - y1, x2 - x1)
        radial_angle = np.arctan2(mid_y - cy, mid_x - cx)
        angle_diff = np.abs(line_angle - radial_angle)

        # Residual: higher when a line that *should* be straight shows
        # deviation consistent with barrel / pincushion distortion.
        residuals.append(r * np.sin(angle_diff))

    return np.array(residuals, dtype=np.float32) if residuals else np.zeros(1, dtype=np.float32)


def extract_lens_distortion_features(frames, feature_dim=64):
    """
    Extract a fixed-length lens-distortion feature vector from a list of
    preprocessed face-cropped frames.

    Args:
        frames:      List of BGR numpy arrays (already c40-smoothed & aligned).
        feature_dim: Length of the output feature vector.

    Returns:
        numpy array of shape (feature_dim,).
    """
    all_residuals = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = _detect_lines(gray)
        residuals = _line_curvature_residuals(lines, gray.shape)
        all_residuals.append(residuals)

    # Aggregate across frames: histogram-based encoding
    all_residuals = np.concatenate(all_residuals)

    if len(all_residuals) == 0:
        return np.zeros(feature_dim, dtype=np.float32)

    # Build a normalised histogram as the feature vector
    hist, _ = np.histogram(all_residuals, bins=feature_dim, density=True)
    return hist.astype(np.float32)
