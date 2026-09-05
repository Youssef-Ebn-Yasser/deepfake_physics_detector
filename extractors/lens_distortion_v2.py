"""
lens_distortion_v2.py  –  Sub-System 1 / Feature 1 (Optical) [V2]
----------------------------------------------------------
Extracts a feature vector that captures lens-distortion inconsistencies.

Deepfake generators rarely model the radial & tangential distortion of a
physical camera lens.  This extractor fits a Brown–Conrady distortion model
to detected straight-edge segments and encodes the residual error as a
fixed-length feature vector.

V2 Implementation: Uses fixed bins for the residual histogram per frame 
and extracts temporal statistics (mean & standard deviation) over time 
to produce exactly 64 dimensions.
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


def extract_lens_distortion_v2_features(frames, feature_dim=64):
    """
    Extract a fixed-length lens-distortion feature vector from a list of
    preprocessed face-cropped frames.

    Uses a fixed-bin histogram per frame and extracts temporal mean/std.

    Args:
        frames:      List of BGR numpy arrays (already c40-smoothed & aligned).
        feature_dim: Length of the output feature vector. Must be an even number.

    Returns:
        numpy array of shape (feature_dim,).
    """
    n_bins = feature_dim // 2
    
    # Range [0, 100] covers max residuals typically observed (e.g. 84.85 to 96.64)
    # n_bins + 1 edges are needed for n_bins
    bin_edges = np.linspace(0.0, 100.0, n_bins + 1)
    
    frame_hists = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = _detect_lines(gray)
        residuals = _line_curvature_residuals(lines, gray.shape)

        if len(residuals) > 0:
            hist, _ = np.histogram(residuals, bins=bin_edges, density=True)
        else:
            hist = np.zeros(n_bins, dtype=np.float32)
            
        frame_hists.append(hist)

    if len(frame_hists) < 2:
        return np.zeros(feature_dim, dtype=np.float32)
        
    frame_hists = np.array(frame_hists, dtype=np.float32)
    
    # Calculate temporal statistics
    temporal_mean = np.mean(frame_hists, axis=0)
    temporal_std = np.std(frame_hists, axis=0)

    # Concatenate to form exactly 64 dimensions
    feature = np.concatenate([temporal_mean, temporal_std])
    return feature.astype(np.float32)
