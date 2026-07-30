"""
motion_blur.py  –  Sub-System 1 / Feature 2 (Optical Flow)
-----------------------------------------------------------
Extracts a feature vector that captures motion-blur consistency.

Physical cameras produce spatially-coherent motion blur dictated by shutter
speed and object velocity.  Deepfake generators typically render sharp
per-frame outputs and then (optionally) apply post-hoc blur that lacks the
correct directional coherence.

Strategy for c40 data:
    Increase Farneback iterations (5) and window size (21) to handle the
    blocky motion fields introduced by heavy H.264 compression.
"""

import cv2
import numpy as np


def _compute_dense_optical_flow(prev_gray, curr_gray,
                                 pyr_scale=0.5,
                                 levels=3,
                                 winsize=21,
                                 iterations=5,
                                 poly_n=7,
                                 poly_sigma=1.5):
    """Compute dense optical flow with Farneback, tuned for c40."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0,
    )
    return flow  # shape (H, W, 2)


def _flow_statistics(flow):
    """
    Compute statistical descriptors of an optical-flow field:
        magnitude mean/std, angle mean/std, divergence mean/std,
        curl mean/std  →  8 scalars per frame pair.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Divergence (∂u/∂x + ∂v/∂y)
    du_dx = np.gradient(flow[..., 0], axis=1)
    dv_dy = np.gradient(flow[..., 1], axis=0)
    divergence = du_dx + dv_dy

    # Curl (∂v/∂x − ∂u/∂y)
    dv_dx = np.gradient(flow[..., 1], axis=1)
    du_dy = np.gradient(flow[..., 0], axis=0)
    curl = dv_dx - du_dy

    stats = np.array([
        mag.mean(), mag.std(),
        ang.mean(), ang.std(),
        divergence.mean(), divergence.std(),
        curl.mean(), curl.std(),
    ], dtype=np.float32)

    return stats


def extract_motion_blur_features(frames, feature_dim=64):
    """
    Extract a fixed-length motion-blur coherence feature vector from a
    sequence of preprocessed frames.

    The feature is built by computing dense optical flow between consecutive
    frame pairs, extracting per-pair statistics, and aggregating them into a
    fixed-length vector.

    Args:
        frames:      List of BGR numpy arrays (already c40-smoothed & aligned).
        feature_dim: Length of the output feature vector.

    Returns:
        numpy array of shape (feature_dim,).
    """
    if len(frames) < 2:
        return np.zeros(feature_dim, dtype=np.float32)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    pair_stats = []
    for i in range(len(grays) - 1):
        flow = _compute_dense_optical_flow(grays[i], grays[i + 1])
        pair_stats.append(_flow_statistics(flow))

    # Stack: (N-1, 8)
    pair_stats = np.stack(pair_stats, axis=0)

    # Aggregate across time: mean + std of each statistic  →  16 values
    agg = np.concatenate([pair_stats.mean(axis=0), pair_stats.std(axis=0)])

    # Pad or truncate to feature_dim
    feature = np.zeros(feature_dim, dtype=np.float32)
    length = min(len(agg), feature_dim)
    feature[:length] = agg[:length]

    return feature
