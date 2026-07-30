"""
lighting_sh.py  –  Sub-System 2 / Feature 4 (Spherical Harmonics)
------------------------------------------------------------------
Extracts a feature vector encoding lighting consistency via Spherical
Harmonics (SH) analysis.

Physical scenes have a single, consistent environment illumination.
Deepfake generators often produce faces whose shading is inconsistent
with the background or temporally unstable.  This extractor estimates
low-order SH coefficients from facial normals and measures their
consistency.

Robustness note:
    SH-based lighting features are highly stable under c40 compression
    because they operate on low-frequency shading signals rather than
    high-frequency texture detail.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


# First 9 real SH basis functions (order 0, 1, 2) evaluated at (nx, ny, nz)
def _sh_basis(normals):
    """
    Evaluate the first 9 real spherical harmonic basis functions.

    Args:
        normals: (N, 3) array of unit normals.

    Returns:
        (N, 9) array of SH basis values.
    """
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    basis = np.stack([
        np.ones_like(nx),                       # Y_0^0
        ny,                                      # Y_1^{-1}
        nz,                                      # Y_1^0
        nx,                                      # Y_1^1
        nx * ny,                                 # Y_2^{-2}
        ny * nz,                                 # Y_2^{-1}
        3 * nz ** 2 - 1,                         # Y_2^0
        nx * nz,                                 # Y_2^1
        nx ** 2 - ny ** 2,                       # Y_2^2
    ], axis=-1)
    return basis.astype(np.float32)


def _estimate_face_normals(frame):
    """
    Estimate approximate per-pixel face normals from a 2D greyscale gradient.

    This is a simplified Shape-from-Shading approach:  the gradient of the
    luminance image is treated as the tangent plane, and the normal is
    derived accordingly.

    Returns:
        (H*W, 3) array of approximate unit normals, or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    # Normal: (-dI/dx, -dI/dy, 1), then normalise
    normals = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norms

    return normals.reshape(-1, 3)


def _fit_sh_coefficients(normals, intensities, n_coeffs=9):
    """
    Fit SH coefficients via least-squares:  I ≈ B @ c

    Args:
        normals:     (N, 3)
        intensities: (N,)
        n_coeffs:    Number of SH coefficients (default 9 = order 2).

    Returns:
        (n_coeffs,) array of SH coefficients.
    """
    B = _sh_basis(normals)[:, :n_coeffs]
    # Least squares: c = (B^T B)^{-1} B^T I
    try:
        coeffs, _, _, _ = np.linalg.lstsq(B, intensities, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(n_coeffs, dtype=np.float32)
    return coeffs.astype(np.float32)


def extract_lighting_sh_features(frames, feature_dim=64):
    """
    Extract a fixed-length SH lighting-consistency feature vector.

    Pipeline:
        1. Estimate per-pixel face normals (shape-from-shading approx).
        2. Fit order-2 SH coefficients per frame.
        3. Measure temporal consistency (mean, std, pairwise cosine
           similarity) of the SH coefficient vectors.
        4. Aggregate into a fixed-length feature vector.

    Args:
        frames:      List of BGR numpy arrays.
        feature_dim: Length of the output feature vector.

    Returns:
        numpy array of shape (feature_dim,).
    """
    sh_coeffs_list = []

    for frame in frames:
        normals = _estimate_face_normals(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        intensities = gray.ravel()

        coeffs = _fit_sh_coefficients(normals, intensities)
        sh_coeffs_list.append(coeffs)

    if len(sh_coeffs_list) < 2:
        return np.zeros(feature_dim, dtype=np.float32)

    sh_seq = np.stack(sh_coeffs_list, axis=0)  # (T, 9)

    # Temporal statistics of SH coefficients
    mean_coeffs = sh_seq.mean(axis=0)   # (9,)
    std_coeffs = sh_seq.std(axis=0)     # (9,)

    # Pairwise cosine similarity between consecutive frames
    cos_sims = []
    for i in range(len(sh_seq) - 1):
        a, b = sh_seq[i], sh_seq[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        cos_sims.append(cos_sim)

    cos_sims = np.array(cos_sims, dtype=np.float32)
    cos_stats = np.array([cos_sims.mean(), cos_sims.std()], dtype=np.float32)

    # Temporal derivative of SH coefficients
    sh_diff = np.diff(sh_seq, axis=0)
    diff_mean = sh_diff.mean(axis=0)  # (9,)
    diff_std = sh_diff.std(axis=0)    # (9,)

    raw = np.concatenate([mean_coeffs, std_coeffs, cos_stats, diff_mean, diff_std])

    feature = np.zeros(feature_dim, dtype=np.float32)
    length = min(len(raw), feature_dim)
    feature[:length] = raw[:length]
    return feature
