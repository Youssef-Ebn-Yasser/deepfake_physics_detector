"""
biomechanics.py  --  Sub-System 2 / Feature 3 (3D Pose Kinematics)
------------------------------------------------------------------
Extracts a feature vector encoding biomechanical plausibility.

Real human motion obeys Newtonian mechanics -- joint angles, angular
velocities, and accelerations follow smooth, physically-consistent
trajectories.  Deepfake generators frequently produce subtle but
detectable violations of these constraints.

This extractor uses MediaPipe Pose (Tasks API for v1.0+, or legacy
Solutions API for older versions) to obtain 3D landmark trajectories
and then computes kinematic statistics as a fixed-length feature vector.

Robustness note:
    Biomechanical features are highly stable under c40 compression because
    they rely on landmark *positions* rather than pixel-level detail.
"""

import os
import urllib.request
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Pose backend detection
# ---------------------------------------------------------------------------
_POSE_BACKEND = None  # 'tasks', 'solutions', or None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        _POSE_BACKEND = 'solutions'
    else:
        # MediaPipe >= 1.0 uses the Tasks API
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as _bo
        _POSE_BACKEND = 'tasks'
except ImportError:
    mp = None

# Path for the downloaded pose model (MediaPipe Tasks API)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'mediapipe')
_POSE_MODEL_PATH = os.path.join(_MODEL_DIR, 'pose_landmarker_lite.task')
_POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_lite/float16/latest/'
    'pose_landmarker_lite.task'
)


# Key joint triplets for angle computation (shoulder-elbow-wrist, etc.)
_JOINT_TRIPLETS = [
    (11, 13, 15),  # Left shoulder - elbow - wrist
    (12, 14, 16),  # Right shoulder - elbow - wrist
    (23, 25, 27),  # Left hip - knee - ankle
    (24, 26, 28),  # Right hip - knee - ankle
    (11, 23, 25),  # Left shoulder - hip - knee
    (12, 24, 26),  # Right shoulder - hip - knee
]


def _ensure_pose_model():
    """Download the pose landmarker model if it doesn't exist."""
    if os.path.exists(_POSE_MODEL_PATH):
        return True
    try:
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"  Downloading pose model to {_POSE_MODEL_PATH} ...")
        urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL_PATH)
        print(f"  Done.")
        return True
    except Exception as e:
        print(f"  [WARN] Could not download pose model: {e}")
        return False


def _compute_angle(a, b, c):
    """Angle at vertex *b* formed by points a-b-c (in radians)."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Landmark extraction — Tasks API (MediaPipe >= 1.0)
# ---------------------------------------------------------------------------

def _extract_landmarks_tasks(frame):
    """Extract 3D pose landmarks using the MediaPipe Tasks API (v1.0+)."""
    if not _ensure_pose_model():
        return None

    BaseOptions = _bo.BaseOptions
    PoseLandmarker = mp_vision.PoseLandmarker
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in result.pose_landmarks[0]
        ], dtype=np.float32)
        return landmarks


# ---------------------------------------------------------------------------
# Landmark extraction — Solutions API (MediaPipe < 1.0)
# ---------------------------------------------------------------------------

def _extract_landmarks_solutions(frame):
    """Extract 3D pose landmarks using the legacy Solutions API."""
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks is None:
            return None

        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)
        return landmarks


def _extract_landmarks(frame):
    """Route to the correct backend."""
    if _POSE_BACKEND == 'tasks':
        return _extract_landmarks_tasks(frame)
    elif _POSE_BACKEND == 'solutions':
        return _extract_landmarks_solutions(frame)
    else:
        return None


def _joint_angles_from_landmarks(landmarks):
    """Compute joint angles for each triplet.  Returns array of shape (N_triplets,)."""
    angles = []
    for a_idx, b_idx, c_idx in _JOINT_TRIPLETS:
        angle = _compute_angle(
            landmarks[a_idx],
            landmarks[b_idx],
            landmarks[c_idx],
        )
        angles.append(angle)
    return np.array(angles, dtype=np.float32)


def extract_biomechanics_features(frames, feature_dim=64):
    """
    Extract a fixed-length biomechanical plausibility feature vector.

    Pipeline:
        1. Detect 3D pose landmarks per frame.
        2. Compute joint angles for predefined triplets.
        3. Derive angular velocity and acceleration across time.
        4. Aggregate into a fixed-length feature vector.

    Args:
        frames:      List of BGR numpy arrays.
        feature_dim: Length of the output feature vector.

    Returns:
        numpy array of shape (feature_dim,).
    """
    if _POSE_BACKEND is None:
        return np.zeros(feature_dim, dtype=np.float32)

    all_angles = []

    for frame in frames:
        lm = _extract_landmarks(frame)
        if lm is not None:
            angles = _joint_angles_from_landmarks(lm)
            all_angles.append(angles)

    if len(all_angles) < 2:
        return np.zeros(feature_dim, dtype=np.float32)

    angle_seq = np.stack(all_angles, axis=0)  # (T, N_triplets)

    # Angular velocity  (1st finite difference)
    angular_vel = np.diff(angle_seq, axis=0)       # (T-1, N_triplets)
    # Angular acceleration  (2nd finite difference)
    angular_acc = np.diff(angular_vel, axis=0)      # (T-2, N_triplets)

    # Statistics per joint triplet
    stats_parts = []
    for data in [angle_seq, angular_vel, angular_acc]:
        if data.shape[0] > 0:
            stats_parts.extend([
                data.mean(axis=0),
                data.std(axis=0),
            ])
        else:
            stats_parts.extend([
                np.zeros(angle_seq.shape[1], dtype=np.float32),
                np.zeros(angle_seq.shape[1], dtype=np.float32),
            ])

    raw = np.concatenate(stats_parts)  # 6 * N_triplets

    feature = np.zeros(feature_dim, dtype=np.float32)
    length = min(len(raw), feature_dim)
    feature[:length] = raw[:length]
    return feature
