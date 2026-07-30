"""
preprocessing.py
----------------
Frame-level preprocessing utilities optimised for FaceForensics++ c40
(heavy H.264 compression) data.

Key strategies:
    1. Spatial Gaussian pre-filtering (sigma=0.8) to suppress macroblock
       boundary artifacts before optical feature extraction.
    2. Face detection via OpenCV FaceDetectorYN (OpenCV 5.0+) with
       a centre-crop fallback.
    3. Uniform temporal sampling of N frames from a video clip.
"""

import os
import urllib.request

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Face detector  (OpenCV FaceDetectorYN — built into OpenCV 4.5.4+)
# ---------------------------------------------------------------------------
_FACE_DETECTOR = None

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'opencv')
_YUNET_MODEL_PATH = os.path.join(_MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
_YUNET_MODEL_URL = (
    'https://github.com/opencv/opencv_zoo/raw/main/models/'
    'face_detection_yunet/face_detection_yunet_2023mar.onnx'
)


def _ensure_yunet_model():
    """Download the YuNet face detection model if not present."""
    if os.path.exists(_YUNET_MODEL_PATH):
        return True
    try:
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"  Downloading YuNet face model ...")
        urllib.request.urlretrieve(_YUNET_MODEL_URL, _YUNET_MODEL_PATH)
        print(f"  Saved to {_YUNET_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"  [WARN] Could not download YuNet model: {e}")
        return False


def _get_face_detector(width, height):
    """Create or reconfigure the FaceDetectorYN instance."""
    global _FACE_DETECTOR

    if not hasattr(cv2, 'FaceDetectorYN'):
        return None

    if not _ensure_yunet_model():
        return None

    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = cv2.FaceDetectorYN.create(
            _YUNET_MODEL_PATH,
            "",
            (width, height),
            score_threshold=0.5,
        )
    else:
        _FACE_DETECTOR.setInputSize((width, height))

    return _FACE_DETECTOR


# ---------------------------------------------------------------------------
# Temporal sampling
# ---------------------------------------------------------------------------

def sample_frames(video_path, n_frames=16):
    """
    Uniformly sample *n_frames* from a video file.

    Args:
        video_path: Path to the video file.
        n_frames:   Number of frames to sample.

    Returns:
        List of BGR numpy arrays, each of shape (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# C40 pre-filtering
# ---------------------------------------------------------------------------

def apply_c40_smoothing(frame, sigma=0.8):
    """
    Apply a Gaussian blur to suppress H.264 macroblock boundary artefacts
    that are prominent in c40 (heavy compression) clips.

    Args:
        frame: BGR numpy array.
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        Smoothed BGR numpy array.
    """
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)


# ---------------------------------------------------------------------------
# Face detection & cropping
# ---------------------------------------------------------------------------

def _centre_crop_face(frame, target_size):
    """
    Fallback: take a centre crop assuming the face is roughly centred
    (common in FaceForensics++ videos).
    """
    h, w = frame.shape[:2]
    crop_size = min(h, w)
    y1 = (h - crop_size) // 2
    x1 = (w - crop_size) // 2
    crop = frame[y1:y1 + crop_size, x1:x1 + crop_size]
    return cv2.resize(crop, target_size)


def detect_and_align_face(frame, target_size=(256, 256)):
    """
    Detect the primary face in *frame* using OpenCV FaceDetectorYN
    (YuNet model) and return a tightly-cropped, resized face ROI.

    Falls back to a centre crop when the detector is unavailable or
    no face is detected.

    Args:
        frame:       BGR numpy array.
        target_size: (width, height) tuple for the output crop.

    Returns:
        Face-cropped BGR numpy array of shape (target_size[1], target_size[0], 3).
    """
    h, w = frame.shape[:2]
    detector = _get_face_detector(w, h)

    if detector is None:
        return _centre_crop_face(frame, target_size)

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return _centre_crop_face(frame, target_size)

    # Pick the face with the highest confidence score (last column)
    best = faces[np.argmax(faces[:, -1])]
    x, y, fw, fh = best[:4].astype(int)

    # Add 20% padding
    pad_x = int(fw * 0.2)
    pad_y = int(fh * 0.2)

    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + fw + pad_x, w)
    y2 = min(y + fh + pad_y, h)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return _centre_crop_face(frame, target_size)

    return cv2.resize(face_crop, target_size)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_video(video_path, n_frames=16, target_size=(256, 256), sigma=0.8):
    """
    End-to-end preprocessing for a single video:
        1. Sample *n_frames* uniformly.
        2. Smooth each frame with Gaussian (c40 denoising).
        3. Detect & crop the face.

    Returns:
        List of preprocessed BGR numpy arrays.
    """
    raw_frames = sample_frames(video_path, n_frames)
    processed = []
    for frame in raw_frames:
        smoothed = apply_c40_smoothing(frame, sigma)
        aligned = detect_and_align_face(smoothed, target_size)
        processed.append(aligned)
    return processed
