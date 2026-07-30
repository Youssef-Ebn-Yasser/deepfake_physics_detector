import numpy as np
import cv2

# Approximate eye region indices in our 6-point landmark set
# landmarks[2] = left eye, landmarks[3] = right eye
LEFT_EYE_IDX  = 2
RIGHT_EYE_IDX = 3

def extract_reflection_features(image_linear, landmarks):
    """
    Extracts specular highlights from approximate eye regions.
    landmarks: (N, 2) numpy array of [x, y] pixel coords (from PhysicsFrame.landmarks).
    """
    h, w = image_linear.shape[:2]
    eye_half = int(w * 0.07)  # rough eye patch half-size

    def get_eye_highlight(eye_idx):
        if eye_idx >= len(landmarks):
            return 0, 0, 0, 0
        cx, cy = int(landmarks[eye_idx, 0]), int(landmarks[eye_idx, 1])

        x1 = max(cx - eye_half, 0)
        y1 = max(cy - eye_half, 0)
        x2 = min(cx + eye_half, w)
        y2 = min(cy + eye_half, h)

        if x2 <= x1 or y2 <= y1:
            return 0, 0, 0, 0

        crop = image_linear[y1:y2, x1:x2]
        intensity = np.mean(crop, axis=2)

        thresh = np.percentile(intensity, 95)
        if thresh < 0.5:
            return 0, 0, 0, 0

        highlight_mask = intensity > thresh
        y_idx, x_idx = np.where(highlight_mask)

        if len(y_idx) > 0:
            cy_h = (np.mean(y_idx) + y1) / h
            cx_h = (np.mean(x_idx) + x1) / w
            area = len(y_idx) / (w * h)
            brightness = np.mean(intensity[highlight_mask])
            return cx_h, cy_h, area, brightness
        return 0, 0, 0, 0

    lx, ly, la, lb = get_eye_highlight(LEFT_EYE_IDX)
    rx, ry, ra, rb = get_eye_highlight(RIGHT_EYE_IDX)

    features = np.zeros(16, dtype=np.float32)
    features[0] = lx
    features[1] = ly
    features[2] = la
    features[3] = lb
    features[4] = rx
    features[5] = ry
    features[6] = ra
    features[7] = rb

    # Consistency: both eyes should have similar highlight Y position and area
    features[8]  = abs(ly - ry)
    features[9]  = abs(la - ra)
    features[10] = abs(lb - rb)

    return features

