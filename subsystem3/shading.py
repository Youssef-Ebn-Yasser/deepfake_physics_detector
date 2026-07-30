import numpy as np
import cv2

def compute_shading_consistency(image_linear, skin_mask, normals, L, I0, ambient):
    """
    Computes features based on the difference between expected and observed shading.
    """
    # Expected shading: I_exp = I0 * max(0, N dot L) + ambient
    dot_product = np.sum(normals * L, axis=2)
    expected_shading = I0 * np.clip(dot_product, 0, None) + ambient
    expected_shading = np.clip(expected_shading, 0, 1)
    
    # Observed intensity
    observed_intensity = np.mean(image_linear, axis=2)
    
    valid_pixels = skin_mask > 0
    if not np.any(valid_pixels):
        return np.zeros(16, dtype=np.float32)
        
    obs = observed_intensity[valid_pixels]
    exp = expected_shading[valid_pixels]
    
    diff = obs - exp
    
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    
    # Gradient differences inside skin mask
    gx_obs = cv2.Sobel(observed_intensity.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_obs = cv2.Sobel(observed_intensity.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_obs_mag = np.sqrt(gx_obs**2 + gy_obs**2)
    
    gx_exp = cv2.Sobel(expected_shading.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_exp = cv2.Sobel(expected_shading.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_exp_mag = np.sqrt(gx_exp**2 + gy_exp**2)
    
    grad_diff_mse = np.mean((grad_obs_mag[valid_pixels] - grad_exp_mag[valid_pixels])**2)
    
    # Build 16D feature vector
    hist, _ = np.histogram(diff, bins=10, range=(-1.0, 1.0), density=True)
    
    features = np.zeros(16, dtype=np.float32)
    features[0] = mse
    features[1] = mae
    features[2] = grad_diff_mse
    features[3:13] = hist
    # 13, 14, 15 are left as 0
    
    return features
