import numpy as np

def extract_shadow_features(image_linear, skin_mask, expected_shading):
    """
    Identifies shadows where observed brightness is significantly lower than expected,
    and extracts shadow-specific features.
    """
    observed_intensity = np.mean(image_linear, axis=2)
    valid_pixels = skin_mask > 0
    
    if not np.any(valid_pixels):
        return np.zeros(16, dtype=np.float32)
        
    obs = observed_intensity[valid_pixels]
    exp = expected_shading[valid_pixels]
    
    # Shadowed pixels: observed is much darker than expected
    diff = exp - obs
    is_shadow = diff > 0.2
    
    shadow_fraction = np.mean(is_shadow)
    
    if shadow_fraction < 1e-4:
        return np.zeros(16, dtype=np.float32)
        
    shadow_strength_mean = np.mean(diff[is_shadow])
    shadow_strength_max = np.max(diff[is_shadow])
    
    # Centroid of shadows
    y_indices, x_indices = np.where((expected_shading - observed_intensity > 0.2) & valid_pixels)
    if len(y_indices) > 0:
        cy = np.mean(y_indices) / image_linear.shape[0]
        cx = np.mean(x_indices) / image_linear.shape[1]
    else:
        cy = 0.5
        cx = 0.5
        
    features = np.zeros(16, dtype=np.float32)
    features[0] = shadow_fraction
    features[1] = shadow_strength_mean
    features[2] = shadow_strength_max
    features[3] = cx
    features[4] = cy
    
    return features
