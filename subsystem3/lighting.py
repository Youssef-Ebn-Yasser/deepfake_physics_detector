import numpy as np

def estimate_lighting(image_linear, skin_mask, normals):
    """
    Estimates a dominant light direction (L) and intensity (I0) 
    using a simple Lambertian assumption: I = I0 * max(0, N dot L) + ambient.
    We approximate this by solving a linear system on the skin pixels.
    """
    valid_pixels = skin_mask > 0
    
    if not np.any(valid_pixels):
        return np.array([0, 0, 1.0]), 1.0, 0.0
        
    N = normals[valid_pixels]  # (K, 3)
    
    # We use intensity for lighting estimation
    # Convert linear RGB to grayscale intensity
    intensity = np.mean(image_linear, axis=2)[valid_pixels]  # (K,)
    
    # Model: intensity = N dot (I0 * L) + ambient
    # Let A = [N, 1], x = [L_x * I0, L_y * I0, L_z * I0, ambient]^T
    # A * x = intensity
    
    ones = np.ones((N.shape[0], 1))
    A = np.hstack([N, ones])  # (K, 4)
    
    # Least squares solution
    x, residuals, rank, s = np.linalg.lstsq(A, intensity, rcond=None)
    
    # x[:3] = I0 * L
    ambient = x[3]
    I0_L = x[:3]
    I0 = np.linalg.norm(I0_L)
    
    if I0 > 1e-6:
        L = I0_L / I0
    else:
        L = np.array([0.0, 0.0, 1.0])
        I0 = 0.0
        
    return L, I0, ambient
