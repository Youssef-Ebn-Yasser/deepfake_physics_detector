import cv2
import numpy as np

def extract_fft_spectrum(image_gray, num_bins=64):
    """
    Computes 1D azimuthal radial average of 2D FFT power spectrum.
    Input: Grayscale face crop (256x256)
    Output: 64-D feature vector (f5)
    """
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    h, w = magnitude_spectrum.shape
    center = (int(w / 2), int(h / 2))
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)

    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)

    feature_64 = np.interp(
        np.linspace(0, len(radial_profile) - 1, num_bins),
        np.arange(len(radial_profile)),
        radial_profile
    )
    return feature_64.astype(np.float32)
