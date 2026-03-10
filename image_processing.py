"""
Image Processing Pipeline
=========================
1. Load your own colour photograph
2. Convert to Grayscale
3. Add 25% Salt & Pepper Noise
4. Apply Max, Min, Mean, Median, Adaptive Median Filters
5. Sharpen using Laplacian Filter (on Adaptive Median result)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter, median_filter

# ─────────────────────────────────────────────
# LOAD YOUR OWN IMAGE — change path here
# ─────────────────────────────────────────────
color_img = cv2.imread(
    r"C:\Code Files\Project\Image-Processing-\trial image.jpg"
)  # ← put your image path here

if color_img is None:
    raise FileNotFoundError("Image not found! Check your file path.")


# ─────────────────────────────────────────────
# STEP 1 — Grayscale conversion
# ─────────────────────────────────────────────
def to_grayscale(color_img):
    return cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────
# STEP 2 — 25% Salt & Pepper Noise
# ─────────────────────────────────────────────
def add_salt_pepper(gray, noise_ratio=0.25):
    noisy = gray.copy()
    total = gray.size
    n_noise = int(total * noise_ratio)

    # Salt (white)
    salt_coords = [np.random.randint(0, d, n_noise // 2) for d in gray.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255

    # Pepper (black)
    pepper_coords = [np.random.randint(0, d, n_noise // 2) for d in gray.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    return noisy

# ─────────────────────────────────────────────
# STEP 3a — Max Filter
# ─────────────────────────────────────────────
def max_filter(img, size=3):
    return maximum_filter(img, size=size).astype(np.uint8)


# ─────────────────────────────────────────────
# STEP 3b — Min Filter
# ─────────────────────────────────────────────
def min_filter(img, size=3):
    return minimum_filter(img, size=size).astype(np.uint8)


# ─────────────────────────────────────────────
# STEP 3c — Mean Filter
# ─────────────────────────────────────────────
def mean_filter(img, size=3):
    return uniform_filter(img.astype(np.float32), size=size).astype(np.uint8)


# ─────────────────────────────────────────────
# STEP 3d — Median Filter
# ─────────────────────────────────────────────
def apply_median_filter(img, size=3):
    return median_filter(img, size=size).astype(np.uint8)

# ─────────────────────────────────────────────
# STEP 3e — Adaptive Median Filter
# ─────────────────────────────────────────────
def adaptive_median_filter(img, s_max=7):
    padded = np.pad(img, s_max // 2, mode="reflect")
    output = img.copy().astype(np.uint8)
    rows, cols = img.shape

    for r in range(rows):
        for c in range(cols):
            s = 3
            while s <= s_max:
                half = s // 2
                r0 = r + (s_max // 2) - half
                c0 = c + (s_max // 2) - half
                window = padded[r0 : r0 + s, c0 : c0 + s].flatten()

                z_min = int(window.min())
                z_max = int(window.max())
                z_med = int(np.median(window))
                z_xy = int(img[r, c])

                A1 = z_med - z_min
                A2 = z_med - z_max

                if A1 > 0 and A2 < 0:
                    B1 = z_xy - z_min
                    B2 = z_xy - z_max
                    output[r, c] = z_xy if (B1 > 0 and B2 < 0) else z_med
                    break
                else:
                    s += 2
                    if s > s_max:
                        output[r, c] = z_med
    return output

# ─────────────────────────────────────────────
# STEP 4 — Laplacian Sharpening
# ─────────────────────────────────────────────
def laplacian_sharpen(img, alpha=1.0):
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    sharpened = np.clip(img.astype(np.float64) + alpha * lap, 0, 255)
    return sharpened.astype(np.uint8)

# ─────────────────────────────────────────────
# PSNR helper
# ─────────────────────────────────────────────
def psnr(original, filtered):
    mse = np.mean((original.astype(np.float64) - filtered.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))
