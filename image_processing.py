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
