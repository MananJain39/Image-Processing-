# Image Processing Pipeline

A Python-based image processing pipeline that demonstrates classical noise removal and image sharpening techniques using OpenCV, SciPy, and Matplotlib.

---

## Overview

This project takes a colour photograph, converts it to grayscale, applies salt and pepper noise, and then demonstrates the effect of five different spatial filters. The best result (Adaptive Median) is then sharpened using a Laplacian filter. PSNR values are printed for every stage so results can be compared objectively.

---

## Pipeline Steps

1. Load a colour photograph
2. Convert to grayscale
3. Add 25% salt and pepper noise
4. Apply the following filters to the noisy image:
   - Max Filter
   - Min Filter
   - Mean Filter
   - Median Filter
   - Adaptive Median Filter
5. Sharpen the Adaptive Median result using a Laplacian filter

---

## Requirements

Python 3.7 or above is required. Install dependencies with:

```bash
pip install numpy opencv-python matplotlib scipy
```

---

## Usage

1. Place your image in the same folder as the script.
2. Open `image_processing.py` and update the following line with your image filename:

```python
color_img = cv2.imread('your_photo.jpg')
```

1. Run the script:

```bash
python image_processing.py
```

---

## Output

The script produces four PNG figures saved to the working directory, and also displays them in interactive windows if run locally.

| File | Contents |
|---|---|
| `pipeline_all_steps.png` | All nine stages shown in a 3x3 grid |
| `filter_comparison.png` | Side-by-side comparison of all five filters with PSNR values |
| `laplacian_sharpening.png` | AMF input, Laplacian edge map, and sharpened result |
| `histograms.png` | Pixel intensity histograms for every stage |

A PSNR summary table is also printed to the console.

---

## Filter Reference

| Filter | Behaviour | Strength |
|---|---|---|
| Max | Replaces each pixel with the neighbourhood maximum | Removes pepper noise, distorts image |
| Min | Replaces each pixel with the neighbourhood minimum | Removes salt noise, distorts image |
| Mean | Averages all pixels in the neighbourhood | Smooths noise but introduces blur |
| Median | Picks the middle value of sorted neighbourhood pixels | Good edge preservation, effective on salt and pepper |
| Adaptive Median | Dynamically grows the window; skips pixels that are not noise | Best detail preservation, highest PSNR |

---

## Performance

Typical PSNR results on a standard test image with 25% salt and pepper noise:

| Stage | PSNR |
|---|---|
| Noisy input | ~11 dB |
| Max Filter | ~9 dB |
| Min Filter | ~5 dB |
| Mean Filter | ~19 dB |
| Median Filter | ~28 dB |
| Adaptive Median Filter | ~35 dB |

Higher PSNR indicates less distortion relative to the original clean image.

Note: The Adaptive Median Filter is computationally slower than the others due to its pixel-by-pixel adaptive window logic. On large images, processing time may be noticeable.

---

## Project Structure

```
image_processing.py   - Main script containing all functions and figures
README.md             - Project documentation
your_photo.jpg        - Your input image (user supplied)
```

---

## Notes

- All filter kernels use a default size of 3x3. This can be adjusted via the `size` parameter in each filter function.
- The Adaptive Median Filter uses a maximum window size of 7x7, configurable via the `s_max` parameter.
- The Laplacian sharpening strength is controlled by the `alpha` parameter (default 1.0). Increasing it produces stronger edge enhancement.
- If running in Jupyter Notebook, `plt.show()` renders figures inline automatically.
