"""Image preprocessing for the self-driving car model (section 5 of the spec).

The preprocessing pipeline mirrors the one used at inference time in
``TestSimulation.py`` so that the images the model sees during training match the
images it receives from the simulator:

  1. Crop the road area from the full frame (remove sky/hood).
  2. Convert RGB -> YUV (the color space used by the Nvidia model).
  3. Apply a Gaussian blur to reduce noise.
  4. Resize to 200 x 66 (the Nvidia model input size).
  5. Normalize pixel values to the [0, 1] range.

All functions expect RGB ``uint8`` images (as produced by
``augmentation.load_image``) and return a float32 image ready for the network.
"""

from __future__ import annotations

import cv2
import numpy as np

# Avoid OpenCV thread-pool deadlocks when preprocessing runs inside the Keras
# batch-generator worker thread (see augmentation.py for details).
cv2.setNumThreads(0)

# Vertical crop bounds (rows) matching TestSimulation.py: img[60:135, :, :].
CROP_TOP = 60
CROP_BOTTOM = 135

# Nvidia model input size (width x height).
TARGET_WIDTH = 200
TARGET_HEIGHT = 66


def preprocess(image: np.ndarray) -> np.ndarray:
    """Run the full preprocessing pipeline on a single RGB image.

    Parameters
    ----------
    image:
        RGB ``uint8`` image (e.g. a 160x320x3 simulator frame).

    Returns
    -------
    numpy.ndarray
        A ``float32`` image of shape ``(66, 200, 3)`` with values in ``[0, 1]``.
    """
    image = image[CROP_TOP:CROP_BOTTOM, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    # image = image.astype(np.float32) / 255.0
    image = image / 255
    return image


if __name__ == "__main__":
    # Demo: visualize each preprocessing stage for a few sample images.
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from augmentation import load_image

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    balanced_csv = os.path.join(project_root, "outputs", "driving_log_balanced.csv")
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    df = pd.read_csv(balanced_csv)
    sample = df.sample(n=4, random_state=1).reset_index(drop=True)

    stages = ["original", "cropped", "YUV", "blurred", "resized (200x66)"]
    fig, axes = plt.subplots(len(sample), len(stages), figsize=(3 * len(stages), 2.2 * len(sample)))

    for i, row in sample.iterrows():
        original = load_image(row["center"])
        cropped = original[CROP_TOP:CROP_BOTTOM, :, :]
        yuv = cv2.cvtColor(cropped, cv2.COLOR_RGB2YUV)
        blurred = cv2.GaussianBlur(yuv, (3, 3), 0)
        resized = cv2.resize(blurred, (TARGET_WIDTH, TARGET_HEIGHT))

        for ax, img, title in zip(
            axes[i], [original, cropped, yuv, blurred, resized], stages
        ):
            ax.imshow(img)
            if i == 0:
                ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    preview_path = os.path.join(outputs_dir, "preprocessing_preview.png")
    plt.savefig(preview_path, dpi=120)
    plt.close()
    print(f"Saved preprocessing preview -> {preview_path}")

    # Report the final tensor shape/range so training code knows what to expect.
    final = preprocess(load_image(sample.iloc[0]["center"]))
    print(f"Preprocessed shape: {final.shape}, dtype: {final.dtype}, "
          f"min: {final.min():.3f}, max: {final.max():.3f}")
