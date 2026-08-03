"""Data augmentation for the self-driving car dataset (section 4 of the spec).

To help the model generalize, we randomly apply the following transforms to a
*portion* of the images:

  * Flipping        -- horizontal mirror; the steering angle is reversed (x -1).
  * Brightness      -- random brightening / darkening.
  * Zooming         -- small *centered* scale-in crop.
  * Panning         -- random translation; horizontal shift also adjusts steering.
  * Rotation        -- small *centered* tilt of a few degrees.

Every transform is applied independently with a configurable probability, so any
given image may receive none, some, or all of them.

IMPORTANT: transforms that move the road sideways MUST adjust the steering label
to match, otherwise the model learns to associate a shifted road with the wrong
angle and ends up *under-steering* (predicting angles that are too small, so the
car drifts to the outside of curves and leaves the lane). That is why:
  * ``flip`` negates the steering angle,
  * ``pan`` shifts the steering angle in proportion to the horizontal shift,
  * ``zoom`` crops around the image center so the road does not move sideways,
  * ``rotate`` turns the frame about its center, so the road stays centered.
Rotation is deliberately limited to a few degrees and given a lower probability
than the other transforms: the simulator's horizon is always level, so a large
tilt is a frame the car never actually sees and would only inject label noise.

IMPORTANT: augmentation must only be applied to the **training** set, never to
the validation/test set.

All functions operate on RGB ``uint8`` images (as returned by :func:`load_image`).
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV's internal thread pool can deadlock when these functions are called
# from Keras' generator worker thread on macOS. Disabling it keeps batching safe.
cv2.setNumThreads(0)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as an RGB ``uint8`` array."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def flip(image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
    """Mirror the image horizontally and negate the steering angle."""
    return cv2.flip(image, 1), -float(steering)


def adjust_brightness(
    image: np.ndarray,
    low: float = 0.4,
    high: float = 1.3,
) -> np.ndarray:
    """Scale image brightness by a random factor in ``[low, high]``."""
    factor = np.random.uniform(low, high)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def zoom(
    image: np.ndarray,
    low: float = 1.0,
    high: float = 1.2,
) -> np.ndarray:
    """Zoom into the image by a random factor in ``[low, high]``.

    The crop is taken around the image **center** so the road does not move
    left/right; that keeps the original steering angle valid.
    """
    h, w = image.shape[:2]
    scale = np.random.uniform(low, high)
    # Size of the region to crop before scaling back up to the full frame.
    crop_h, crop_w = int(h / scale), int(w / scale)
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def pan(
    image: np.ndarray,
    steering: float,
    max_shift: float = 0.08,
    steer_per_pixel: float = 0.003,
) -> tuple[np.ndarray, float]:
    """Translate the image and adjust the steering to match the horizontal shift.

    A horizontal shift of ``tx`` pixels makes the road appear to move sideways,
    which is equivalent to the car sitting off-center. We compensate the steering
    label by ``tx * steer_per_pixel`` so the (image, angle) pair stays consistent
    (this is the same idea as the left/right-camera correction). The vertical
    shift does not affect steering.
    """
    h, w = image.shape[:2]
    tx = np.random.uniform(-max_shift, max_shift) * w
    ty = np.random.uniform(-max_shift, max_shift) * h
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    steering = float(np.clip(float(steering) + tx * steer_per_pixel, -1.0, 1.0))
    return shifted, steering


def rotate(
    image: np.ndarray,
    max_angle: float = 5.0,
) -> np.ndarray:
    """Rotate the image by a random angle in ``[-max_angle, max_angle]`` degrees.

    The rotation is about the image **center**, so the road keeps its left/right
    position and the original steering angle stays valid. ``max_angle`` is small
    on purpose -- see the module docstring.
    """
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_augment(
    image: np.ndarray,
    steering: float,
    prob: float = 0.4,
    rotate_prob: float = 0.15,
) -> tuple[np.ndarray, float]:
    """Randomly apply the augmentation transforms to a single sample.

    Each transform is applied independently with probability ``prob`` so only a
    portion of the data is augmented, and augmented images vary in which
    transforms they received. Every transform that moves the road sideways
    (``pan``, ``flip``) also updates the steering label so the model still learns
    the correct steering magnitude.

    Parameters
    ----------
    image:
        RGB ``uint8`` image.
    steering:
        The steering label associated with the image.
    prob:
        Per-transform probability of being applied.
    rotate_prob:
        Probability of applying ``rotate``; kept lower than ``prob`` because the
        simulator's horizon is always level.

    Returns
    -------
    (image, steering)
        The (possibly) transformed image and its (updated) steering.
    """
    steering = float(steering)

    if np.random.random() < prob:
        image, steering = pan(image, steering)
    if np.random.random() < prob:
        image = zoom(image)
    if np.random.random() < rotate_prob:
        image = rotate(image)
    if np.random.random() < prob:
        image = adjust_brightness(image)
    if np.random.random() < prob:
        image, steering = flip(image, steering)

    return image, steering


if __name__ == "__main__":
    # Demo: build a preview grid of original vs. augmented images.
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    balanced_csv = os.path.join(project_root, "outputs", "driving_log_balanced.csv")
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    np.random.seed(0)
    df = pd.read_csv(balanced_csv)
    sample = df.sample(n=5, random_state=0).reset_index(drop=True)

    fig, axes = plt.subplots(len(sample), 2, figsize=(8, 2.4 * len(sample)))
    for i, row in sample.iterrows():
        original = load_image(row["center"])
        augmented, new_steer = random_augment(original.copy(), row["steering"])

        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"original (steer={row['steering']:.3f})", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(augmented)
        axes[i, 1].set_title(f"augmented (steer={new_steer:.3f})", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    preview_path = os.path.join(outputs_dir, "augmentation_preview.png")
    plt.savefig(preview_path, dpi=120)
    plt.close()
    print(f"Saved augmentation preview -> {preview_path}")
