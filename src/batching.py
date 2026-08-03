"""Batching the dataset (section 6 of the spec).

Loading every preprocessed image into memory at once is wasteful, so we use a
generator that yields one batch at a time. This generator is used by Keras'
``fit`` during training.

Behaviour:
  * Training batches: a random subset of samples is drawn each step and each
    image is randomly augmented (flip / brightness / zoom / pan / rotate) before
    preprocessing. Augmentation is applied **only** when ``is_training`` is True.
  * Validation batches: samples are walked in order (no random draws) and images
    are used as-is (no augmentation), only preprocessed, so the model is always
    evaluated on realistic frames. Walking in order means ``val_loss`` is the
    mean over the validation set rather than over a random resample of it.

Every image - augmented or not - goes through the same :func:`preprocess`
pipeline used at inference time.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from augmentation import load_image, random_augment
from preprocessing import preprocess


def batch_generator(
    image_paths: Sequence[str],
    steerings: Sequence[float],
    batch_size: int,
    is_training: bool,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(images, steerings)`` batches indefinitely.

    Parameters
    ----------
    image_paths:
        Sequence of center-image file paths.
    steerings:
        Matching sequence of steering angles.
    batch_size:
        Number of samples per yielded batch.
    is_training:
        If True, draw samples at random and apply random augmentation to each
        image. If False, walk the samples in order and only preprocess them
        (used for validation/testing).
    seed:
        Optional seed for reproducible sampling/augmentation. Unused when
        ``is_training`` is False, since those batches are deterministic.

    Yields
    ------
    (images, steerings)
        ``images`` has shape ``(batch_size, 66, 200, 3)`` (float32) and
        ``steerings`` has shape ``(batch_size,)`` (float32).
    """
    image_paths = np.asarray(image_paths)
    steerings = np.asarray(steerings, dtype=np.float32)
    n = len(image_paths)
    rng = np.random.default_rng(seed)
    cursor = 0  # position of the sequential (validation) pass

    while True:
        if is_training:
            indices = rng.integers(0, n, size=batch_size)
        else:
            # Walk the set in order and wrap around, so that ceil(n / batch_size)
            # consecutive batches cover every sample.
            indices = (cursor + np.arange(batch_size)) % n
            cursor = (cursor + batch_size) % n

        batch_images = []
        batch_steerings = []

        for index in indices:
            image = load_image(image_paths[index])
            steering = float(steerings[index])

            if is_training:
                image, steering = random_augment(image, steering)

            batch_images.append(preprocess(image))
            batch_steerings.append(steering)

        yield np.asarray(batch_images, dtype=np.float32), np.asarray(
            batch_steerings, dtype=np.float32
        )


if __name__ == "__main__":
    # Demo: pull a single batch from the balanced dataset and report its shape.
    import os

    import pandas as pd

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    balanced_csv = os.path.join(project_root, "outputs", "driving_log_balanced.csv")
    df = pd.read_csv(balanced_csv)

    train_gen = batch_generator(
        df["center"].tolist(), df["steering"].tolist(),
        batch_size=32, is_training=True, seed=0,
    )
    val_gen = batch_generator(
        df["center"].tolist(), df["steering"].tolist(),
        batch_size=32, is_training=False, seed=0,
    )

    x_train, y_train = next(train_gen)
    x_val, y_val = next(val_gen)
    print(f"Train batch: images {x_train.shape} ({x_train.dtype}), "
          f"steerings {y_train.shape}, range [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Val batch:   images {x_val.shape} ({x_val.dtype}), "
          f"steerings {y_val.shape}, range [{x_val.min():.3f}, {x_val.max():.3f}]")
