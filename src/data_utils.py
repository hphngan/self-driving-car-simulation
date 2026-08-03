"""Utilities for loading, reviewing and balancing the driving dataset.

The Udacity simulator writes a ``driving_log.csv`` (no header) with the columns:

    Center, Left, Right, Steering, Throttle, Brake, Speed

For this project we only need the *center* camera image path and the *steering*
angle. These helpers load one or more recording folders into a single
``DataFrame`` and balance the steering-angle distribution so the network is not
biased toward driving straight (steering == 0).
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DRIVING_LOG_COLUMNS = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "brake",
    "speed",
]


def _resolve_center_path(raw_path: str, data_dir: str) -> str:
    """Return an absolute path to the center image.

    The simulator stores absolute paths, but recordings are often moved between
    machines. To stay robust we always rebuild the path from ``data_dir`` and the
    image file name so the dataset keeps working after being relocated.
    """
    filename = os.path.basename(str(raw_path).strip().replace("\\", "/"))
    return os.path.join(data_dir, "IMG", filename)


def load_driving_log(data_dirs: str | Sequence[str]) -> pd.DataFrame:
    """Load and concatenate ``driving_log.csv`` from one or more folders.

    Parameters
    ----------
    data_dirs:
        A single recording folder or an iterable of folders. Each folder must
        contain a ``driving_log.csv`` file and an ``IMG`` sub-folder.

    Returns
    -------
    pandas.DataFrame
        A frame with the resolved ``center`` image path and a numeric
        ``steering`` column, plus the remaining raw columns.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    frames = []
    for data_dir in data_dirs:
        csv_path = os.path.join(data_dir, "driving_log.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"No driving_log.csv found in '{data_dir}'")

        frame = pd.read_csv(csv_path, names=DRIVING_LOG_COLUMNS, header=None)
        frame["center"] = frame["center"].apply(
            lambda p: _resolve_center_path(p, data_dir)
        )
        frame["steering"] = pd.to_numeric(frame["steering"], errors="coerce")
        frame["source"] = os.path.basename(os.path.normpath(data_dir))
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["steering"]).reset_index(drop=True)
    return data


def steering_histogram(
    steering: Iterable[float], num_bins: int = 25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a symmetric steering-angle histogram.

    Returns the per-bin sample counts, the bin edges and the bin centers. The
    bins are forced to be symmetric around zero so the straight-driving bin is
    centered on 0.
    """
    steering = np.asarray(list(steering), dtype=np.float64)
    hist, bin_edges = np.histogram(steering, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    return hist, bin_edges, bin_centers


def balance_steering_data(
    data: pd.DataFrame,
    num_bins: int = 25,
    samples_per_bin: int = 400,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, list[int]]:
    """Balance the dataset by capping the number of samples in each bin.

    The raw data is heavily dominated by ``steering == 0`` samples. For each
    histogram bin we keep at most ``samples_per_bin`` rows (chosen at random) and
    drop the rest, which flattens the distribution and prevents the model from
    learning to always drive straight.

    Parameters
    ----------
    data:
        Frame returned by :func:`load_driving_log`.
    num_bins:
        Number of histogram bins (odd numbers keep a single bin centered on 0).
    samples_per_bin:
        Maximum number of samples to keep per bin.
    seed:
        Seed for reproducible random removal.

    Returns
    -------
    (balanced_data, removed_indices)
        The balanced frame (index reset) and the list of original row indices
        that were removed.
    """
    np.random.seed(seed)
    steering = data["steering"].to_numpy()
    _, bin_edges = np.histogram(steering, bins=num_bins)

    remove_indices: list[int] = []
    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include the right edge only in the last bin to mirror np.histogram.
        if i == num_bins - 1:
            in_bin = np.where((steering >= low) & (steering <= high))[0]
        else:
            in_bin = np.where((steering >= low) & (steering < high))[0]

        if len(in_bin) > samples_per_bin:
            drop = np.random.permutation(in_bin)[samples_per_bin:]
            remove_indices.extend(drop.tolist())

    balanced = data.drop(index=remove_indices).reset_index(drop=True)
    return balanced, remove_indices
