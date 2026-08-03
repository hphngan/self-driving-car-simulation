"""Review and balance the collected driving dataset.

This script implements section 4 of the project ("Reviewing and Balancing the
Dataset"). It:

  1. Loads the ``driving_log.csv`` files from the recording folders.
  2. Plots a histogram of the steering angles to inspect the distribution.
  3. Balances the dataset by capping the number of samples per bin so the
     straight-driving (steering == 0) samples do not dominate training.
  4. Plots the balanced histogram and saves both figures plus the balanced CSV.

Usage
-----
    python src/review_balance.py \
        --data normal_driving1 recovery_driving \
        --bins 25 --samples-per-bin 400 \
        --outputs outputs
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: save figures to files instead of opening a window
import matplotlib.pyplot as plt

from data_utils import balance_steering_data, load_driving_log, steering_histogram


def plot_steering_histogram(
    steering,
    num_bins: int,
    samples_per_bin: int | None,
    title: str,
    save_path: str,
) -> None:
    """Plot and save a steering-angle histogram with a symmetric center line."""
    hist, _, bin_centers = steering_histogram(steering, num_bins=num_bins)
    width = 0.05 if len(bin_centers) < 2 else (bin_centers[1] - bin_centers[0]) * 0.9

    plt.figure(figsize=(9, 5))
    plt.bar(bin_centers, hist, width=width, color="#4C72B0", edgecolor="black")
    if samples_per_bin is not None:
        plt.plot(
            (bin_centers.min(), bin_centers.max()),
            (samples_per_bin, samples_per_bin),
            color="red",
            linestyle="--",
            label=f"samples per bin = {samples_per_bin}",
        )
        plt.legend()
    plt.axvline(0.0, color="gray", linewidth=1, alpha=0.6)
    plt.title(title)
    plt.xlabel("Steering angle")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved histogram -> {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs="+",
        default=["normal_driving1", "recovery_driving"],
        help="One or more recording folders (each with driving_log.csv + IMG/).",
    )
    parser.add_argument("--bins", type=int, default=25, help="Number of histogram bins.")
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=400,
        help="Maximum samples to keep per bin when balancing.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Directory to write histograms and the balanced CSV.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible balancing."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve data folders relative to the project root (parent of this file).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dirs = [
        d if os.path.isabs(d) else os.path.join(project_root, d) for d in args.data
    ]
    outputs_dir = (
        args.outputs
        if os.path.isabs(args.outputs)
        else os.path.join(project_root, args.outputs)
    )
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"Loading driving logs from: {', '.join(args.data)}")
    data = load_driving_log(data_dirs)
    print(f"Total samples loaded: {len(data)}")
    print(
        "Steering stats -> "
        f"min={data['steering'].min():.4f}, "
        f"max={data['steering'].max():.4f}, "
        f"mean={data['steering'].mean():.4f}"
    )
    zero_count = int((data["steering"] == 0).sum())
    print(f"Exactly-zero steering samples: {zero_count} "
          f"({100 * zero_count / len(data):.1f}% of data)")

    plot_steering_histogram(
        data["steering"],
        num_bins=args.bins,
        samples_per_bin=args.samples_per_bin,
        title="Steering angle distribution (before balancing)",
        save_path=os.path.join(outputs_dir, "steering_hist_before.png"),
    )

    balanced, removed = balance_steering_data(
        data,
        num_bins=args.bins,
        samples_per_bin=args.samples_per_bin,
        seed=args.seed,
    )
    print(f"Removed {len(removed)} over-represented samples.")
    print(f"Balanced samples remaining: {len(balanced)}")

    plot_steering_histogram(
        balanced["steering"],
        num_bins=args.bins,
        samples_per_bin=args.samples_per_bin,
        title="Steering angle distribution (after balancing)",
        save_path=os.path.join(outputs_dir, "steering_hist_after.png"),
    )

    balanced_csv = os.path.join(outputs_dir, "driving_log_balanced.csv")
    balanced[["center", "steering"]].to_csv(balanced_csv, index=False)
    print(f"Saved balanced dataset -> {balanced_csv}")


if __name__ == "__main__":
    main()
