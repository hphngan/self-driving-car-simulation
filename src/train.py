"""Train the self-driving car CNN (section 7 of the spec).

Pipeline:
  1. Load + balance the dataset (normal_driving1 + recovery_driving).
  2. Split into training and validation sets.
  3. Train the Nvidia model using the batch generators (augmentation on the
     training set only; validation uses raw preprocessed frames).
  4. Plot the training/validation loss curves.
  5. Save the trained model for use in the simulator, together with a run record
     describing the data and settings it was trained on.

Usage
-----
    python src/train.py --epochs 15 --batch-size 100 \
        --steps-per-epoch 300 --data normal_driving1 recovery_driving

The saved model is compatible with TestSimulation.py (Keras HDF5 ``model.h5``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# IMPORTANT: import TensorFlow/Keras (via model) BEFORE any module that imports
# OpenCV (batching -> augmentation/preprocessing). On macOS, importing cv2 before
# TensorFlow triggers a thread-pool deadlock that hangs model.fit().
from model import build_model

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from batching import batch_generator
from data_utils import balance_steering_data, load_driving_log


def plot_history(history, save_path: str) -> None:
    """Plot training/validation loss curves and save them."""
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title("Training / validation loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved loss curve -> {save_path}")


def write_run_record(
    save_path: str,
    args: argparse.Namespace,
    data_dirs: list[str],
    raw_count: int,
    balanced_count: int,
    train_count: int,
    val_count: int,
    history,
) -> None:
    """Write the dataset and settings that produced the saved model.

    review_balance.py balances the data independently of this script, so its
    histograms and CSV can describe a different dataset than the model. This
    record is what ties a given model.h5 back to a specific dataset, bin count,
    samples-per-bin and seed.
    """
    metrics = history.history

    def last(name: str) -> str:
        return f"{metrics[name][-1]:.6f}" if name in metrics else "n/a"

    entries = [
        ("## Dataset", None),
        ("data", ", ".join(os.path.basename(os.path.normpath(d)) for d in data_dirs)),
        ("raw samples", raw_count),
        ("bins", args.bins),
        ("samples per bin", args.samples_per_bin),
        ("balanced samples", balanced_count),
        ("training samples", train_count),
        ("validation samples", val_count),
        ("seed", args.seed),
        ("## Hyperparameters", None),
        ("epochs", args.epochs),
        ("batch size", args.batch_size),
        ("steps per epoch", args.steps_per_epoch),
        ("validation steps", args.validation_steps),
        ("learning rate", args.learning_rate),
        ("dropout", args.dropout),
        ("## Final metrics", None),
        ("loss (mse)", last("loss")),
        ("mae", last("mae")),
        ("val_loss (mse)", last("val_loss")),
        ("val_mae", last("val_mae")),
        ("best val_loss", f"{min(metrics['val_loss']):.6f}"),
    ]

    width = max(len(label) for label, value in entries if value is not None) + 2
    lines = [
        "# Training run record",
        f"{'timestamp:':<{width}}{datetime.now().isoformat(timespec='seconds')}",
        f"{'command:':<{width}}{' '.join(sys.argv)}",
    ]
    for label, value in entries:
        if value is None:  # section heading
            lines += ["", label]
        else:
            lines.append(f"{label + ':':<{width}}{value}")

    with open(save_path, "w") as record:
        record.write("\n".join(lines) + "\n")
    print(f"Saved run record -> {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=["normal_driving1", "recovery_driving"])
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--samples-per-bin", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=None,
        help="Batches per validation pass. Defaults to one full pass over the "
        "validation set, i.e. ceil(validation_samples / batch_size).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument(
        "--model-path",
        default="model.h5",
        help="Where to save the trained model (HDF5). Defaults to the project-root "
        "model.h5 that TestSimulation.py loads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(project_root, path)

    data_dirs = [resolve(d) for d in args.data]
    outputs_dir = resolve(args.outputs)
    model_path = resolve(args.model_path)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 1 + 2. Load and balance.
    data = load_driving_log(data_dirs)
    balanced, removed = balance_steering_data(
        data, num_bins=args.bins, samples_per_bin=args.samples_per_bin, seed=args.seed
    )
    print(f"Loaded {len(data)} samples; balanced to {len(balanced)} "
          f"(removed {len(removed)}).")

    # 3. Train / validation split.
    x = balanced["center"].to_numpy()
    y = balanced["steering"].to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Training samples: {len(x_train)} | Validation samples: {len(x_val)}")

    if args.validation_steps is None:
        args.validation_steps = math.ceil(len(x_val) / args.batch_size)
        print(f"Validation steps: {args.validation_steps} (one full pass)")

    train_gen = batch_generator(x_train, y_train, args.batch_size, is_training=True, seed=args.seed)
    val_gen = batch_generator(x_val, y_val, args.batch_size, is_training=False, seed=args.seed)

    # 4. Build and train.
    model = build_model(learning_rate=args.learning_rate, dropout=args.dropout)
    model.summary()

    history = model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=args.validation_steps,
        verbose=1,
    )

    # 5. Plot and save.
    plot_history(history, os.path.join(outputs_dir, "training_loss.png"))
    model.save(model_path)
    print(f"Saved trained model -> {model_path}")
    write_run_record(
        os.path.join(outputs_dir, "run_record.txt"),
        args,
        data_dirs,
        raw_count=len(data),
        balanced_count=len(balanced),
        train_count=len(x_train),
        val_count=len(x_val),
        history=history,
    )
    print("Run the simulator test with:  python TestSimulation.py")


if __name__ == "__main__":
    main()
