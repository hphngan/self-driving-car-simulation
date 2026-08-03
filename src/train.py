"""Train the self-driving car CNN (section 7 of the spec).

Pipeline:
  1. Load + balance the dataset (normal_driving1 + recovery_driving).
  2. Split into training and validation sets.
  3. Train the Nvidia model using the batch generators (augmentation on the
     training set only; validation uses raw preprocessed frames).
  4. Plot the training/validation loss curves.
  5. Save the trained model for use in the simulator.

Usage
-----
    python src/train.py --epochs 15 --batch-size 100 \
        --steps-per-epoch 300 --data normal_driving1 recovery_driving

The saved model is compatible with TestSimulation.py (Keras HDF5 ``model.h5``).
"""

from __future__ import annotations

import argparse
import os

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=["normal_driving1", "recovery_driving"])
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--samples-per-bin", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--validation-steps", type=int, default=200)
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
    print("Run the simulator test with:  python TestSimulation.py")


if __name__ == "__main__":
    main()
