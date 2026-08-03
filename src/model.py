"""The self-driving car CNN (Nvidia architecture, Figure 7 of the spec).

This is the Nvidia "End to End Learning for Self-Driving Cars" network: five
convolutional layers followed by four fully-connected layers, ending in a single
linear neuron that predicts the steering angle.

Because predicting a steering angle is a **regression** problem, the network uses
a linear output and is trained with mean-squared-error loss.
"""

# --- Previous implementation (kept for reference) ---
# from __future__ import annotations
#
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
#
# # Preprocessed input size: height x width x channels (matches preprocessing.py).
# INPUT_SHAPE = (66, 200, 3)
#
#
# def build_model(learning_rate: float = 1e-3, dropout: float = 0.5) -> Sequential:
#     """Build and compile the Nvidia steering-prediction model."""
#     model = Sequential(name="nvidia_self_driving")
#
#     model.add(Input(shape=INPUT_SHAPE))
#
#     # Convolutional feature extractor (ELU activations, as in the Nvidia paper).
#     model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
#     model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
#     model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
#     model.add(Conv2D(64, (3, 3), activation="elu"))
#     model.add(Conv2D(64, (3, 3), activation="elu"))
#
#     model.add(Flatten())
#     model.add(Dropout(dropout))
#
#     # Fully connected layers.
#     model.add(Dense(100, activation="elu"))
#     model.add(Dense(50, activation="elu"))
#     model.add(Dense(10, activation="elu"))
#
#     # Single linear neuron -> continuous steering angle (regression).
#     model.add(Dense(1))
#
#     model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
#     return model

from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam


def build_model(learning_rate=1e-3, dropout=0.5):
    # MODEL
    model = Sequential([
        layers.Input(shape=(66, 200, 3)),
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.Flatten(),
        layers.Dropout(dropout),
        layers.Dense(100, activation='elu'),
        layers.Dense(50, activation='elu'),
        layers.Dense(10, activation='elu'),
        layers.Dense(1),
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    build_model().summary()
