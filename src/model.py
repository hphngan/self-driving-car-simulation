"""The self-driving car CNN (Nvidia architecture, Figure 7 of the spec).

This is the Nvidia "End to End Learning for Self-Driving Cars" network: five
convolutional layers followed by four fully-connected layers, ending in a single
linear neuron that predicts the steering angle.

Because predicting a steering angle is a **regression** problem, the network uses
a linear output and is trained with mean-squared-error loss.
"""

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
