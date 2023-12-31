import numpy as np
from network import Network
from losses import Loss
from regularizers import Regularizer
from utils import Starting_values
from activations import *


def kfold_crossval(
        X: np.ndarray,
        y: np.ndarray,
        k: int,
        n_layers: int,
        n_units: list[int],
        output_size: int,
        starting: Starting_values,
        loss: Loss,
        regularizer: Regularizer,
        momentum: tuple[str, float],
        eta: float,
        epochs: int,
        seed: int = 0
):
    # TODO shuffle
    X_split = np.array_split(X, k)
    y_split = np.array_split(y, k)

    for i in range(k):
        X_train = np.concatenate(X_split[:i] + (X_split[i + 1:]))
        y_train = np.concatenate(y_split[:i] + (y_split[i + 1:]))

        X_val = X_split[i]
        y_val = y_split[i]

        np.random.seed(seed)

        # Reset the network
        net = Network(loss=loss)

        input_size = len(X[0])
        for j in range(n_layers - 1):
            net.add_layer(
                input_size=input_size,
                units_size=n_units[j],
                starting=starting,
                regularizer=regularizer,
                activation=ReLU(),
                momentum=momentum
            )
            input_size = n_units[j]

        # Output layer
        net.add_layer(
            input_size=input_size,
            units_size=output_size,
            starting=starting,
            regularizer=regularizer,
            activation=Sigmoid(),
            momentum=momentum
        )

        # TODO train model
