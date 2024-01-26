import numpy as np
import matplotlib.pyplot as plt
from network import Network
from metrics import Metric
from regularizers import *
from losses import *
from utils import *
from activations import *


def kfold_crossval(
        X: np.ndarray,
        y: np.ndarray,
        k: int,
        net: Network,
        metric: Metric,
        epochs: int = 500,
        patience: int = 25,
        seed: int = None,
        scaler=None,
        verbose: bool = False,
):
    """Perform k-fold cross validation training the given network with the given
    parameters, and return the mean and std of loss and metric values found
    across all k folds.
    """

    # Saving parameters needed to reset the net for each fold
    loss = net.loss
    num_layer = len(net.layers)
    layers_size = []
    starting = []
    activations = []
    regularizers = []
    momentums = []
    for layer in net.layers:
        layers_size.append(layer.units_size)
        starting.append(layer.starting)
        activations.append(layer.activation)
        regularizers.append(layer.regularizer)
        momentums.append((layer.momentum, layer.alpha) if
                         layer.momentum is not None else None)
    eta = net.eta
    tau = net.tau
    cyclic = net.cyclic

    tr_losses = []
    tr_metrics = []
    val_losses = []
    val_metrics = []

    X_split = np.array_split(X, k)
    y_split = np.array_split(y, k)

    for i in range(k):
        X_train = np.concatenate(X_split[:i] + (X_split[i + 1:]))
        y_train = np.concatenate(y_split[:i] + (y_split[i + 1:]))

        X_val = X_split[i]
        y_val = y_split[i]

        # Shuffling the training set
        tr_indexes = np.arange(len(X_train))
        np.random.shuffle(tr_indexes)

        info = net.train(
            X_train[tr_indexes],
            y_train[tr_indexes],
            X_val,
            y_val,
            epochs=epochs,
            patience=patience,
            metric=metric,
        )

        tr_loss, tr_metric = net.statistics(X_train, y_train, metric)
        val_loss, val_metric = net.statistics(X_val, y_val, metric)

        tr_losses.append(tr_loss)
        tr_metrics.append(tr_metric)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        if verbose:
            show_plots(info)

            print(f"Fold {i + 1} of {k} completed")
            print(f"Train Loss: {tr_loss}")
            print(f"Train Metric: {tr_metric}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Metric: {val_metric}")

        # Reset the net
        net = create_net(
            seed=seed,
            loss=loss,
            input_size=len(X_train[0]),
            num_layer=num_layer,
            layers_size=layers_size,
            starting=starting,
            activations=activations,
            regularizers=regularizers,
            momentums=momentums,
            eta=eta,
            tau=tau,
            cyclic=cyclic,
        )

    statistics = {
        'tr_loss': np.mean(tr_losses),
        'tr_metric': np.mean(tr_metrics),
        'val_loss': np.mean(val_losses),
        'val_metric': np.mean(val_metrics),
        'tr_loss_std': np.std(tr_losses),
        'tr_metric_std': np.std(tr_metrics),
        'val_loss_std': np.std(val_losses),
        'val_metric_std': np.std(val_metrics),
    }

    return statistics


def create_net(
        seed: int,
        loss: Loss,
        input_size: int,
        num_layer: int,
        layers_size: list[int],
        starting: list[Starting_values],
        activations: list[Function],
        regularizers: list[Regularizer],
        momentums: list[tuple[str, float]],
        eta: float,
        tau: int = 1000,
        cyclic: bool = False,
) -> Network:
    """Create a network with the specified parameters (provided as objects)."""
    np.random.seed(seed)
    net = Network(loss, eta=eta, tau=tau, cyclic=cyclic)

    layers_size.insert(0, input_size)  # this way we don't have to check if we are in the first hidden layer

    for i in range(num_layer):
        net.add_layer(
            input_size=layers_size[i],
            units_size=layers_size[i + 1],
            starting=starting[i],
            regularizer=regularizers[i],
            activation=activations[i],
            momentum=momentums[i]
        )
    layers_size.pop(0)

    return net


def create_all_net(seed: int,
                   loss: str,
                   batch_size: int,
                   input_size: int,
                   num_layer: int,
                   layers_size: list[int],
                   starting: list[str],
                   starting_range: list[tuple[float, float]],
                   activations: list[str],
                   regularizers: list[str],
                   regularizers_lambda: list[float],
                   momentums: list[tuple[str, float]],
                   eta: float,
                   ) -> Network:
    """Create a network with the specified parameters (provided as strings)."""
    np.random.seed(seed)

    loss = eval(loss)(batch_size=batch_size)
    net = Network(loss, eta=eta)

    layers_size.insert(0, input_size)  # this way we don't have to check if we are in the first hidden layer

    # convert string list into objects list
    regularizers_ = []
    momentums_ = []

    # starting and activations cannot be None
    starting = [eval(starting[i])(starting_range[i][0], starting_range[i][1]) for i in range(len(starting))]
    activations = [eval(activations[i])() for i in range(len(activations))]

    # We could choose to not have momentum and regularizer for some layers
    for i, reg in enumerate(regularizers):
        if reg != "None":
            regularizers_.append(eval(reg)(lambda_=regularizers_lambda[i]))
        else:
            regularizers_.append(None)

    for m in momentums:
        if m[0] != "None":
            momentums_.append(m)
        else:
            momentums_.append(None)

    for i in range(num_layer):
        net.add_layer(
            input_size=layers_size[i],
            units_size=layers_size[i + 1],
            starting=starting[i],
            regularizer=regularizers_[i],
            activation=activations[i],
            momentum=momentums_[i]
        )

    layers_size.pop(0)

    # destroy all objects for safety
    del starting
    del activations
    del regularizers
    del momentums
    del regularizers_
    del momentums_

    return net


def show_plots(statistics):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(statistics['tr_losses'], label='Train Loss')
    plt.plot(statistics['val_losses'], label='Val Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(statistics['tr_metrics'], label='Train Metric')
    plt.plot(statistics['val_metrics'], label='Val Metric')
    plt.legend()

    plt.tight_layout()
    plt.show()
