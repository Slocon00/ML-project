from network import Network
from regularizers import *
from losses import *
from utils import *
from activations import *
import matplotlib.pyplot as plt


def kfold_crossval(
        X: np.ndarray,
        y: np.ndarray,
        k: int,
        net: Network,
        epochs: int = 1000,
        patience: int = 25,
        seed: int = None
):
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
        momentums.append((layer.momentum, layer.alpha))
    eta = net.eta

    tr_losses = []
    tr_accuracies = []
    val_losses = []
    val_accuracies = []

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
        )
        plt.plot(info['tr_losses'], label='Train Loss')
        plt.plot(info['val_losses'], label='Val Loss')
        plt.legend()
        plt.show()

        plt.plot(info['tr_accuracies'], label='Train Acc')
        plt.plot(info['val_accuracies'], label='Val Acc')
        plt.legend()
        plt.show()

        tr_loss, tr_acc = net.statistics(X_train, y_train)
        val_loss, val_acc = net.statistics(X_val, y_val)

        tr_losses.append(tr_loss)
        tr_accuracies.append(tr_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

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
            eta=eta
        )

    statistics = {
        'tr_loss': np.mean(tr_losses),
        'tr_accuracy': np.mean(tr_accuracies),
        'val_loss': np.mean(val_losses),
        'val_accuracy': np.mean(val_accuracies)
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
) -> Network:
    """Create a network with the specified parameters."""
    np.random.seed(seed)
    net = Network(loss)

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

    net.set_eta(eta)  # we could set the seed here

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
    """Create a network with the specified parameters."""
    np.random.seed(seed)
    net = Network(loss)

    layers_size.insert(0, input_size)  # this way we don't have to check if we are in the first hidden layer

    # convert string list into objects list

    starting = [eval(starting[i])(starting_range[i][0], starting_range[i][1]) for i in range(len(starting))]
    activations = [eval(activations[i])() for i in range(len(activations))]
    regularizers = [eval(regularizers[i])(lambda_=regularizers_lambda[i]) for i in range(len(regularizers))]

    for i in range(num_layer):
        net.add_layer(
            input_size=layers_size[i],
            units_size=layers_size[i + 1],
            starting=starting[i],
            regularizer=regularizers[i],
            activation=activations[i],
            momentum=momentums[i]
        )

    net.set_eta(eta)  # we could set the seed here

    # destroy all objects for safety
    del starting
    del activations
    del regularizers

    return net

'''# Reset the network
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
        )'''
