import numpy as np
from tqdm import tqdm

from losses import Loss
from activations import Function
from layer import HiddenLayer
from regularizers import Regularizer
from utils import Starting_values


class Network:
    """Class that represents a neural network. It has a variable number of
    hidden layers, and a loss function used to calculate the error between the
    produced output and the expected output.
    """

    def __init__(self, loss: Loss):
        """Initialize the network.
        The network is initialized with an empty list of layers and a loss function.
        """

        self.layers = []
        self.loss = loss

        self.inputs = None
        self.eta = None

    def add_layer(
            self,
            input_size,
            units_size,
            starting: Starting_values,
            activation: Function,
            regularizer: Regularizer = None,
            momentum: tuple[str, float] = None
    ):
        """Add a layer with the specified parameters to the network."""
        if len(self.layers) > 0:
            self.check_layers_shape(self.layers[-1].units_size, input_size)

        self.layers.append(
            HiddenLayer(
                input_size=input_size,
                units_size=units_size,
                starting=starting,
                activation=activation,
                regularizer=regularizer,
                momentum=momentum
            )
        )

    def forward(self, inputs: np.ndarray):
        """Calculate the output of the neural network by forwarding the inputs
        through the layers.
        """
        self.inputs = inputs

        o = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            o = layer.forward(o)

        return o

    def backward(self, curr_delta: np.ndarray, eta: float = 10e-4):
        """Backpropagate the error through the network."""
        for layer in reversed(self.layers):
            delta_prop = layer.backward(curr_delta, eta)
            curr_delta = delta_prop

    def check_layers_shape(self, units_size: int, input_size: int):
        """Check whether the provided sizes are equal, raising an exception if not."""
        if units_size != input_size:
            raise ValueError(
                f"Error: vectors must have same shape."
                f"The shapes are units_size: {units_size} and input_size: {input_size}"
            )

    def set_eta(self, eta: float):
        """Set the learning rate of the network."""
        self.eta = eta
        
    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int,
            patience: int,
            eta: float = 0.001
    ):
        """Train the neural network on the provided training data. Losses and
        accuracies are calculated for each epoch (both for training and
        validation sets), and then returned as part of a single dictionary.
        """
        # TODO add early stopping
        tr_losses = []
        tr_accuracies = []
        val_losses = []
        val_accuracies = []

        with tqdm(total=epochs, desc="Epochs", colour='yellow') as pbar:
            lowest = np.inf  # Variable that stores the lowest loss recorded

            for epoch in range(epochs):
                # Training the network
                for X, y in zip(X_train, y_train):
                    out = self.forward(inputs=X)
                    self.backward(
                        self.loss.backward(y_pred=out, y_true=y),
                        eta=eta
                    )

                # Calculating loss and accuracy for the epoch
                # Training loss and acc
                tr_loss, tr_acc = self.statistics(X_train, y_train)
                tr_losses.append(tr_loss)
                tr_accuracies.append(tr_acc)

                # Validation loss and acc
                val_loss, val_acc = self.statistics(X_val, y_val)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                # Check early stopping condition
                #TODO: threshold early stopping?
                if val_loss < lowest:
                    lowest = val_loss
                    epochs_since_lowest = 0

                    # Record all the layers' weights and bias that correspond to
                    # the lowest validation loss
                    best_W = []
                    best_b = []
                    for layer in self.layers:
                        best_W.append(layer.W.copy())
                        best_b.append(layer.b.copy())
                else:
                    epochs_since_lowest += 1

                if epochs_since_lowest >= patience:
                    # Early stopping cond is true
                    # TODO force network with optimal W and b? Or return them?
                    break

                pbar.update(1)

        statistics = {
            'tr_losses': tr_losses,
            'tr_accuracies': tr_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

        return statistics

    def statistics(self, X, y):
        """Calculate loss and accuracy for the given input and output data."""
        pred = []
        for X in X:
            out = self.forward(X)
            pred.append(out)
        pred = np.array(pred)

        loss = self.loss.forward(y_pred=pred, y_true=y)
        accuracy = self.accuracy(y_pred=pred, y_true=y)

        return loss, accuracy


    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred = y_pred.reshape(len(y_pred), 1)
        y_true_reshaped = y_true.reshape(len(y_true), 1)

        # TODO adjust threshold depending on activation func
        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        # check the accuracy in percentage
        accuracy = np.sum(y_pred == y_true_reshaped) / len(y_true_reshaped) * 100
        return accuracy

    def __str__(self) -> str:
        """Print the network."""
        return f"Network: {len(self.layers)} layers \nLoss: {self.loss}"
