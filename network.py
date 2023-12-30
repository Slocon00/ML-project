import numpy as np
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

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            epochs: int,
            eta: float = 0.001
    ):
        """Train the neural network on the provided training data."""
        losses = []
        accuracies =[]

        for epoch in range(epochs):
            epoch_loss = 0

            for X, y in zip(X_train, y_train):
                out = self.forward(inputs=X)

                loss = self.loss.forward(y_pred=out, y_true=y)
                epoch_loss += loss
                self.backward(
                    self.loss.backward(y_pred=out, y_true=y),
                    eta=eta
                )

            y_pred = []
            for X in X_test:
                out = self.forward(inputs=X)
                y_pred.append(out)
            accuracies.append(self.accuracy(y_pred=np.array(y_pred), y_true=y_test))

            losses.append(epoch_loss / len(X_train))

        return losses, accuracies

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
