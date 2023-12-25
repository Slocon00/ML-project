import numpy as np

from layer import HiddenLayer
from losses import Loss


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

    def add_layer(self, layer: HiddenLayer):
        """Add a layer to the network."""
        if len(self.layers) > 0:
            self.check_layers_shape(self.layers[-1].units_size, layer.input_size)
        self.layers.append(layer)

    def forward(self, inputs: np.ndarray):
        """Calculate the output of the neural network by forwarding the inputs
        through the layers.
        """
        self.inputs = inputs

        o = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            o = layer.forward(o)

        return o

    def backward(self, curr_delta: np.ndarray, eta=0.1):
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

    def __str__(self) -> str:
        """Print the network."""
        return f"Network: {len(self.layers)} layers \nLoss: {self.loss}"
