import numpy as np

from layer import HiddenLayer
from losses import Loss


class Network:
    def __init__(self, loss: Loss):
        """Initialize the network.
        The network is initialized with an empty list of layers and a loss function.
        """

        self.layers = []
        self.loss = loss

    def add_layer(self, layer: HiddenLayer):
        """Add a layer to the network."""
        if self.get_layer_count() > 0:
            self.check_layers_shape(self.layers[-1].units_size, layer.input_size)
        self.layers.append(layer)

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs

        o = self.layers[0].forward(inputs)

        """Forward the inputs through the network."""
        for layer in self.layers[1:]:
            o = layer.forward(o)

    def backward(self):
        """Backpropagate the error through the network."""
        pass


    def check_layers_shape(self, units_size: int, input_size: int):
        if units_size != input_size:
            raise ValueError(
                f"Error: vectors must have same shape. The shapes are units_size: {units_size} and input_size: {input_size}"
            )

    def get_layer_count(self):
        return len(self.layers)

    def __str__(self) -> str:
        """Print the network."""
        return f"Network: of {self.n_layers} layers. Loss: {self.loss}"
