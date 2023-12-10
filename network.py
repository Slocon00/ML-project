from layer import HiddenLayer


class Network:
    def __init__(self, loss):
        """Initialize the network.
        The network is initialized with an empty list of layers and a loss function.
        """

        self.layers = []
        self.loss = None

    def add_layer(self, num_inputs, num_units, activation):
        """Add a layer to the network."""
        self.layers.append(HiddenLayer(num_inputs, num_units, activation))

    def forward(self):
        """Forward the inputs through the network."""
        pass

    def backward(self):
        """Backpropagate the error through the network."""
        pass
