import numpy as np


class HiddenLayer:
    """Class that represents a hidden layer of the network."""
    def __init__(self, num_inputs, num_units, activation):
        """Initialize the hidden layer with num_inputs inputs,
        num_units units, and the specified activation function.
        """
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.activation = activation
        # Initialize weights and biases
        # TODO: initial initialization of weights and biases is currently random
        self.setup(num_inputs, num_units)

    def setup(self, num_inputs, num_units):
        """Setups the weights and biases of the layer."""

        # Create a weight matrix of appropriate shape; the order is reversed
        # because we want the shape to be (n_inputs, n_neurons)
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(num_inputs, num_units))

        # Create a bias vector of appropriate shape; the first parameter of zeros()
        # is the shape of the array, in this case it is a 1D array with n_neurons elements
        self.b = np.random.uniform(low=-0.5, high=0.5, size=(1, num_units))

    def forward(self, inputs: np.ndarray):
        """Forwards the output of the layer units."""
        self.net = inputs.dot(self.W) + self.b
        self.o = self.activation(self.net)
        return self.o
    
    def __str__(self) -> str:
        """Returns a string description of the layer."""
        return (f"HiddenLayer(num_units={self.num_units}, num_inputs={self.num_inputs},"
                f"activation={self.activation.__name__})\nW={self.W}\nb={self.b})")
