import numpy as np


class HiddenLayer:
    """Class that represents a hidden layer of the network."""

    def __init__(self, input_size, units_size, activation):
        """Initialize the hidden layer with input_size inputs,
        units_size units, and the specified activation function.
        """
        self.input_size = input_size
        self.units_size = units_size

        self.activation = activation
        # Initialize weights and biases
        # TODO: initial initialization of weights and biases is currently random
        self.setup(input_size, units_size)

    def setup(self, input_size, units_size):
        """Setups the weights and biases of the layer."""

        # Create a weight matrix of appropriate shape; the order is reversed
        # because we want the shape to be (n_inputs, n_neurons)
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(input_size, units_size))

        # Create a bias vector of appropriate shape; the first parameter of zeros()
        # is the shape of the array, in this case it is a 1D array with n_neurons elements
        self.b = np.random.uniform(low=-0.5, high=0.5, size=(1, units_size))

    def forward(self, inputs: np.ndarray):
        """Forwards the output of the layer units."""
        self.net = inputs.dot(self.W) + self.b
        self.out = self.activation(self.net)
        return self.out

    # TODO: implement backpropagation

    def __str__(self) -> str:
        """Returns a string description of the layer."""
        return (
            f"HiddenLayer(input_size={self.input_size}, units_size={self.units_size}, "
            f"activation={self.activation})\nW={self.W}\nb={self.b})"
        )
    
    def getW(self) -> str:
        # print the weights as a matrix
        return self.W

    

