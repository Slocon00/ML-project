import numpy as np
from activations import Function
from regularizers import Regularizer
from utils import Starting_values


class HiddenLayer:
    """Class that represents a hidden layer of the network. Each layer has a
    number of units, each with the same number of weights and inputs, and an
    activation function that is used to calculate the output.
    """

    def __init__(
            self,
            input_size: int,
            units_size: int,
            starting: Starting_values,
            activation: Function,
            regularizer: Regularizer = None
    ):
        """Initialize the hidden layer with input_size inputs,
        units_size units, and the specified activation function.
        """
        self.input_size = input_size
        self.units_size = units_size
        self.inputs = None
        self.net = None
        self.out = None
        self.W = None
        self.b = None
        self.starting = starting
        self.activation = activation
        self.regularizer = regularizer
        # Initialize weights and biases
        # TODO: initial initialization of weights and biases is currently random
        self.setup(input_size, units_size)

    def setup(self, input_size: int, units_size: int):
        """Setup the weights and biases of the layer."""

        self.W = self.starting(input_size, units_size)

        # Create a bias vector of appropriate shape; the first parameter of zeros()
        # is the shape of the array, in this case it is a 1D array with n_neurons elements
        self.b = np.random.uniform(low=-0.5, high=0.5, size=(units_size, 1))

    def forward(self, inputs: np.ndarray):
        """Forward the output of the layer units."""
        self.inputs = inputs

        self.net = self.W.T.dot(inputs) + self.b
        self.out = self.activation(self.net)
        return self.out

    def backward(self, curr_delta: np.ndarray, eta: float = 0.001):
        """Backpropagate the error through the layer and update the weights of
        the layer's units.
        """
        delta = curr_delta * self.activation.derivative(self.net)
        #delta_prop = delta.dot(self.W.T)
        delta_prop = self.W.dot(delta)

        if self.regularizer is not None:
            self.W -= eta * self.inputs.dot(delta.T) + self.regularizer.derivative(self.W)
        else:
            self.W -= eta * self.inputs.dot(delta.T)
        # TODO: regularization on the bias?
        self.b -= eta * np.sum(delta, axis=0, keepdims=True)
        return delta_prop

    def __str__(self) -> str:
        """Return a string description of the layer."""
        return (
            f"HiddenLayer(input_size={self.input_size}, units_size={self.units_size}, "
            f"activation={self.activation})\nW=\n{self.W}\nb=\n{self.b})"
        )
