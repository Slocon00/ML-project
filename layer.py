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
            regularizer: Regularizer = None,
            momentum: tuple[str, float] = None
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

        # Momentum
        self.momentum = None
        self.alpha = 0
        self.delta_old = 0
        if momentum is not None:
            self.momentum = momentum[0]
            self.alpha = momentum[1]
            if self.momentum not in ['Nesterov', 'Standard']:
                raise ValueError("Momentum must be either 'Nesterov' or 'Standard'.")

        # Initialize weights and biases
        self.setup(input_size, units_size)

    def setup(self, input_size: int, units_size: int):
        """Setup the weights and biases of the layer."""

        self.W = self.starting(input_size, units_size)
        self.b = np.random.uniform(low=-0.5, high=0.5, size=(units_size, 1))

    def forward(self, inputs: np.ndarray):
        """Forward the output of the layer units."""
        self.inputs = inputs

        self.net = self.W.T.dot(inputs) + self.b
        self.out = self.activation(self.net)
        return self.out

    def backward(self, curr_delta: np.ndarray, eta: float = 1e-1):
        """Backpropagate the error through the layer and update the weights of
        the layer's units.
        """
        delta = curr_delta * self.activation.derivative(self.net)
        delta_grad = self.inputs.dot(delta.T)

        alpha = self.alpha
        W = self.W.copy()  # We want to regularize only on the original weights
        if self.momentum == 'Nesterov':
            self.W += alpha * self.delta_old
            alpha = 0

        delta_prop = self.W.dot(delta)

        # Update weights
        if self.regularizer is not None:
            self.W -= eta * delta_grad - alpha * self.delta_old + self.regularizer.derivative(W)
            self.delta_old = - (eta * delta_grad)
        else:
            self.W -= eta * delta_grad - alpha * self.delta_old
            self.delta_old = - (eta * delta_grad)

        # Update bias
        self.b -= eta * delta
        return delta_prop

    def __str__(self) -> str:
        """Return a string description of the layer."""
        return (
            f"Hidden layer of {self.units_size} units."
            f"\n\nInput size: {self.input_size}"
            f"\nStarting: {self.starting}"
            f"\nActivation: {self.activation}"
            f"\nRegularizer: {self.regularizer}"
            f"\nMomentum: {self.momentum} with alpha {self.alpha}"
            f"\n\nW = \n{self.W}\n\nb = \n{self.b})"
        )
