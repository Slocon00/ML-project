import numpy as np

class HiddenLayer:
    def __init__(self, inputs, num_units, activation):

        self.inputs = inputs
        self.num_inputs = inputs.shape

        self.num_units = num_units

        self.activation = activation
        # Initialize weights and biases
        # TODO: initial initialization of weights and biases is currently random
        self.setup(inputs, num_units)
        
        self.net = np.dot(inputs, self.W) + self.b
        self.o = self.activation(self.net)

    def setup(self, num_inputs, num_units):
        # Create a weight matrix of appropriate shape
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(num_inputs, num_units)) # the order is reversed because we want the shape to be (n_inputs, n_neurons)
        # Create a bias vector of appropriate shape
        self.b =  np.random.uniform(low=-0.5, high=0.5, size=(1, num_units)) # the first parameter of zeros() is the shape of the array, in this case it is a 1D array with n_neurons elements

    def forward(self):
        return self.o
    
    def __str__(self) -> str:
        return f"HiddenLayer(num_units={self.num_units}, num_inputs={self.num_inputs}, activation={self.activation.__name__})\nW={self.W}\nb={self.b})"