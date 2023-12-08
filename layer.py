import numpy as np

class HiddenLayer:
    def __init__(self, num_inputs, num_units, activation):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.activation = activation

        self.setup(num_inputs, num_units)

    def setup(self, num_inputs, num_units):
        # Create a weight matrix of appropriate shape
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(num_inputs, num_units)) # the order is reversed because we want the shape to be (n_inputs, n_neurons)
        # Create a bias vector of appropriate shape
        self.b =  np.zeros((1, num_units)) # the first parameter of zeros() is the shape of the array, in this case it is a 1D array with n_neurons elements