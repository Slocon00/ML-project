import numpy as np

from network import Network
from regularizers import *
from losses import *
from utils import *
from activations import *




def create_net(seed:int,
               loss: Loss,
               input_size: int,
               num_layer:int,
               layers_size: list[int],
               starting: list[Starting_values],
               activations: list[Function],
               regularizers: list[Regularizer],
               momentums: list[tuple[str, float]],
               eta: float,
               ) -> Network:
    """Create a network with the specified parameters."""
    np.random.seed(seed)
    net = Network(loss)

    layers_size.insert(0,input_size) # this way we don't have to check if we are in the first hidden layer

    for i in range(num_layer):
        net.add_layer(
            input_size=layers_size[i],
            units_size=layers_size[i+1],
            starting=starting[i],
            regularizer=regularizers[i],
            activation=activations[i],
            momentum=momentums[i]
        )

    net.set_eta(eta) # we could set the seed here

    return net


def create_all_net(seed:int,
               loss: str,
               batch_size: int,
               input_size: int,
               num_layer:int,
               layers_size: list[int],
               starting: list[str],
               starting_range: list[tuple[float, float]],
               activations: list[str],
               regularizers: list[str],
               regularizers_lambda: list[float],
               momentums: list[tuple[str, float]],
               eta: float,
               ) -> Network:
    """Create a network with the specified parameters."""
    np.random.seed(seed)
    net = Network(loss)

    layers_size.insert(0,input_size) # this way we don't have to check if we are in the first hidden layer

    # convert string list into objects list

    starting = [eval(starting[i])(starting_range[i][0],starting_range[i][1]) for i in range(len(starting))]
    activations = [eval(activations[i])() for i in range(len(activations))]
    regularizers = [eval(regularizers[i])(lambda_=regularizers_lambda[i]) for i in range(len(regularizers))]

    for i in range(num_layer):
        net.add_layer(
            input_size=layers_size[i],
            units_size=layers_size[i+1],
            starting=starting[i],
            regularizer=regularizers[i],
            activation=activations[i],
            momentum=momentums[i]
        )

    net.set_eta(eta) # we could set the seed here

    #destroy all objects for safety
    del starting
    del activations
    del regularizers
    
    return net

    """
    np.random.seed(3)
    net = Network(MSE(1))

    net.add_layer(
        input_size=len(X_train[0]),
        units_size=5,
        starting=Range_random(),
        regularizer=L2(lambda_=10e-4),
        activation=ReLU(),
        momentum=('Standard',1e-2)
    )
    net.add_layer(
        input_size=5,
        units_size=1,
        starting=Range_random(),
        activation=Sigmoid(),
        momentum=('Standard',1e-2)
    )
    """ 