import numpy as np

class Starting_values:
    def __init__(self):
        pass

    def __call__(self, input_size: int, units_size: int):
        raise NotImplementedError("starting_values.__call__() not implemented.")

 
class Range_random(Starting_values):
    def __call__(self, input_size: int, units_size: int):
        return np.random.uniform(low=-0.5, high=0.5, size=(input_size, units_size))
    

class Fan_in(Starting_values):
    def __call__(self, input_size: int, units_size: int):
        return np.random.uniform(low=-0.5, high=0.5, size=(input_size, units_size)) * 2 / np.sqrt(input_size)


# Controllare slides NN_part_2 pag 12 per altri metodi di inizializzazione