import numpy as np


class Function:
    def __call__(self, x):
        raise NotImplementedError("Function.__call__() not implemented.")

    def derivative(self, x):
        raise NotImplementedError("Function.derivative() not implemented.")
    
    def __str__(self) -> str:
        raise NotImplementedError("Function.__str__() not implemented.")


class Sigmoid(Function):
    def derivative(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def __str__(self) -> str:
        return "Sigmoid"


class ReLU(Function):
    def derivative(self, x):
        return np.where(x <= 0, 0, 1)

    def __call__(self, x):
        return np.maximum(0, x)

    def __str__(self) -> str:
        return "ReLU"


class Tanh(Function):
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def __call__(self, x):
        return np.tanh(x)

    def __str__(self) -> str:
        return "Tanh"


class Identity(Function):
    def derivative(self, x):
        return 1

    def __call__(self, x):
        return x

    def __str__(self) -> str:
        return "Identity"
