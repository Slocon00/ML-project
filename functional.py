import numpy as np


class Function:
    def __call__(self, x):
        raise NotImplementedError("Function.__call__() not implemented.")

    def df(self, x):
        raise NotImplementedError("Function.df() not implemented.")


class Sigmoid(Function):
    def df(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def __str__(self) -> str:
        return "Sigmoid"


class ReLU(Function):
    def df(self, x):
        return np.where(x <= 0, 0, 1)

    def __call__(self, x):
        return np.maximum(0, x)

    def __str__(self) -> str:
        return "ReLU"


class Tanh(Function):
    def df(self, x):
        return 1 - np.tanh(x) ** 2

    def __call__(self, x):
        return np.tanh(x)

    def __str__(self) -> str:
        return "Tanh"


class Softmax(Function):
    def df(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def __str__(self) -> str:
        return "Softmax"
