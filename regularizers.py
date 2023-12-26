import numpy as np


class Regularizer:
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def __call__(self, weights: np.ndarray):
        raise NotImplementedError("Regularizer.__call__() not implemented.")

    def derivative(self, weights: np.ndarray):
        raise NotImplementedError("Regularizer.derivative() not implemented.")


class L1(Regularizer):
    def __call__(self, weights: np.ndarray):
        return self.lambda_ * np.sum(np.abs(weights))

    def derivative(self, weights: np.ndarray):
        return self.lambda_ * np.sign(weights)


class L2(Regularizer):
    def __call__(self, weights: np.ndarray):
        return self.lambda_ * np.sum(weights ** 2)

    def derivative(self, weights: np.ndarray):
        return 2 * self.lambda_ * weights
