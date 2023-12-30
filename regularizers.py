import numpy as np


class Regularizer:
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def __call__(self, weights: np.ndarray):
        raise NotImplementedError("Regularizer.__call__() not implemented.")

    def __str__(self):
        raise NotImplementedError("Regularizer.__str__() not implemented.")

    def derivative(self, weights: np.ndarray):
        raise NotImplementedError("Regularizer.derivative() not implemented.")


class L1(Regularizer):
    def __call__(self, weights: np.ndarray):
        return self.lambda_ * np.sum(np.abs(weights))

    def __str__(self):
        return f"L1 with lambda {self.lambda_}"

    def derivative(self, weights: np.ndarray):
        return self.lambda_ * np.sign(weights)


class L2(Regularizer):
    def __call__(self, weights: np.ndarray):
        return self.lambda_ * np.sum(weights ** 2)

    def __str__(self):
        return f"L2 with lambda {self.lambda_}"

    def derivative(self, weights: np.ndarray):
        return 2 * self.lambda_ * weights


class SmoothL1(Regularizer):
    """
    Smooth approximation of L1 regularizer: for phi -> +infty, abs_phi converges to abs(weights).
    See https://www.cs.ubc.ca/~schmidtm/Documents/2007_ECML_L1General.pdf for more details.
    """
    def __call__(self, weights: np.ndarray, phi: float = 3):
        abs_phi = 1/phi * (np.log(1 + np.exp(-phi * weights)) + np.log(1 + np.exp(phi * weights)))
        return self.lambda_ * np.sum(abs_phi)

    def __str__(self):
        return f"Smooth-L1 with lambda {self.lambda_}"

    def derivative(self, weights: np.ndarray, phi: float = 3):
        return self.lambda_ * ((np.exp(phi * weights) - 1) / (np.exp(phi * weights) + 1))


class ElasticNet(Regularizer):
    """Convex combination of L1 and L2 regularizers."""
    def __call__(self, weights: np.ndarray, rho: float = 0.5):
        return self.lambda_ * ((1 - rho) * np.sum(weights ** 2) + rho * np.sum(np.abs(weights)))

    def __str__(self):
        return f"ElasticNet with lambda {self.lambda_}"

    def derivative(self, weights: np.ndarray, rho: float = 0.5):
        return self.lambda_ * (2 * (1 - rho) * weights + rho * np.sign(weights))