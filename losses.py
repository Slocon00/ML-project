import numpy as np


class Loss:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError("Loss.forward() not implemented.")

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError("Loss.backward() not implemented.")

    def check_shape(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Error: vectors must have same shape. The shapes are y_pred: {y_pred.shape} and y_true: {y_true.shape}"
            )


class MSE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.check_shape(y_pred, y_true)
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.check_shape(y_pred, y_true)
        return y_pred - y_true  # / self.batch_size
    
    def __str__(self) -> str:
        return "MSE"


class CrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.check_shape(y_pred, y_true)
        return -np.sum(y_true * np.log(y_pred)) / self.batch_size

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.check_shape(y_pred, y_true)
        return -y_true / y_pred / self.batch_size
    
    def __str__(self) -> str:
        return "CrossEntropy"


class BinaryCrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.check_shape(y_pred, y_true)
        return (
            -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            / self.batch_size
        )

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.check_shape(y_pred, y_true)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / self.batch_size
    
    def __str__(self) -> str:
        return "BinaryCrossEntropy"

