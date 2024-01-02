import numpy as np


class Metric:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError


class Accuracy(Metric):
    """Metric used to calculate accuracy for binary classification tasks for a
    net with the sigmoid activation function on its output layer.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        y_true_reshaped = y_true.reshape(y_true.shape[0], y_true.shape[1])

        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        # check the accuracy in percentage
        accuracy = np.sum(y_pred == y_true_reshaped) / len(y_true_reshaped) * 100
        return accuracy


class MEE(Metric):
    """Metric used to calculate Mean Euclidean Error for regression tasks."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        y_true_reshaped = y_true.reshape(y_true.shape[0], y_true.shape[1])

        return np.mean(np.sqrt(np.sum(((y_pred - y_true_reshaped) ** 2), axis=1)))
