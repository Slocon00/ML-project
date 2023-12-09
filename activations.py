import numpy as np

def idebtity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x <= 0, 0, 1)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

#serve? bho l'ha messa copilot
def d_mse(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

