import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred, epsilon=1e-8):
    p = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(y_true * np.log(p) + (1-y_true) * np.log(1-p))

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-8):
    p = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(y_true * np.log(p), axis=1))
