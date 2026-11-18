import numpy as np

def numerical_gradient(function, parameter, epsilon=1e-5):
    """Finite-difference gradient; useful for checking analytic gradients."""
    gradient = np.zeros_like(parameter, dtype=float)
    iterator = np.nditer(parameter, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = parameter[index]
        parameter[index] = original + epsilon
        plus = function()
        parameter[index] = original - epsilon
        minus = function()
        parameter[index] = original
        gradient[index] = (plus - minus) / (2 * epsilon)
        iterator.iternext()
    return gradient
