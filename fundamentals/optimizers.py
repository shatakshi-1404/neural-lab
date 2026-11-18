import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter -= self.learning_rate * gradient

class Adam:
    """Minimal Adam optimizer for NumPy arrays."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr, self.beta1, self.beta2, self.epsilon = learning_rate, beta1, beta2, epsilon
        self.t = 0
        self.m = None
        self.v = None

    def step(self, parameters, gradients):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in parameters]
            self.v = [np.zeros_like(p) for p in parameters]
        self.t += 1
        for i, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*gradient
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(gradient**2)
            m_hat = self.m[i] / (1-self.beta1**self.t)
            v_hat = self.v[i] / (1-self.beta2**self.t)
            parameter -= self.lr*m_hat/(np.sqrt(v_hat)+self.epsilon)
