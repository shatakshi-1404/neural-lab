import numpy as np

class Perceptron:
    """Binary perceptron trained with the classic perceptron update rule."""
    def __init__(self, learning_rate=0.1, epochs=20, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = int(np.dot(xi, self.weights) + self.bias >= 0)
                error = target - prediction
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.weights + self.bias >= 0).astype(int)
