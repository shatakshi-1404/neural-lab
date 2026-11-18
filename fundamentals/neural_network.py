import numpy as np
from .activations import sigmoid

class TinyMLP:
    """One-hidden-layer binary classifier with manual batch backpropagation."""
    def __init__(self, input_size, hidden_size=8, learning_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, 0.5, (hidden_size, 1))
        self.b2 = np.zeros((1, 1))
        self.learning_rate = learning_rate
        self.loss_history = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def fit(self, X, y, epochs=3000):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n = len(X)
        self.loss_history = []
        for _ in range(epochs):
            predictions = self.forward(X)
            eps = 1e-8
            loss = -np.mean(y*np.log(predictions+eps) + (1-y)*np.log(1-predictions+eps))
            self.loss_history.append(float(loss))

            dz2 = (predictions - y) / n
            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - self.a1**2)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
        return self

    def predict_proba(self, X):
        return self.forward(np.asarray(X, dtype=float)).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
