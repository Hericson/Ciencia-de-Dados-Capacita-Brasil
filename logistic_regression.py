import numpy as np

class LinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        w = np.linalg.pinv(X.T @ X) @ X.T @ y
        return w
    def predict(self, X, w):
        return X @ w
