import torch

# Define the linear regression model (prediction function)
def predict_regression(X, w, b):
    return X @ w + b

# Manual MSELoss
def squared_loss(y_hat, y):
    return ((y_hat - y) ** 2).mean()
