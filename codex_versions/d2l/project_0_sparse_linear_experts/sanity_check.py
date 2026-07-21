import torch

X = torch.randn(4, 6)
w = torch.randn(6)
b = torch.tensor(0.5)
y_hat = X @ w + b

print("torch version:", torch.__version__)
print("X:", X.shape)
print("w:", w.shape)
print("y_hat:", y_hat.shape) # (4, 6) @ (6, ) = (4, )
print(y_hat)