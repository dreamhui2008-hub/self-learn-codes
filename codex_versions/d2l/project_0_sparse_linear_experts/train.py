import torch

def sgd(params, lr):
    with torch.no_grad(): # Don't track the parameter updates in the computation graph (to do at optimizer.step())
        for p in params: # Update each model parameter once (e.g., w and b); not repeatedly until convergence (e.g. not searching for local minimum).
            p -= lr * p.grad # Gradient descent update: new param = old param − lr × gradient
            p.grad.zero_() # Clear the gradient so it doesn't accumulate on the next backward pass.