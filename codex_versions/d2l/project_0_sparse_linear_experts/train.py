import torch

def sgd(params, lr):

     # Don't track the parameter updates in the computation graph (to do at optimizer.step())
    with torch.no_grad():

         # Update each model parameter once (e.g., w and b); not repeatedly until convergence (e.g. not searching for local minimum).
        for p in params:

            # Gradient descent update: new param = old param − lr × gradient
            p -= lr * p.grad
            
            # Clear the gradient so it doesn't accumulate on the next backward pass.
            p.grad.zero_()