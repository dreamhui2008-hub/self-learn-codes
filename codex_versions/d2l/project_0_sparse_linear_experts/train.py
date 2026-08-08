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

def l2_penalty(w):
    return (w ** 2).sum() / 2

def rescale_expert_weights(expert_W, target_norm=1.0):
    # This function directly changes expert_W values, so do it outside autograd

    with torch.no_grad():

        # Compute one L2 norm per expert row
        # If expert_W shape is [regions, features], norms shape is [regions, 1]
        norms = expert_W.norm(dim=1, keepdim=True)

        # Compute one multiplier per expert row
        # The small 1e-8 prevents division by zero
        scale = target_norm / (norms + 1e-8)

        # Multiply each expert row by its own scale
        # Broadcasting works because scale shape is [regions, 1]
        expert_W *= scale