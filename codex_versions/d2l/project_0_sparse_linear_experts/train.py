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

class ReplayBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.X = []
        self.y = []

    # Store detached copies of new examples, then keep only the most recent max_size examples. This drops the oldest/leading examples from the buffer if it gets too large
    def add(self, X, y):

        for i in range(X.shape[0]):
            self.X.append(X[i].detach().clone())
            self.y.append(y[i].detach().clone())

        self.X = self.X[-self.max_size:]
        self.y = self.y[-self.max_size:]

    # Randomly sample stored examples and stack them into batch tensors. It can also sample the same stored example more than once.
    def sample(self, batch_size):
        n = len(self.X)
        idx = torch.randint(0, n, (batch_size,))
        X = torch.stack([self.X[int(i)] for i in idx])
        y = torch.stack([self.y[int(i)] for i in idx])

        return X, y