import torch

# Generate a synthetic linear regression dataset.
# The targets are created from a known linear relationship: y = X @ true_w + true_b + noise
def make_regression_data(num_examples, num_features, noise_std=0.1):
    true_w = torch.tensor([2.0, -3.0, 1.5, 0.0, 0.5, -1.0])
    true_b = torch.tensor(0.7)
    
    X = torch.randn(num_examples, num_features)
    noise = torch.randn(num_examples) * noise_std
    y = X @ true_w + true_b + noise
    return X, y, true_w, true_b

# Shape contract:
    # X:      [num_examples, num_features]
    # y:      [num_examples]
    # true_w: [num_features]
    # true_b: scalar

# Split the dataset into training and test sets.
# The training set is used to fit the model.
# The test set is used later to evaluate how well the trained model generalizes to unseen data.
def train_test_split(X, y, train_fraction=0.8):
     # Take the # of examples as n
    n = X.shape[0]

    # Creates a random permutation/order of the integers from range(0, n); e.g. [1, 0, 3, 2, 4] for n = 5
    shuffled = torch.randperm(n) 

     # Number of examples to include in the training set (80% by default)
    train_size = int(n * train_fraction)

     # Takes the first train_size shuffled indices e.g. [1, 0, 3, 2]
    train_idx = shuffled[:train_size]

     # Takes the remainder train_size shuffled indices e.g. [4]
    test_idx = shuffled[train_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]