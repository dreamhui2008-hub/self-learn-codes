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

# Create fixed region prototypes for routing.
# Later, each input/query is compared against these unit vectors. The closest prototype determines which expert should handle it.
# Shape contract: region_table: [num_regions, num_features]
# Each row is one region embedding for similarity routing against X @ region_table.T
def make_region_table(num_regions, num_features):
    table = torch.randn(num_regions, num_features)
    table = table / table.norm(dim=1, keepdim=True)
    return table

# Create hidden ground-truth regression rules for each region to generate y (output)
# These are used to generate synthetic labels, not trained by the model, therefore its fine for them to be random.
# The goal is not that our random keys are true, but that our model can discover underlying pattern/train towards the random keys.
# Shape contract: 
    # true_W: [num_regions, num_features]
    # true_b: [num_regions]
# For region r, the hidden rule is:
    # y = x @ true_W[r] + true_b[r] + noise
# Later, you will see expert_W and expert_b as trainable MoE params
def make_region_rules(num_regions, num_features):
    true_W  = torch.randn(num_regions, num_features)
    true_b = torch.randn(num_regions)
    return true_W, true_b

# Shape contract:
    # X:          [num_examples, num_features]
    # y:          [num_examples]
    # region_ids: [num_examples]
def make_sparse_regression_data(
        num_examples,
        region_table,
        true_W,
        true_b,
        mixture,
        feature_noise=0.3,
        label_noise=0.1,
):
    num_regions, num_features = region_table.shape

    # Create num_examples region IDs using the probabilities in mixture. mixture is a vector of probabilities that should sum to 1.
    # The regions_ids are randomly created with fixed probabilties (e.g. region_0 always have 25% probability, but id placement is random)
    region_ids = torch.multinomial(mixture, num_examples, replacement=True)
    
    # Creates input vectors near assigned region prototype. region_ids decide where each X is located in feature space
    X = region_table[region_ids] + torch.randn(num_examples, num_features) * feature_noise

    # Generates labels using the matching region’s hidden rule from true_W and true_b
    y = (X * true_W[region_ids]).sum(dim=1) + true_b[region_ids]
    y = y + torch.randn(num_examples) * label_noise

    return X, y, region_ids
