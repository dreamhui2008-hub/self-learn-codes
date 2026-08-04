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

# Splitting training regions into batches
def train_test_split_with_regions(X, y, region_ids, train_fraction=0.8):
    n = X.shape[0]
    shuffled = torch.randperm(n)
    train_size = int(n * train_fraction)
    train_idx = shuffled[:train_size]
    test_idx = shuffled[train_size:]
    return (
        X[train_idx],
        y[train_idx],
        region_ids[train_idx],
        X[test_idx],
        y[test_idx],
        region_ids[test_idx],
    )


def make_region_class_rules(num_regions, num_features, num_classes):
    # Create one hidden class-weight matrix per region.
    # Shape: [regions, features, classes]
    true_W = torch.randn(num_regions, num_features, num_classes)

    # Create one hidden class-bias vector per region.
    # Shape: [regions, classes]
    true_b = torch.randn(num_regions, num_classes)

    # These are answer-key parameters used to generate labels, not trained parameters.
    return true_W, true_b

# Shape contract:
    # true_W:     [regions, features, classes]
    # true_b:     [regions, classes]
    # X:          [examples, features]
    # y:          [examples]
    # region_ids: [examples]
    # logits:     [examples, classes]
def make_sparse_classification_data(
        num_examples,
        region_table,
        true_W,
        true_b,
        mixture,
        feature_noise=0.3,
):
    # region_table shape is [regions, features].
    # Unpack those dimensions so the rest of the function stays shape-driven.
    num_regions, num_features = region_table.shape

    # Sample one true region ID per example (num_examples) using the mixture probabilities.
    # replacement=True means the same region can be chosen again and again. Each draw is independent.
    # Shape: [num_examples]
    region_ids = torch.multinomial(mixture, num_examples, replacement=True)


    # Build input rows near their assigned region prototype.
    # region_table[region_ids] shape: [num_examples, features]
    # noise shape: [num_examples, features]
    # X shape: [num_examples, features]
    X = region_table[region_ids] + torch.randn(num_examples, num_features) * feature_noise

    # Allocate the hidden logits table that will become class labels.
    # true_W.shape[2] is num_classes.
    # logits shape: [num_examples, classes]
    logits = torch.zeros(num_examples, true_W.shape[2])


    # Fill logits region by region so each example uses its own region's hidden rule.
    for r in range(num_regions):
        # mask shape: [num_examples]
        # True entries mark examples whose hidden region is r.
        mask = region_ids == r

        # Skip empty regions so X[mask] never becomes an empty training block here.
        if mask.any():

            # X[mask] shape: [examples_for_r, features]
            # true_W[r] shape: [features, classes]
            # true_b[r] shape: [classes]
            # logits[mask] shape: [examples_for_r, classes]
            logits[mask] = X[mask] @ true_W[r] + true_b[r]

    # Convert hidden class scores into integer class IDs.
    # y shape: [num_examples]
    y = logits.argmax(dim=1)

    # Return inputs, class labels, and true synthetic region IDs.
    return X, y, region_ids