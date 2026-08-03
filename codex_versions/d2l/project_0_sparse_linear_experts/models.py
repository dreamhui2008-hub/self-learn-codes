import torch
from router import route_topk

# Define the linear regression model (prediction function)
def predict_regression(X, w, b):
    return X @ w + b

# Manual MSELoss
def squared_loss(y_hat, y):
    return ((y_hat - y) ** 2).mean()


# Computes MSEloss for each routed expert/region, sums those losses across all experts, counts how many examples contributed, 
# and returns the overall MSELoss across all routed examples.
def routed_regression_loss(X, y, expert_W, expert_b, route_ids):
    losses = []
    counts = []

    # Loops through each expert
    for r in range(expert_W.shape[0]):

        # For each region/expert r, it finds the examples routed there:
        # Example: route_ids = tensor([0, 2, 2, 1, 0]), mask for r=2 -> tensor([False, True, True, False, False])
        mask = route_ids == r

        # If there is one element inside mask that is True (at least 1 example being routed to this expert)
        if mask.any():

            # For all examples routed to region r, predict using region r’s linear model.
            y_hat = X[mask] @ expert_W[r] + expert_b[r]

            # Computes squared error only for those examples
            errors = (y_hat - y[mask]) ** 2

            # Saves the losses and counts (how many examples contributed to expert's loss) by appending to lists
            # Notices how losses are stored by the sums, not their mean
            losses.append(errors.sum())
            counts.append(errors.numel()) # numel() calculates the # of elements in a tensor (or how many squared-error values this expert produced in this case)

    # Gives the average squared error over all examples, not over experts
    # If expert 0 saw >200 examples and expert 1 only saw 2 examples, you usually do not want those experts to have equal weight in the final loss
    total_loss = torch.stack(losses).sum() # stack() takes a Python list and convert it to a tensor
    total_count = sum(counts)
    return total_loss / total_count


# Routes each example to its nearest region, then uses that region's expert to make a regression prediction
def routed_predict_regression(X, expert_W, expert_b, region_table):

    # Use route_topk to assign each input row to one region/expert
    top_ids, _, _ = route_topk(X, region_table, k=1)

    # Convert top_ids from shape [batch, 1] to route_ids with shape [batch], so each route id can be compared with the expert index r
    route_ids = top_ids.squeeze(1)

    # Create an empty prediction tensor y_hat
    y_hat = torch.zeros(X.shape[0])

    # Loop through each expert
    # For examples routed to that expert, predict using expert_W[r] and expert_b[r]
    for r in range(expert_W.shape[0]):
        mask = route_ids == r
        if mask.any():
            y_hat[mask] = X[mask] @ expert_W[r] + expert_b[r]

    # Return both predictions and route_ids
    return y_hat, route_ids
