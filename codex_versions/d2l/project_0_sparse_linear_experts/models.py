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


# Computes one regression prediction per top-k routed expert, then averages the k predictions per example. Also returns top_ids so we can inspect routing
def top2_routed_predict_regression(X, expert_W, expert_b, region_table):

    # Ask the router for the best 2 experts per input row. top_ids shape: [batch, 2]
    top_ids, _, _ = route_topk(X, region_table, k=2)

    # Allocate 2 prediction columns/example. Column 0 stores 1st expert's prediction, column 1 stores 2nd expert's prediction. preds shape: [batch, 2]
    preds = torch.zeros(X.shape[0], 2)

    # j selects which top-k slot we are filling: 0 for best expert, 1 for second-best
    for j in range(2):

        # Pull 1 route column out of top_ids. route_ids shape: [batch]
        route_ids = top_ids[:, j]

        # For this route column, compute predictions expert by expert
        for r in range(expert_W.shape[0]):

            # mask selects examples whose j-th route is expert r
            mask = route_ids == r

            # Skip experts that no examples selected in this slot
            if mask.any():

                # X[mask] shape: [examples_for_r, features] ; expert_W[r] shape: [features]; preds[mask, j] shape: [examples_for_r]
                preds[mask, j] = X[mask] @ expert_W[r] + expert_b[r]

    # Average the 2 expert predictions for each example. preds.mean(dim=1) shape: [batch]
    return preds.mean(dim=1), top_ids

# Create routed classification logits from input features and expert parameters
def routed_classification_logits(X, expert_W, expert_b, route_ids):
    num_examples = X.shape[0]
    num_classes = expert_b.shape[1]
    logits = torch.zeros(num_examples, num_classes)

    # For each expert r, find rows routed to r, compute class scores for those rows, write those scores back into the matching rows of logits
    for r in range(expert_W.shape[0]):
        mask = route_ids == r
        if mask.any():
            logits[mask] = X[mask] @ expert_W[r] + expert_b[r]

    return logits