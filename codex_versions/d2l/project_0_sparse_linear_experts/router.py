import torch

# X / row_norms divides every feature in each row by that row's length, so vector length does not dominate the cosine similarity score.
# The output shape stays [num_examples, num_features], but each row now has length about 1.
def normalize_rows(X):
    return X / (X.norm(dim=1, keepdim=True) + 1e-8)

# Match respective cosine similarty to nearest MoE
# Shape contract:
    # X:            [batch, features]
    # region_table: [regions, features]
    # scores:       [batch, regions]
    # top_ids:      [batch, k]
    # top_scores:   [batch, k]
def route_topk(X, region_table, k=1):

    # Normalized input vectors.
    X_norm = normalize_rows(X)

    # Normalized region prototype vectors. This is required to calculate the cosine similarity between input against the MoEs
    table_norm = normalize_rows(region_table)

    # Computes cosine similarity between every input and every region prototype
    # X norm's shape is [num_examples, num_features] and table_norm.T's shape is [num_features, num_regions]
    # Scores' shape is then [batch, regions] where the cosine similarity for each input row is compared against each region prototype row
    scores = X_norm @ table_norm.T

    # For each input row, this picks the best k regions.
    # e.g. If k=2, each input gets two best regions: top_ids[i] = the 2 closest region ids for input i
    top_scores, top_ids = torch.topk(scores, k=k, dim=1)

    # top_ids     = chosen region/expert ids
    # top_scores  = cosine scores for those chosen ids
    # scores      = full similarity table against all regions
    return top_ids, top_scores, scores

# Choose random regions
def random_routes(num_examples, num_regions):
    return torch.randint(0, num_regions, (num_examples,))

# Use cosine similarity to region table
def similarity_routes(X, region_table):
    top_ids, _, _ = route_topk(X, region_table, k=1)
    return top_ids.squeeze(1)