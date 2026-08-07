import torch

def per_region_mse(y_hat, y, region_ids, num_regions):

    values = []

    for r in range(num_regions):
        mask = region_ids == r
        if mask.any():
            mse = ((y_hat[mask] - y[mask]) ** 2).mean()
            values.append(mse.item())
        else:
            values.append(None)

    return values

def accuracy(logits, y):
    predictions = logits.argmax(dim=1)
    return (predictions == y).float().mean()