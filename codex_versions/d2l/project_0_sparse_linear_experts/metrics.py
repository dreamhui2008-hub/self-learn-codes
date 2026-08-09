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

def confusion_matrix(pred, y, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64) # Rows are true labels. Columns are predicted labels

    # Count the occurance of each true to guessed pairs inside a matrix. true (y) is in rows and guessed (pred) is in columns
    for true, guessed in zip(y, pred):

        # If true and guessed matched (y == pred), add 1 inside matrix for index position matrix[true, guessed]. Correct predictions land on the diagonal
        matrix[true, guessed] += 1

    return matrix