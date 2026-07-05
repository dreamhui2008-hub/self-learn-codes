"""Generate Chapter 4 notebooks for the D2L rewrite project."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Chapter 4 - Linear Neural Networks for Classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    text = text.strip("\n") + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    text = text.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def section(title: str, intuition: str, why: str, examples, breakdown: str, connection: str, confusion: str) -> list[dict]:
    cells = [md(f"""### {title}

#### 1. Intuition

{intuition.strip()}

#### 2. Why this exists

{why.strip()}""")]
    cells.append(md("#### 3. Examples\n\n" + examples[0][1].strip()))
    for kind, content in examples[1:]:
        cells.append(code(content) if kind == "code" else md(content))
    cells.append(md(f"""#### 4. Step-by-step breakdown

{breakdown.strip()}

#### 5. Connection to ML systems

{connection.strip()}

#### 6. Common confusion points

{confusion.strip()}"""))
    return cells


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    (OUT_DIR / filename).write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")


def title_cell(title: str, description: str) -> dict:
    return md(f"### {title}\n\n{description.strip()}")


def imports_cell(extra: str = "") -> dict:
    base = "import math\nimport random\nimport numpy as np\nimport torch"
    return code(base + ("\n" + extra.strip() if extra.strip() else ""))



def build_41() -> None:
    cells = [
        title_cell(
            "Chapter 4.1 - Softmax Regression",
            "Softmax regression is linear regression's classification counterpart. It predicts a probability distribution over classes instead of one numerical value.",
        ),
        imports_cell(),
    ]

    cells += section(
        "4.1.1 Classification",
        """Classification means predicting a category. A category is a named group, such as `cat`, `dog`, or `shirt`.

A class is one possible category. A label is the correct class for one example. A score, also called a logit, is a raw number a model assigns to a class before turning scores into probabilities.""",
        """Many ML tasks ask for a decision among choices rather than a number. The model must compare classes, not just output one scalar prediction.""",
        [
            ("md", "Manual Python: choose the class with the largest score."),
            ("code", """classes = ["cat", "dog", "car"]
scores = [1.2, 3.4, 0.5]
best_index = scores.index(max(scores))
predicted_class = classes[best_index]
predicted_class"""),
            ("md", "PyTorch: compute class scores from features and weights."),
            ("code", """X = torch.tensor([[1.0, 2.0]])
W = torch.tensor([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]])
b = torch.tensor([0.0, 0.1, -0.1])
logits = X @ W + b
logits"""),
        ],
        """The manual example compares three class scores.

`max(scores)` finds the largest score.

`scores.index(...)` finds the position of that largest score.

In the tensor example, `X` has one example with two features.

`W` maps two input features to three class scores.

`X @ W + b` returns one score per class.""",
        """Classification models usually output one score per class. Later, softmax turns those scores into probabilities that sum to 1.""",
        """- A class score is not automatically a probability.
- The largest score determines the predicted class.
- The number of output scores should match the number of classes.
- Classification labels are categories, even when represented by integer IDs.""",
    )

    cells += section(
        "4.1.2 Loss Function",
        """Softmax turns raw class scores into probabilities. A probability is a number between 0 and 1, and class probabilities should sum to 1.

Cross-entropy loss penalizes the model when it assigns low probability to the correct class.""",
        """Classification training needs a loss that rewards probability placed on the correct class and penalizes probability placed elsewhere.""",
        [
            ("md", "Manual softmax for three class scores."),
            ("code", """scores = torch.tensor([1.0, 2.0, 0.0])
exp_scores = torch.exp(scores)
probs = exp_scores / exp_scores.sum()
probs"""),
            ("md", "Cross-entropy for the correct class index."),
            ("code", """correct_class = 1
loss = -torch.log(probs[correct_class])
loss"""),
        ],
        """`torch.exp(scores)` makes all values positive.

Dividing by `exp_scores.sum()` makes the values sum to 1.

`correct_class = 1` means the second class is the true label.

`probs[correct_class]` selects the probability assigned to the true class.

`-torch.log(...)` gives a small loss for high probability and a large loss for low probability.""",
        """PyTorch usually combines softmax and cross-entropy in `torch.nn.CrossEntropyLoss` for numerical stability. Numerical stability means avoiding avoidable overflow, underflow, or precision problems.""",
        """- Softmax probabilities sum to 1 across classes.
- Cross-entropy uses the correct class probability.
- A confident wrong prediction gets a large loss.
- In PyTorch, `CrossEntropyLoss` expects raw logits, not already-softmaxed probabilities.""",
    )

    cells += section(
        "4.1.3 Information Theory Basics",
        """Information theory studies uncertainty and information.

Surprisal measures how unexpected an event is. A high-probability event is not surprising. A low-probability event is surprising.

The negative logarithm, `-log(p)`, is a common mathematical measure of surprisal.""",
        """Cross-entropy loss is easier to understand when viewed as surprisal: the model is penalized by how surprised it is when the true class occurs.""",
        [
            ("md", "Compare surprisal for high and low probabilities."),
            ("code", """high_prob = torch.tensor(0.9)
low_prob = torch.tensor(0.1)
high_surprisal = -torch.log(high_prob)
low_surprisal = -torch.log(low_prob)
high_surprisal, low_surprisal"""),
        ],
        """`high_prob` represents a confident probability for the event.

`low_prob` represents an unlikely event.

`-log(0.9)` is small because the event is not surprising.

`-log(0.1)` is larger because the event is surprising.

Cross-entropy applies this idea to the true class probability.""",
        """Classification training minimizes average surprisal of the correct labels under the model's predicted probabilities.""",
        """- Information theory terms can sound abstract, but here the key idea is surprise.
- Logarithms turn probability products into sums, which is useful for optimization.
- Cross-entropy is not accuracy; it uses probability confidence.
- A model can have good accuracy but poor cross-entropy if it is badly calibrated.""",
    )

    cells += section(
        "4.1.4 Summary and Discussion",
        """Softmax regression maps input features to one score per class, converts scores to probabilities, and trains with cross-entropy loss.""",
        """It is the simplest linear classifier and the foundation for later neural classifiers.""",
        [
            ("md", "One compact softmax-regression forward pass."),
            ("code", """X = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
W = torch.randn(2, 3)
b = torch.zeros(3)
logits = X @ W + b
probs = torch.softmax(logits, dim=1)
probs.shape, probs.sum(dim=1)"""),
        ],
        """`X @ W + b` computes raw scores for each example and class.

`torch.softmax(logits, dim=1)` normalizes across class columns.

`probs.shape` is `(2, 3)`: two examples and three classes.

`probs.sum(dim=1)` checks that each example's class probabilities sum to 1.""",
        """Deep classifiers often end with the same pattern: class logits followed by cross-entropy loss.""",
        """- Use `dim=1` when rows are examples and columns are classes.
- Softmax is about relative scores, not absolute score size alone.
- Cross-entropy trains probabilities, not only the final class choice.
- The model is still linear unless hidden layers are added.""",
    )

    cells += section(
        "4.1.5 Exercises",
        """Exercises should make you inspect logits, probabilities, labels, and loss separately.""",
        """Classification bugs often come from mixing up class dimension, label format, or whether values are logits or probabilities.""",
        [
            ("md", "Exercise 1: turn logits into probabilities."),
            ("code", """logits = torch.tensor([[2.0, 1.0, 0.0]])
probs = torch.softmax(logits, dim=1)
probs, probs.sum(dim=1)"""),
            ("md", "Exercise 2: compute cross-entropy with PyTorch from raw logits."),
            ("code", """labels = torch.tensor([0])
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
loss"""),
        ],
        """Exercise 1 checks softmax normalization across classes.

Exercise 2 checks the PyTorch convention: pass raw logits and integer class labels.

The label `0` means the first class is correct.""",
        """These exercises prepare for from-scratch and concise softmax-regression implementations.""",
        """- Do not pass one-hot labels to `CrossEntropyLoss` in the basic PyTorch use case.
- Do not apply softmax before `CrossEntropyLoss` unless you know the alternative API expects it.
- Check the class dimension before using softmax.
- A probability vector should sum to 1 across classes.""",
    )

    write_nb("Chapter 4.1 - Softmax Regression.ipynb", cells)


def build_42() -> None:
    cells = [
        title_cell(
            "Chapter 4.2 - The Image Classification Dataset",
            "Image classification uses images as inputs and class IDs as labels. This notebook uses tiny synthetic image tensors so the data shape and batching ideas are visible without downloading a dataset.",
        ),
        imports_cell(),
    ]

    cells += section(
        "4.2.1 Loading the Dataset",
        """An image dataset contains image tensors and labels. A grayscale image can be stored as a matrix of pixel values.

A pixel is one small measurement in an image. A channel is one layer of pixel measurements, such as grayscale, red, green, or blue.""",
        """Image models need a consistent tensor shape. For many PyTorch image datasets, the shape is `(examples, channels, height, width)`.""",
        [
            ("md", "Create a tiny image dataset with four grayscale images."),
            ("code", """images = torch.arange(4 * 1 * 2 * 2, dtype=torch.float32)
images = images.reshape(4, 1, 2, 2)
labels = torch.tensor([0, 1, 0, 1])
dataset = torch.utils.data.TensorDataset(images, labels)
len(dataset), images.shape, labels.shape"""),
        ],
        """`torch.arange(...)` creates enough pixel values for four small images.

`reshape(4, 1, 2, 2)` means 4 examples, 1 channel, height 2, width 2.

`labels` stores one class ID per image.

`TensorDataset(images, labels)` keeps each image aligned with its label.""",
        """Real datasets such as Fashion-MNIST follow the same idea but with many more images and larger spatial dimensions.""",
        """- Image tensors have shape conventions; always check them.
- Labels should align with the first dimension of images.
- Pixel values are numbers, even when the image looks visual to humans.
- Loading data is separate from training a model.""",
    )

    cells += section(
        "4.2.2 Reading a Minibatch",
        """A minibatch is a small group of examples returned together for one training step.

For images, a minibatch keeps the image dimensions and adds or preserves the batch dimension.""",
        """Training one image at a time wastes hardware efficiency. Training all images at once may use too much memory. Minibatches balance both concerns.""",
        [
            ("md", "Read one minibatch using a DataLoader."),
            ("code", """loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
batch_images, batch_labels = next(iter(loader))
batch_images.shape, batch_labels.shape"""),
            ("md", "Flatten images when using a linear classifier."),
            ("code", """flat = batch_images.reshape(batch_images.shape[0], -1)
flat.shape"""),
        ],
        """`DataLoader` groups dataset items into batches.

`next(iter(loader))` asks for the first batch.

`batch_images.shape` is `(2, 1, 2, 2)` for two tiny grayscale images.

`reshape(batch_images.shape[0], -1)` keeps the batch size and flattens each image into one feature vector.""",
        """Softmax regression is a linear classifier, so it expects feature vectors. For images, that often means flattening pixels before the linear layer.""",
        """- Flattening removes spatial layout information.
- `-1` asks PyTorch to infer the remaining dimension.
- The batch dimension should stay first.
- Shuffling should not break image-label pairing.""",
    )

    cells += section(
        "4.2.3 Visualization",
        """Visualization means displaying data so humans can inspect it.

For images, visual inspection helps catch wrong labels, wrong shapes, inverted colors, or normalization mistakes.""",
        """Models only see numbers, but humans often need visual checks to verify the data pipeline.""",
        [
            ("md", "Prepare one tiny image for display as a 2D grid."),
            ("code", """image = images[0]
grid = image.squeeze(0)
label = labels[0]
grid, label"""),
            ("md", "A class-name lookup turns label IDs into readable names."),
            ("code", """class_names = ["dark", "bright"]
name = class_names[int(label)]
name"""),
        ],
        """`images[0]` selects the first image.

`squeeze(0)` removes the channel dimension when it has size 1.

The label is an integer class ID.

`class_names[int(label)]` maps the class ID to a human-readable name.""",
        """Dataset visualization is usually done before training. It is a cheap way to find data mistakes that no optimizer can fix.""",
        """- Visualization is for inspection, not training.
- A label ID needs a class-name mapping to be readable.
- Squeezing removes dimensions only when their size is 1.
- Do not confuse display shape with model input shape.""",
    )

    cells += section(
        "4.2.4 Summary",
        """Image classification datasets pair image tensors with class labels.

The key shape convention is usually `(batch, channels, height, width)`.""",
        """Correct data shape and label alignment are prerequisites for meaningful classification training.""",
        [
            ("md", "Summarize one batch's structure."),
            ("code", """summary = {
    "batch_size": batch_images.shape[0],
    "channels": batch_images.shape[1],
    "height": batch_images.shape[2],
    "width": batch_images.shape[3],
}
summary"""),
        ],
        """The summary names each dimension.

Naming dimensions reduces confusion when tensors become larger.

The label tensor stores one class ID per image in the batch.""",
        """Future image models will preserve spatial layout longer, but softmax regression starts by flattening images into feature vectors.""",
        """- Always inspect batch shapes.
- Always preserve image-label alignment.
- Flattening is convenient but loses spatial structure.
- Class labels are usually integer IDs in PyTorch classification.""",
    )

    cells += section(
        "4.2.5 Exercises",
        """These exercises practice reading and reshaping image batches.""",
        """Most image-classification bugs start as data-shape bugs.""",
        [
            ("md", "Exercise 1: create a batch of three 4 by 4 grayscale images."),
            ("code", """X = torch.zeros(3, 1, 4, 4)
y = torch.tensor([0, 1, 2])
X.shape, y.shape"""),
            ("md", "Exercise 2: flatten the image batch for a linear classifier."),
            ("code", """flat = X.reshape(X.shape[0], -1)
flat.shape"""),
        ],
        """Exercise 1 checks the image batch shape convention.

Exercise 2 checks flattening while preserving the batch dimension.

The flattened feature count is `channels * height * width`.""",
        """Flattened image batches feed directly into softmax regression and multilayer perceptrons.""",
        """- The batch dimension is not an image dimension.
- Flatten only the dimensions that belong to each example.
- The number of labels should equal the batch size.
- The class ID range should match the number of classes.""",
    )

    write_nb("Chapter 4.2 - The Image Classification Dataset.ipynb", cells)


def build_43() -> None:
    cells = [
        title_cell(
            "Chapter 4.3 - The Base Classification Model",
            "A base classification model collects shared behavior for classifiers, especially prediction and accuracy measurement. This notebook builds those ideas manually before using class structure.",
        ),
        imports_cell(),
    ]

    cells += section(
        "4.3.1 The Classifier Class",
        """A classifier is a model that predicts class labels.

A base class is a parent class that stores behavior shared by several related classes. In Python, inheritance lets a child class reuse parent behavior.""",
        """Many classifiers share the same evaluation logic even when their internal model architecture differs. A base class avoids rewriting that shared code.""",
        [
            ("md", "Plain Python classifier interface."),
            ("code", """class SimpleClassifier:
    def predict(self, logits):
        return torch.argmax(logits, dim=1)

model = SimpleClassifier()
logits = torch.tensor([[1.0, 3.0], [2.0, 0.5]])
model.predict(logits)"""),
        ],
        """`class SimpleClassifier:` defines a class.

`predict` is a method that receives logits.

`torch.argmax(logits, dim=1)` selects the class index with the largest score for each row.

`model.predict(logits)` returns predicted class IDs.""",
        """D2L and PyTorch code often define classifier classes so training, validation, and prediction can use a common interface.""",
        """- A classifier predicts categories, not continuous values.
- `argmax` chooses the index of the largest value.
- `dim=1` means compare across class columns.
- A base class should contain shared behavior, not task-specific hidden assumptions.""",
    )

    cells += section(
        "4.3.2 Accuracy",
        """Accuracy is the fraction of predictions that match the correct labels.

A correct prediction is one where predicted class ID equals the true class ID.""",
        """Classification loss measures probability quality, while accuracy measures final decision correctness. Both are useful but answer different questions.""",
        [
            ("md", "Manual accuracy computation."),
            ("code", """preds = torch.tensor([1, 0, 1, 1])
labels = torch.tensor([1, 1, 1, 0])
correct = (preds == labels)
accuracy = correct.float().mean()
accuracy"""),
            ("md", "Accuracy from logits."),
            ("code", """logits = torch.tensor([[0.1, 0.9], [2.0, 1.0]])
labels = torch.tensor([1, 0])
preds = torch.argmax(logits, dim=1)
(preds == labels).float().mean()"""),
        ],
        """`preds == labels` creates Boolean values.

`True` means correct and `False` means incorrect.

`.float()` converts `True` to 1.0 and `False` to 0.0.

`.mean()` averages those 0/1 values into a fraction correct.""",
        """Training dashboards often report both loss and accuracy. Loss can improve before accuracy changes because probabilities can improve without changing the final class choice.""",
        """- Accuracy ignores confidence.
- Accuracy can be misleading on imbalanced datasets.
- Use `argmax` on logits or probabilities; both preserve score ordering.
- Accuracy is not differentiable, so it is usually not the training loss.""",
    )

    cells += section(
        "4.3.3 Summary",
        """A classifier maps examples to class scores, predicted class IDs, and evaluation metrics.

Accuracy is the simplest classification metric.""",
        """Separating prediction logic from metric logic makes training code easier to reuse.""",
        [
            ("md", "Package prediction and accuracy together."),
            ("code", """def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()

accuracy_from_logits(logits, labels)"""),
        ],
        """The helper computes predicted class IDs from logits.

It compares predictions against labels.

It returns the average correctness as a scalar tensor.""",
        """This helper is a tiny version of the evaluation code used by classifier classes.""",
        """- Prediction and evaluation are related but separate steps.
- Accuracy returns an estimate over the examples provided.
- Larger evaluation sets usually give more stable estimates.
- Metric helpers should document expected shapes.""",
    )

    cells += section(
        "4.3.4 Exercises",
        """These exercises test whether you can compute classifier predictions and accuracy from logits.""",
        """Accuracy is simple, but shape mistakes can still produce wrong results.""",
        [
            ("md", "Exercise 1: predict class IDs for three examples."),
            ("code", """logits = torch.tensor([[1.0, 0.0], [0.2, 0.8], [3.0, 4.0]])
preds = torch.argmax(logits, dim=1)
preds"""),
            ("md", "Exercise 2: compute accuracy."),
            ("code", """labels = torch.tensor([0, 1, 0])
acc = (preds == labels).float().mean()
acc"""),
        ],
        """Exercise 1 checks `argmax` across the class dimension.

Exercise 2 checks Boolean comparison and averaging.

The third example is wrong because class 1 has the larger score while the label is 0.""",
        """These exact operations appear in validation loops for softmax regression and later classifiers.""",
        """- Class IDs start at 0 in PyTorch examples.
- Prediction shape should match label shape.
- Accuracy treats all errors equally.
- Accuracy should be computed without updating model parameters.""",
    )

    write_nb("Chapter 4.3 - The Base Classification Model.ipynb", cells)


def build_44() -> None:
    cells = [
        title_cell(
            "Chapter 4.4 - Softmax Regression Implementation from Scratch",
            "This notebook implements softmax regression directly with tensors so every part of classification training is visible before framework shortcuts.",
        ),
        imports_cell(),
    ]

    cells += section(
        "4.4.1 The Softmax",
        """Softmax converts one row of class logits into probabilities.

It exponentiates each score and divides by the sum of exponentiated scores in that row.""",
        """A classifier needs comparable class probabilities for cross-entropy loss and interpretation.""",
        [
            ("md", "Define softmax for a batch of logits."),
            ("code", """def softmax(logits):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(shifted)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)

logits = torch.tensor([[1.0, 2.0, 0.0]])
softmax(logits)"""),
        ],
        """`logits.max(..., keepdim=True)` finds the largest score in each row.

Subtracting the row maximum improves numerical stability without changing softmax probabilities.

`torch.exp` makes scores positive.

Dividing by the row sum makes each row add to 1.""",
        """Stable softmax is used inside classification losses. PyTorch's built-in cross-entropy combines this idea with log loss efficiently.""",
        """- Softmax is applied across classes, not across examples.
- Subtracting the maximum is a stability trick, not a model change.
- Softmax outputs probabilities but does not choose a class by itself.
- Very large logits can cause numerical problems without stabilization.""",
    )

    cells += section(
        "4.4.2 The Model",
        """The softmax-regression model first computes linear class scores, then converts them to probabilities.

The learnable parameters are a weight matrix and a bias vector.""",
        """The model must produce one class score per example per class. For images, flattened pixels become input features.""",
        [
            ("md", "Define a softmax-regression model from scratch."),
            ("code", """num_inputs = 4
num_outputs = 3
W = torch.randn(num_inputs, num_outputs) * 0.01
W.requires_grad_()
b = torch.zeros(num_outputs, requires_grad=True)
def net(X):
    X = X.reshape(X.shape[0], -1)
    return softmax(X @ W + b)"""),
            ("md", "Run the model on two tiny images."),
            ("code", """X = torch.randn(2, 1, 2, 2)
y_hat = net(X)
y_hat.shape, y_hat.sum(dim=1)"""),
        ],
        """`num_inputs` is the flattened feature count.

`num_outputs` is the number of classes.

`W` maps input features to class scores.

`b` adds one bias per class.

`X.reshape(X.shape[0], -1)` keeps the batch size and flattens each example.

`softmax(X @ W + b)` returns probabilities.""",
        """This is the manual version of a linear classifier followed by softmax.""",
        """- The weight matrix shape is `(inputs, classes)` in this formulation.
- The bias vector has one value per class.
- Flattening must preserve the batch dimension.
- `requires_grad_()` turns gradient tracking on for `W` after its small random initialization.""",
    )

    cells += section(
        "4.4.3 The Cross-Entropy Loss",
        """Cross-entropy selects the predicted probability assigned to the correct class and takes negative log.

For a batch, the loss is usually averaged or summed across examples.""",
        """The loss should become small when the model assigns high probability to the true class.""",
        [
            ("md", "Define cross-entropy from predicted probabilities."),
            ("code", """def cross_entropy(y_hat, y):
    row = torch.arange(y_hat.shape[0])
    return -torch.log(y_hat[row, y])

y_hat = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
y = torch.tensor([1, 0])
cross_entropy(y_hat, y)"""),
        ],
        """`torch.arange(y_hat.shape[0])` creates row indices for the batch.

`y` stores the correct class index for each row.

`y_hat[row, y]` selects the correct-class probability for each example.

`-torch.log(...)` converts those probabilities into losses.""",
        """This manual loss explains what `CrossEntropyLoss` is trying to optimize, even though PyTorch's implementation expects logits and is more stable.""",
        """- The manual version expects probabilities, not logits.
- The label tensor should contain integer class IDs.
- Low correct-class probability gives high loss.
- Logarithm of zero is invalid, so stable implementations avoid exact zero probabilities.""",
    )

    cells += section(
        "4.4.4 Training",
        """Training softmax regression repeats the same pattern as regression: read a batch, predict, compute loss, call backward, update parameters.

The difference is the prediction and loss are classification-specific.""",
        """The from-scratch loop proves that classification training is still ordinary tensor computation plus gradients.""",
        [
            ("md", "A tiny training step on synthetic image-like data."),
            ("code", """X = torch.randn(4, 1, 2, 2)
y = torch.tensor([0, 1, 2, 1])
y_hat = net(X)
loss = cross_entropy(y_hat, y).mean()
loss.backward()
with torch.no_grad():
    W -= 0.1 * W.grad
    b -= 0.1 * b.grad
    W.grad.zero_(); b.grad.zero_()
loss"""),
        ],
        """The batch contains four tiny image-like examples.

`net(X)` returns class probabilities.

`cross_entropy(...).mean()` creates one scalar loss.

`loss.backward()` computes gradients.

The `torch.no_grad()` block updates parameters without tracking the update as part of the graph.

Gradients are cleared after the update.""",
        """This is the same control flow used later with PyTorch optimizers, only written by hand.""",
        """- The loss must be scalar before the simplest `.backward()` call.
- Parameter updates should not be tracked by autograd.
- Gradients must be cleared before the next step.
- This manual loop uses probabilities, while PyTorch's built-in loss usually uses logits.""",
    )

    cells += section(
        "4.4.5 Prediction",
        """Prediction means choosing the class with the highest score or probability.

For softmax outputs, the largest probability corresponds to the predicted class.""",
        """Training optimizes probability quality, but many applications need a final class decision.""",
        [
            ("md", "Predict class IDs from probabilities."),
            ("code", """probs = net(torch.randn(3, 1, 2, 2))
preds = torch.argmax(probs, dim=1)
preds"""),
            ("md", "Map predicted IDs to class names."),
            ("code", """class_names = ["top", "trouser", "shoe"]
names = [class_names[int(i)] for i in preds]
names"""),
        ],
        """`torch.argmax(probs, dim=1)` chooses the largest class probability in each row.

The result is a tensor of class IDs.

The list comprehension maps each ID to a readable class name.""",
        """Prediction code is usually used during validation, testing, and deployment. It should not update parameters.""",
        """- Prediction is different from training.
- `argmax` removes probability confidence information.
- Class-name lists must match the class-ID order.
- Use `dim=1` when columns are classes.""",
    )

    cells += section(
        "4.4.6 Summary",
        """From-scratch softmax regression contains linear logits, softmax probabilities, cross-entropy loss, and gradient updates.""",
        """Seeing the full implementation makes concise PyTorch classifier code easier to audit.""",
        [
            ("md", "The from-scratch classification pipeline."),
            ("code", """pipeline = [
    "flatten inputs",
    "compute logits",
    "softmax probabilities",
    "cross-entropy loss",
    "gradient update",
]
pipeline"""),
        ],
        """The pipeline lists the execution order.

Each step produces values used by the next step.

Training repeats this pipeline over batches.""",
        """More complex classifiers keep the same output-and-loss pattern even when the model body changes.""",
        """- Softmax regression is linear before softmax.
- Cross-entropy is the training signal.
- Accuracy is an evaluation metric, not the differentiable loss.
- Built-in implementations are more stable than naive manual formulas.""",
    )

    cells += section(
        "4.4.7 Exercises",
        """These exercises check the pieces of a manual classifier.""",
        """Debugging classification is easier when logits, probabilities, labels, and predictions are inspected separately.""",
        [
            ("md", "Exercise 1: compute probabilities for two logit rows."),
            ("code", """logits = torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.5, 0.5]])
probs = softmax(logits)
probs.sum(dim=1)"""),
            ("md", "Exercise 2: compute predicted classes."),
            ("code", """preds = torch.argmax(probs, dim=1)
preds"""),
        ],
        """Exercise 1 checks row-wise normalization.

Exercise 2 checks class selection.

The second row has equal scores, so `argmax` returns the first maximum index.""",
        """These same checks are useful when validating a real classifier's outputs.""",
        """- Equal logits create equal probabilities.
- `argmax` tie-breaking can matter.
- Probabilities should be checked along the class dimension.
- Manual softmax is for learning; built-in loss is preferred in real training.""",
    )

    write_nb("Chapter 4.4 - Softmax Regression Implementation from Scratch.ipynb", cells)


def build_45() -> None:
    cells = [
        title_cell(
            "Chapter 4.5 - Concise Implementation of Softmax Regression",
            "The concise implementation uses PyTorch layers, loss functions, optimizers, and data loaders while preserving the same training control flow.",
        ),
        imports_cell(),
    ]

    cells += section(
        "4.5.1 Defining the Model",
        """A concise softmax-regression model is a linear layer that maps flattened inputs to class logits.

The model returns logits. The loss function will handle softmax internally.""",
        """PyTorch layers register parameters automatically, so optimizers can find and update them.""",
        [
            ("md", "Define a linear classifier for tiny 2 by 2 images and three classes."),
            ("code", """net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(4, 3),
)
X = torch.randn(2, 1, 2, 2)
logits = net(X)
logits.shape"""),
        ],
        """`torch.nn.Flatten()` converts each image into a feature vector.

`torch.nn.Linear(4, 3)` maps four pixels to three class logits.

`torch.nn.Sequential` runs the listed layers in order.

The output shape is `(2, 3)`: two examples and three classes.""",
        """This replaces the manual flattening, weight matrix, bias vector, and logits computation.""",
        """- The model returns logits, not probabilities.
- `Sequential` runs layers in the order provided.
- The input size to `Linear` must match flattened feature count.
- Registered parameters can be found with `net.parameters()`.""",
    )

    cells += section(
        "4.5.2 Softmax Revisited",
        """Softmax is still conceptually part of classification, but PyTorch's `CrossEntropyLoss` combines log-softmax and negative log likelihood internally.

Log-softmax means taking logarithms after softmax in a numerically stable way.""",
        """Combining these steps avoids numerical problems and reduces unnecessary computation.""",
        [
            ("md", "Use CrossEntropyLoss directly on logits."),
            ("code", """labels = torch.tensor([0, 2])
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
loss"""),
            ("md", "Softmax is still useful for inspection."),
            ("code", """probs = torch.softmax(logits, dim=1)
probs.sum(dim=1)"""),
        ],
        """`CrossEntropyLoss` receives raw logits and integer labels.

It internally applies a stable log-softmax and computes the correct-class loss.

`torch.softmax` can be used after the model when you want readable probabilities.

The row sums confirm the inspected probabilities add to 1.""",
        """In real PyTorch training, pass logits to `CrossEntropyLoss`. Use softmax mainly for interpretation or prediction confidence.""",
        """- Do not softmax before `CrossEntropyLoss` in the standard case.
- Labels should be class IDs, not one-hot rows.
- Softmax probabilities are useful for inspection.
- Logits can be any real numbers.""",
    )

    cells += section(
        "4.5.3 Training",
        """Concise classification training follows: zero gradients, compute logits, compute cross-entropy loss, call backward, step optimizer.""",
        """The framework shortens the code, but the training order stays the same as the from-scratch version.""",
        [
            ("md", "One compact training epoch on tiny synthetic image data."),
            ("code", """X = torch.randn(8, 1, 2, 2)
y = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=4)
net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.1)
for Xb, yb in loader:
    opt.zero_grad(); loss_fn(net(Xb), yb).backward(); opt.step()
loss_fn(net(X), y)"""),
        ],
        """The DataLoader returns batches of images and labels.

The model produces logits.

`CrossEntropyLoss` computes scalar loss from logits and class IDs.

`backward()` computes gradients.

`opt.step()` updates parameters.""",
        """This is the standard skeleton of many PyTorch classification loops.""",
        """- `zero_grad()` must happen before the next backward pass.
- The model should output one logit per class.
- `CrossEntropyLoss` combines softmax-like behavior with loss computation.
- Synthetic data only checks code structure, not real accuracy.""",
    )

    cells += section(
        "4.5.4 Summary",
        """The concise classifier uses `Flatten`, `Linear`, `CrossEntropyLoss`, `SGD`, and `DataLoader`.""",
        """These objects package the manual softmax-regression implementation into reusable PyTorch components.""",
        [
            ("md", "Map concise objects to manual concepts."),
            ("code", """mapping = {
    "Flatten": "reshape images into vectors",
    "Linear": "compute class logits",
    "CrossEntropyLoss": "stable softmax plus loss",
    "SGD": "update parameters",
}
mapping"""),
        ],
        """Each concise object replaces a manual operation.

The execution order is still the training-loop order.

The mapping is the key to reading concise code without losing the underlying mechanics.""",
        """Later neural classifiers replace only the model body. The output logits and cross-entropy pattern remain common.""",
        """- Concise code should still be explainable line by line.
- The model outputs logits.
- The loss function owns the stable softmax-loss combination.
- Optimizer objects update registered parameters.""",
    )

    cells += section(
        "4.5.5 Exercises",
        """These exercises practice modifying concise classifier dimensions and reading outputs.""",
        """Shape control is the main practical skill for concise classification code.""",
        [
            ("md", "Exercise 1: define a classifier for 4 by 4 grayscale images and five classes."),
            ("code", """net5 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 5))
X5 = torch.randn(3, 1, 4, 4)
logits5 = net5(X5)
logits5.shape"""),
            ("md", "Exercise 2: compute predictions from logits."),
            ("code", """preds = torch.argmax(logits5, dim=1)
preds.shape"""),
        ],
        """Exercise 1 checks flattened input size and class count.

Exercise 2 checks prediction shape.

Three examples should produce three predicted class IDs.""",
        """These checks prevent common mistakes before training on larger image datasets.""",
        """- Flattened size is channels times height times width.
- Output size is number of classes.
- Predictions are class IDs.
- Prediction shape should match label shape.""",
    )

    write_nb("Chapter 4.5 - Concise Implementation of Softmax Regression.ipynb", cells)


def build_46() -> None:
    cells = [
        title_cell(
            "Chapter 4.6 - Generalization in Classification",
            "Generalization in classification asks whether a classifier makes correct decisions on new examples, not only training examples.",
        ),
        imports_cell(),
    ]

    cells += section(
        "4.6.1 The Test Set",
        """A test set is held-out data used to estimate final performance after model choices are made.

A validation set is used during development to choose models or hyperparameters.""",
        """If we evaluate only on training data, we can fool ourselves into believing a memorizing model is useful.""",
        [
            ("md", "Compute train and test accuracy separately."),
            ("code", """train_preds = torch.tensor([0, 1, 1, 0])
train_y = torch.tensor([0, 1, 0, 0])
test_preds = torch.tensor([1, 1])
test_y = torch.tensor([0, 1])
train_acc = (train_preds == train_y).float().mean()
test_acc = (test_preds == test_y).float().mean()
train_acc, test_acc"""),
        ],
        """The training accuracy is computed on examples used for learning.

The test accuracy is computed on held-out examples.

A gap between them can indicate overfitting or distribution mismatch.""",
        """Classification experiments usually report test accuracy only after model selection is complete.""",
        """- Test data should not guide repeated model choices.
- Training accuracy is not final evidence of real performance.
- Test accuracy is still an estimate from a finite sample.
- Validation and test sets have different roles.""",
    )

    cells += section(
        "4.6.2 Test Set Reuse",
        """Test set reuse means checking the test set repeatedly while making model decisions.

This can leak information from the test set into the development process.""",
        """Repeated test-set feedback can make the model selection process overfit the test set, even if gradients never directly use test examples.""",
        [
            ("md", "Toy record of repeated test checks."),
            ("code", """runs = [
    {"model": "A", "test_acc": 0.80},
    {"model": "B", "test_acc": 0.83},
    {"model": "C", "test_acc": 0.81},
]
best_seen = max(runs, key=lambda item: item["test_acc"])
best_seen"""),
        ],
        """The code picks the best test accuracy among several attempts.

That looks harmless, but if each attempt was influenced by previous test results, the test set became part of model selection.

A separate validation set should be used for this process instead.""",
        """Benchmarks and competitions are vulnerable to this issue when many people repeatedly tune against the same public leaderboard.""",
        """- Test labels can influence decisions indirectly.
- Public leaderboard performance can be overfit.
- Use validation data for iteration.
- Keep a final test set for the final estimate.""",
    )

    cells += section(
        "4.6.3 Statistical Learning Theory",
        """Statistical learning theory studies how training performance relates to future performance.

A hypothesis is one possible model function. A hypothesis class is the set of functions a learning algorithm can choose from.""",
        """The bigger or more flexible the hypothesis class, the easier it can fit training data and the harder it may be to guarantee generalization.""",
        [
            ("md", "A tiny comparison of model classes."),
            ("code", """classes = {
    "linear": "fewer possible decision boundaries",
    "deep_network": "many possible decision boundaries",
}
classes"""),
        ],
        """The dictionary is conceptual, not a computation.

A linear classifier has a more restricted set of decision boundaries.

A deep network can represent many more functions.

More flexibility can help fit real structure, but it can also fit noise.""",
        """Generalization theory motivates regularization, validation, data collection, and careful evaluation.""",
        """- Theory gives guidance, not automatic guarantees for every real system.
- More flexible models are not always worse.
- More data can support more complex models.
- Evaluation protocol remains necessary even with theory.""",
    )

    cells += section(
        "4.6.4 Summary",
        """Classification generalization depends on held-out evaluation, careful test-set use, and awareness of model flexibility.""",
        """The goal is reliable performance on new examples, not leaderboard luck or training memorization.""",
        [
            ("md", "A compact evaluation protocol."),
            ("code", """protocol = [
    "train on training data",
    "choose settings on validation data",
    "report final result on test data",
]
protocol"""),
        ],
        """The protocol separates learning, selection, and final evaluation.

This separation reduces the chance of overestimating real performance.

It does not remove all uncertainty, but it improves discipline.""",
        """This protocol becomes more important as classifiers become more powerful and datasets become more reused.""",
        """- Do not optimize for test performance during development.
- Validation performance can also be overfit after many tries.
- Report uncertainty when possible.
- Match evaluation data to the deployment problem.""",
    )

    cells += section(
        "4.6.5 Exercises",
        """These exercises practice separating training, validation, and test decisions.""",
        """Clear data roles prevent misleading evaluation.""",
        [
            ("md", "Exercise 1: choose a model by validation accuracy."),
            ("code", """runs = [{"name": "A", "valid": 0.82}, {"name": "B", "valid": 0.85}]
best = max(runs, key=lambda run: run["valid"])
best"""),
            ("md", "Exercise 2: compute final test accuracy for the chosen model."),
            ("code", """test_preds = torch.tensor([0, 1, 1])
test_y = torch.tensor([0, 0, 1])
test_acc = (test_preds == test_y).float().mean()
test_acc"""),
        ],
        """Exercise 1 uses validation accuracy for selection.

Exercise 2 uses test accuracy only after selection.

The separation is the important point.""",
        """This discipline applies to classification, regression, and later deep learning experiments.""",
        """- Choose by validation data, not test data.
- Final test results should be reported honestly.
- One test estimate has sampling noise.
- Keep records of how many model choices were tried.""",
    )

    write_nb("Chapter 4.6 - Generalization in Classification.ipynb", cells)


def build_47() -> None:
    cells = [
        title_cell(
            "Chapter 4.7 - Environment and Distribution Shift",
            "Distribution shift happens when the data a model sees after deployment differs from the data it learned from. This notebook explains the main shift types and why they matter for classification systems.",
        ),
        imports_cell(),
    ]

    cells += section(
        "4.7.1 Types of Distribution Shift",
        """A distribution describes how data values occur. Distribution shift means the training distribution and deployment distribution are different.

Covariate shift means input features change while the label rule is assumed stable. Label shift means class frequencies change. Concept shift means the relationship between inputs and labels changes.""",
        """Models learn patterns from training data. If the future data follows different patterns, training performance may not predict deployment performance.""",
        [
            ("md", "Represent class frequencies before and after deployment."),
            ("code", """train_counts = torch.tensor([80, 20])
deploy_counts = torch.tensor([50, 50])
train_probs = train_counts / train_counts.sum()
deploy_probs = deploy_counts / deploy_counts.sum()
train_probs, deploy_probs"""),
        ],
        """The training set has class 0 much more often than class 1.

The deployment setting has both classes equally often.

The probability vectors show label frequencies.

This is a toy example of label shift.""",
        """Distribution shift is a central deployment risk. A classifier validated on old data can fail when users, sensors, policies, or environments change.""",
        """- Shift is about data-generating conditions, not just random noise.
- Covariate shift affects inputs.
- Label shift affects class proportions.
- Concept shift affects the input-label relationship itself.""",
    )

    cells += section(
        "4.7.2 Examples of Distribution Shift",
        """Distribution shift appears in practical systems when the world changes or the data collection process changes.

Examples include new camera lighting, new user populations, seasonal demand, policy changes, or adversarial behavior.""",
        """Concrete examples make it easier to identify why a model's validation performance may not match real-world performance.""",
        [
            ("md", "A simple feature mean shift."),
            ("code", """train_feature = torch.tensor([0.1, 0.0, -0.1])
deploy_feature = torch.tensor([1.1, 1.0, 0.9])
train_mean = train_feature.mean()
deploy_mean = deploy_feature.mean()
deploy_mean - train_mean"""),
        ],
        """The training feature values are centered near 0.

The deployment feature values are centered near 1.

The mean difference is a simple signal that inputs changed.

Real shift detection uses richer checks, but the basic idea is comparison.""",
        """Monitoring input statistics is one way production ML systems detect that future data may no longer match training data.""",
        """- A shifted feature does not always mean performance fails.
- Some shifts are harmless; others are severe.
- Shift can happen gradually or suddenly.
- The most dangerous shifts often affect labels or decision rules, not only input averages.""",
    )

    cells += section(
        "4.7.3 Correction of Distribution Shift",
        """Correcting distribution shift means changing training, evaluation, or deployment procedures to reduce mismatch.

Common responses include reweighting examples, collecting new data, retraining, domain adaptation, and monitoring.""",
        """Ignoring shift can make a model confidently wrong in the environment where it matters.""",
        [
            ("md", "Toy reweighting for changed class frequencies."),
            ("code", """train_probs = torch.tensor([0.8, 0.2])
deploy_probs = torch.tensor([0.5, 0.5])
weights = deploy_probs / train_probs
weights"""),
        ],
        """The weights compare deployment class probability to training class probability.

Class 0 gets downweighted because it was overrepresented in training.

Class 1 gets upweighted because it was underrepresented in training.

This is a simplified example, not a complete correction method.""",
        """Reweighting appears in class imbalance, label shift correction, and importance sampling. Importance sampling means estimating a target distribution using weighted samples from another distribution.""",
        """- Correction requires knowing or estimating the shift.
- Reweighting can increase variance when weights are large.
- Retraining may be safer when fresh labeled data is available.
- Monitoring should continue after correction.""",
    )

    cells += section(
        "4.7.4 A Taxonomy of Learning Problems",
        """A taxonomy is a classification system for ideas.

ML problems can be grouped by what feedback is available: supervised learning has labels, unsupervised learning has no labels, self-supervised learning creates labels from data itself, and reinforcement learning uses rewards from actions.""",
        """Different problem types have different assumptions about data, feedback, and shift.""",
        [
            ("md", "Map learning problem types to feedback."),
            ("code", """taxonomy = {
    "supervised": "input-label pairs",
    "unsupervised": "inputs without labels",
    "self_supervised": "labels derived from data",
    "reinforcement": "rewards after actions",
}
taxonomy"""),
        ],
        """The dictionary names four learning settings.

Each value describes the kind of feedback available.

Classification is usually supervised learning because labels identify the correct class.

Other settings require different training objectives.""",
        """Knowing the problem type helps choose evaluation methods and identify likely distribution-shift risks.""",
        """- The same dataset can support different learning setups.
- Supervised learning requires labels.
- Reinforcement learning feedback depends on actions.
- A taxonomy helps reasoning, but real systems can combine categories.""",
    )

    cells += section(
        "4.7.5 Fairness, Accountability, and Transparency in Machine Learning",
        """Fairness asks whether model behavior is unjust across people or groups. Accountability asks who is responsible for model decisions and harms. Transparency asks whether relevant people can understand or inspect the system.

These are sociotechnical issues, meaning they involve both technical design and social context.""",
        """Classification systems can affect loans, jobs, healthcare, education, moderation, and safety. Accuracy alone is not enough when errors have unequal consequences.""",
        [
            ("md", "Compare accuracy across two groups."),
            ("code", """group = torch.tensor([0, 0, 1, 1])
correct = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
acc_group0 = correct[group == 0].mean()
acc_group1 = correct[group == 1].mean()
acc_group0, acc_group1"""),
        ],
        """`group` stores a group ID for each example.

`correct` stores whether each prediction was correct.

Boolean indexing selects examples from one group.

The two group accuracies reveal a performance gap in this tiny example.""",
        """Responsible ML evaluation often reports performance by relevant subgroups, not only overall averages.""",
        """- Fairness cannot be reduced to one universal metric.
- Group labels may be sensitive and require careful governance.
- Transparency does not automatically make a system fair.
- Accountability requires human and organizational decisions, not just code.""",
    )

    cells += section(
        "4.7.6 Summary",
        """Distribution shift and responsible ML concerns remind us that models live in environments, not just notebooks.""",
        """A classifier's usefulness depends on how data is collected, how deployment differs from training, and who is affected by errors.""",
        [
            ("md", "A deployment checklist."),
            ("code", """checks = [
    "compare train and deployment data",
    "monitor performance over time",
    "evaluate important subgroups",
    "plan retraining or rollback",
]
checks"""),
        ],
        """The checklist names practical deployment concerns.

Monitoring checks whether data and performance change.

Subgroup evaluation checks whether averages hide harm.

Retraining or rollback plans prepare for failure.""",
        """These ideas become increasingly important as models move from experiments to real applications.""",
        """- Validation performance is not the end of evaluation.
- Deployment data can change.
- Social impact is part of applied ML quality.
- Monitoring and documentation are engineering responsibilities.""",
    )

    cells += section(
        "4.7.7 Exercises",
        """These exercises practice recognizing shift and evaluating subgroup behavior.""",
        """The goal is to build habits that go beyond average accuracy.""",
        [
            ("md", "Exercise 1: identify a label-frequency shift."),
            ("code", """old = torch.tensor([90, 10])
new = torch.tensor([60, 40])
old_p = old / old.sum()
new_p = new / new.sum()
new_p - old_p"""),
            ("md", "Exercise 2: compute group accuracy gap."),
            ("code", """acc0 = torch.tensor(0.95)
acc1 = torch.tensor(0.75)
gap = acc0 - acc1
gap"""),
        ],
        """Exercise 1 compares class proportions before and after a change.

Exercise 2 computes a simple subgroup accuracy gap.

Both are tiny signals that should trigger deeper investigation in real systems.""",
        """Production classifiers often need monitoring reports that include these kinds of checks.""",
        """- A gap is a warning sign, not a complete fairness audit.
- Shift detection needs domain context.
- Monitoring should be planned before deployment.
- Corrective action may require data, model, and process changes.""",
    )

    write_nb("Chapter 4.7 - Environment and Distribution Shift.ipynb", cells)


def main() -> None:
    build_41()
    build_42()
    build_43()
    build_44()
    build_45()
    build_46()
    build_47()


if __name__ == "__main__":
    main()
