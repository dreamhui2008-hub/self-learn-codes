"""Generate Chapter 3 notebooks for the D2L rewrite project."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Chapter 3 - Linear Neural Networks for Regression"
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



def section(
    title: str,
    intuition: str,
    why: str,
    examples: list[tuple[str, str]],
    breakdown: str,
    connection: str,
    confusion: str,
) -> list[dict]:
    cells = [
        md(
            f"""### {title}

#### 1. Intuition

{intuition.strip()}

#### 2. Why this exists

{why.strip()}"""
        )
    ]
    cells.append(md("#### 3. Examples\n\n" + examples[0][1].strip()))
    for kind, content in examples[1:]:
        cells.append(code(content) if kind == "code" else md(content))
    cells.append(
        md(
            f"""#### 4. Step-by-step breakdown

{breakdown.strip()}

#### 5. Connection to ML systems

{connection.strip()}

#### 6. Common confusion points

{confusion.strip()}"""
        )
    )
    return cells


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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
    path = OUT_DIR / filename
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")


def imports_cell(extra: str = "") -> dict:
    base = "import math\nimport random\nimport numpy as np\nimport torch"
    return code(base + ("\n" + extra.strip() if extra.strip() else ""))


def title_cell(title: str, description: str) -> dict:
    return md(f"### {title}\n\n{description.strip()}")


def build_31() -> None:
    cells = [
        title_cell(
            "Chapter 3.1 - Linear Regression",
            "Linear regression is the simplest supervised learning model for predicting a number from input features. This notebook builds it from plain weighted sums before connecting it to vectorized tensor code and neural network language.",
        ),
        imports_cell(),
    ]

    cells += section(
        "3.1.1 Basics",
        """Regression means predicting a numerical value. Linear regression predicts that value by adding weighted input features plus a bias.

A feature is an input measurement. A label is the target value. A weight says how strongly one feature affects the prediction. A bias is a learned offset added after the weighted features.

For one house, a tiny model might say: predicted price = size weight times size + bedroom weight times bedrooms + bias.""",
        """Linear regression gives us the first complete learning problem with inputs, labels, parameters, predictions, and loss.

It is simple enough to compute by hand, but it already has the same structure as larger neural networks: choose parameters, make predictions, measure errors, and adjust parameters.""",
        [
            ("md", "Manual Python: compute one prediction from two features."),
            ("code", """features = [2.0, 3.0]
weights = [4.0, 1.0]
bias = 5.0
prediction = features[0] * weights[0] + features[1] * weights[1] + bias
prediction"""),
            ("md", "NumPy: the same weighted sum as a dot product."),
            ("code", """x_np = np.array([2.0, 3.0])
w_np = np.array([4.0, 1.0])
b_np = 5.0
prediction_np = np.dot(x_np, w_np) + b_np
prediction_np"""),
            ("md", "PyTorch: the same idea using tensors."),
            ("code", """x = torch.tensor([2.0, 3.0])
w = torch.tensor([4.0, 1.0])
b = torch.tensor(5.0)
prediction = torch.dot(x, w) + b
prediction"""),
        ],
        """The manual expression multiplies each feature by its matching weight.

The products are added together, then the bias is added.

`np.dot(x_np, w_np)` performs the multiply-then-sum pattern for vectors.

`torch.dot(x, w)` does the same thing in PyTorch for 1-dimensional tensors.

At this stage the weights and bias are just chosen numbers. Training will later mean changing those numbers so predictions improve.""",
        """A neural network layer can be viewed as a repeated weighted-sum operation. Linear regression is therefore the smallest useful bridge between ordinary algebra and neural network code.""",
        """- Linear does not mean easy; it means the prediction is a weighted sum of inputs.
- The bias is not attached to one feature. It shifts the whole prediction.
- A feature vector and weight vector must have matching length.
- Prediction is not learning. Learning starts when parameters are adjusted from data.""",
    )

    cells += section(
        "3.1.2 Vectorization for Speed",
        """Vectorization means replacing many explicit Python loops with array or tensor operations.

A batch is a small group of examples processed together. If each row is one example, a batch of examples forms a matrix.""",
        """Python loops are slow for large numerical work. Tensor libraries send the repeated arithmetic to optimized low-level code and hardware.

Vectorization also makes the mathematical intent clearer: one matrix of inputs times one vector of weights gives many predictions.""",
        [
            ("md", "Manual Python: predict two examples with a loop."),
            ("code", """X = [[2.0, 3.0], [1.0, 4.0]]
w = [4.0, 1.0]
b = 5.0
preds = []
for x in X:
    preds.append(x[0] * w[0] + x[1] * w[1] + b)
preds"""),
            ("md", "PyTorch: predict the batch with one matrix-vector product."),
            ("code", """X = torch.tensor([[2.0, 3.0], [1.0, 4.0]])
w = torch.tensor([4.0, 1.0])
b = torch.tensor(5.0)
preds = X @ w + b
preds"""),
        ],
        """The manual loop handles one row at a time.

Each row is a feature vector for one example.

`X @ w` multiplies the input matrix by the weight vector. Each output value is one row's dot product with `w`.

`+ b` uses broadcasting to add the same scalar bias to every prediction.""",
        """Training loops usually process minibatches. A minibatch is a batch used for one parameter update. Vectorization makes minibatch training practical.""",
        """- Vectorization changes how computation is expressed, not the math being computed.
- `X @ w` is matrix-vector multiplication, not elementwise multiplication.
- Batch rows must use the same feature order.
- Broadcasting the bias is intentional here, but broadcasting mistakes can hide shape bugs.""",
    )

    cells += section(
        "3.1.3 The Normal Distribution and Squared Loss",
        """A loss is a number measuring how wrong a prediction is. Squared loss uses the squared difference between prediction and label.

Noise means variation we do not fully explain with features. The normal distribution is a bell-shaped probability distribution often used to model small random errors around zero.""",
        """Squared loss punishes large errors more than small errors because errors are squared.

If we assume labels equal a linear prediction plus normal noise, minimizing squared loss is a natural training objective.""",
        [
            ("md", "Manual Python: compute squared errors and their mean."),
            ("code", """preds = [10.0, 12.0, 15.0]
labels = [11.0, 10.0, 15.0]
errors = [preds[i] - labels[i] for i in range(3)]
squared = [e * e for e in errors]
mean_loss = sum(squared) / len(squared)
mean_loss"""),
            ("md", "PyTorch: the same loss with tensors."),
            ("code", """preds = torch.tensor([10.0, 12.0, 15.0])
labels = torch.tensor([11.0, 10.0, 15.0])
losses = (preds - labels) ** 2
loss = losses.mean()
loss"""),
        ],
        """`preds[i] - labels[i]` computes one prediction error.

Squaring makes negative and positive errors both positive.

The mean turns per-example losses into one scalar loss.

In PyTorch, `preds - labels` computes all errors elementwise. `** 2` squares each error. `.mean()` reduces the vector into one scalar.""",
        """Most training code needs one scalar loss before calling `backward()`. Squared loss is the standard first regression loss because it is simple, smooth, and easy to differentiate.""",
        """- Squared loss is not the same as absolute error.
- Squaring makes large errors disproportionately expensive.
- A small mean loss can hide one very bad example.
- The normal-noise story is a modeling assumption, not a guarantee about real data.""",
    )

    cells += section(
        "3.1.4 Linear Regression as a Neural Network",
        """A neural network is a parameterized function built from layers. A layer is a reusable computation block.

Linear regression can be seen as a neural network with one linear layer and no hidden layers. A hidden layer is an intermediate layer between inputs and final outputs.""",
        """This viewpoint matters because it shows that neural networks are not a separate universe. They generalize the same weighted-sum pattern.

Understanding linear regression as a layer makes later PyTorch modules easier to read.""",
        [
            ("md", "PyTorch layer version: one linear layer maps two inputs to one output."),
            ("code", """layer = torch.nn.Linear(2, 1)
X = torch.tensor([[2.0, 3.0]])
y_hat = layer(X)
y_hat.shape"""),
            ("md", "Inspect the parameters stored by the layer."),
            ("code", """weight_shape = layer.weight.shape
bias_shape = layer.bias.shape
weight_shape, bias_shape"""),
        ],
        """`torch.nn.Linear(2, 1)` creates a layer with 2 input features and 1 output value.

`layer(X)` calls the layer on input data. The layer computes a weighted sum plus bias internally.

`layer.weight` stores the learnable weights. `layer.bias` stores the learnable bias.

The initial values are random or default-initialized because training has not happened yet.""",
        """Later models will stack many layers. The first step is recognizing that even a tiny layer contains parameters and a forward computation.""",
        """- A layer object stores parameters; it is not just a formula written on paper.
- Calling `layer(X)` runs the layer's forward computation.
- The layer's output is not meaningful before training.
- Linear regression has no hidden layer, but it can still be described with neural network language.""",
    )

    cells += section(
        "3.1.5 Summary",
        """Linear regression predicts numbers with weighted sums.

The core objects are features, labels, weights, bias, predictions, and loss.""",
        """This chapter gives the smallest complete supervised learning system. Every later training example reuses these pieces with more complex models or data.""",
        [
            ("md", "A compact linear regression prediction and loss."),
            ("code", """X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
w = torch.tensor([2.0, -1.0])
b = torch.tensor(0.5)
y = torch.tensor([0.0, 2.0])
y_hat = X @ w + b
loss = ((y_hat - y) ** 2).mean()
loss"""),
        ],
        """`X @ w + b` computes one prediction per row.

`y_hat - y` computes prediction errors.

Squaring and averaging create a scalar loss.

The code is small, but it contains the basic structure of model training.""",
        """A training system repeatedly computes this kind of loss, asks automatic differentiation for gradients, and updates parameters.""",
        """- Linear regression is a full learning problem, not just a formula.
- Shape tracking is part of correctness.
- Loss defines what the model is trying to improve.
- Parameters are useful only after they are fit to data.""",
    )

    cells += section(
        "3.1.6 Exercises",
        """Exercises should make you predict shapes, values, and loss behavior before relying on the framework.""",
        """Linear regression is simple enough that every number can be inspected. That makes it ideal for debugging your mental model.""",
        [
            ("md", "Exercise 1: compute predictions for three examples."),
            ("code", """X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
w = torch.tensor([3.0, 4.0])
b = torch.tensor(1.0)
y_hat = X @ w + b
y_hat"""),
            ("md", "Exercise 2: compute mean squared loss."),
            ("code", """y = torch.tensor([4.0, 5.0, 15.0])
loss = ((y_hat - y) ** 2).mean()
loss"""),
        ],
        """Exercise 1 checks matrix-vector multiplication and bias broadcasting.

Exercise 2 checks the squared-loss formula.

If you can compute the first prediction by hand, the tensor version becomes less mysterious.""",
        """These exact pieces appear in later from-scratch training loops and concise PyTorch implementations.""",
        """- Check feature order before interpreting a weight.
- Check prediction shape before computing loss.
- Squared loss returns a scalar only after reduction.
- A low loss for one tiny batch does not prove generalization.""",
    )

    write_nb("Chapter 3.1 - Linear Regression.ipynb", cells)


def build_32() -> None:
    cells = [
        title_cell(
            "Chapter 3.2 - Object-Oriented Design for Implementation",
            "Object-oriented design organizes code into objects that carry data and behavior together. This notebook introduces the Python pieces slowly before mapping them to ML training abstractions.",
        ),
        imports_cell(),
    ]

    cells += section(
        "3.2.1 Utilities",
        """A utility is a small helper that supports the main workflow. It is not the model itself.

Object-oriented programming organizes related data and functions into objects. A class is a template. An object is one concrete instance made from that template. An attribute is data stored on an object. A method is a function attached to an object.

`self` is the conventional name for the object receiving a method call.""",
        """ML projects repeatedly need helpers for saving settings, tracking metrics, plotting, batching, and training. Classes can group these helpers so the code has fewer loose variables.""",
        [
            ("md", "Plain Python class: store and update a running average."),
            ("code", """class Average:
    def __init__(self):
        self.total = 0.0
        self.count = 0
    def add(self, value):
        self.total += value
        self.count += 1
        return self.total / self.count

meter = Average()
meter.add(3.0)"""),
            ("md", "Use the same object again, so its attributes remember state."),
            ("code", """first = meter.add(5.0)
second = meter.add(7.0)
first, second, meter.total, meter.count"""),
        ],
        """`class Average:` defines a class template.

`__init__` runs when a new object is created.

`self.total` and `self.count` are attributes stored on the object.

`meter = Average()` creates one object.

`meter.add(3.0)` calls the method. Python passes `meter` as `self` automatically.

The object remembers previous calls because its attributes remain in memory.""",
        """Training code often uses metric accumulators like this to track average loss or accuracy across batches.""",
        """- `self` is not a keyword, but using that name is standard Python style.
- `__init__` initializes object state; it is not called manually in normal use.
- Methods can change attributes, so method calls may have lasting effects.
- Utility classes should make state clearer, not hide important behavior.""",
    )

    cells += section(
        "3.2.2 Models",
        """A model is a function with parameters learned from data.

In object-oriented code, a model object usually stores parameters as attributes and provides a method for computing predictions.""",
        """Keeping parameters and prediction logic together reduces mistakes. If weights and bias are separate loose variables, it is easier to pass the wrong one into a function.""",
        [
            ("md", "Manual model object without PyTorch inheritance."),
            ("code", """class LinearModel:
    def __init__(self):
        self.w = torch.tensor([2.0, -1.0])
        self.b = torch.tensor(0.5)
    def predict(self, X):
        return X @ self.w + self.b

model = LinearModel()
model.predict(torch.tensor([[1.0, 2.0]]))"""),
            ("md", "PyTorch module version. Inheritance means a class reuses behavior from a parent class."),
            ("code", """class TorchLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
    def forward(self, X):
        return self.layer(X)

net = TorchLinear()
net(torch.tensor([[1.0, 2.0]]))"""),
        ],
        """`LinearModel` is a plain Python class that stores `w` and `b`.

`predict` computes the forward prediction formula.

`class TorchLinear(torch.nn.Module)` means `TorchLinear` inherits from PyTorch's `Module` class.

`super().__init__()` runs the parent class initialization. This lets PyTorch set up internal machinery for parameter tracking.

`forward` defines the computation PyTorch should run when the object is called like `net(X)`.""",
        """PyTorch models are usually subclasses of `torch.nn.Module`. The parent class manages parameters, nested layers, device movement, and training/evaluation mode.""",
        """- Inheritance reuses parent behavior; it is not copying code by hand.
- `super().__init__()` is required so PyTorch can initialize module internals.
- `forward` is called indirectly when you write `net(X)`.
- A model object can contain other layer objects as attributes.""",
    )

    cells += section(
        "3.2.3 Data",
        """A data object can store features and labels and provide examples when training code asks for them.

An index is a position used to select one item. A dataset often supports `dataset[i]` to return the `i`th example.""",
        """Training code should not care whether data came from a CSV file, generated tensors, or images on disk. A dataset object gives the training loop a consistent interface.""",
        [
            ("md", "Plain dataset object with `__len__` and `__getitem__`."),
            ("code", """class TinyData:
    def __init__(self):
        self.X = torch.tensor([[1.0], [2.0], [3.0]])
        self.y = torch.tensor([2.0, 4.0, 6.0])
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

data = TinyData()
data[1]"""),
            ("md", "Create small batches manually from the dataset."),
            ("code", """batch_X = torch.stack([data[0][0], data[1][0]])
batch_y = torch.stack([data[0][1], data[1][1]])
batch_X, batch_y"""),
        ],
        """`__len__` is a special method Python calls when you use `len(data)`.

`__getitem__` is a special method Python calls when you use square brackets like `data[1]`.

The dataset returns one feature tensor and one label tensor.

`torch.stack` combines several tensors into a new tensor with a batch dimension.""",
        """PyTorch's `Dataset` and `DataLoader` use these same ideas. A DataLoader repeatedly asks a Dataset for examples and groups them into batches.""",
        """- `__getitem__` receives an index, not a feature value.
- A dataset returns examples; a dataloader returns batches.
- `stack` creates a new dimension; it is different from concatenating along an existing dimension.
- Data abstractions should preserve the pairing between each input and its label.""",
    )

    cells += section(
        "3.2.4 Training",
        """Training means repeatedly changing model parameters to reduce loss.

An epoch is one pass through the training dataset. A batch is the group of examples used in one update. An optimizer is the rule that updates parameters using gradients.""",
        """The training loop is the control center of supervised learning. It defines what runs first, what repeats, what is stored, and when parameters change.""",
        [
            ("md", "A minimal training step written directly."),
            ("code", """w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
X = torch.tensor([[1.0], [2.0]])
y = torch.tensor([2.0, 4.0])
y_hat = X.reshape(-1) * w + b
loss = ((y_hat - y) ** 2).mean()
loss.backward()
w.grad, b.grad"""),
            ("md", "A trainer-like object can group loop settings."),
            ("code", """class TrainerConfig:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

config = TrainerConfig(lr=0.03, epochs=3)
config.lr, config.epochs"""),
        ],
        """The prediction is computed first.

The loss compares predictions with labels.

`loss.backward()` computes gradients for `w` and `b`.

The code stops before updating parameters so the gradient values are visible.

`TrainerConfig` is not a full trainer. It only shows how training settings can be stored in an object.""",
        """D2L-style `Trainer` objects eventually package the repeated training workflow. Before using them, understand the raw sequence: forward pass, loss, backward pass, parameter update, repeat.""",
        """- An epoch is not one gradient update unless the dataset has one batch.
- Gradients are computed before the optimizer update.
- Training objects should not hide concepts you cannot yet explain manually.
- Framework trainers are convenience layers over ordinary control flow.""",
    )

    cells += section(
        "3.2.5 Summary",
        """Object-oriented design groups related state and behavior.

For ML, common objects include models, datasets, metric trackers, and trainers.""",
        """The point is not to make code look fancy. The point is to reduce accidental complexity as projects grow.""",
        [
            ("md", "A tiny object map for a training system."),
            ("code", """objects = {
    "model": "stores parameters and predicts",
    "data": "returns examples or batches",
    "trainer": "runs repeated updates",
}
objects"""),
        ],
        """Each object has a responsibility.

The model handles prediction.

The data object handles example access.

The trainer handles repetition and updates.

Keeping these responsibilities separate makes later code easier to inspect.""",
        """Research code frequently defines custom modules, data objects, and training utilities. Reading them is easier when you can identify each object's responsibility.""",
        """- Classes are tools for organization, not automatically better code.
- Hidden state can confuse debugging if not named clearly.
- `self` marks object state that persists across method calls.
- Framework abstractions should be mapped back to the manual training loop.""",
    )

    cells += section(
        "3.2.6 Exercises",
        """These exercises check whether object-oriented code still feels like ordinary Python control flow.""",
        """If a class confuses you, reduce it to attributes, methods, and method-call order.""",
        [
            ("md", "Exercise 1: create a metric object and call it twice."),
            ("code", """meter = Average()
a = meter.add(2.0)
b = meter.add(6.0)
a, b"""),
            ("md", "Exercise 2: inspect a PyTorch module's parameters."),
            ("code", """net = TorchLinear()
params = list(net.parameters())
[p.shape for p in params]"""),
        ],
        """Exercise 1 checks persistent object state.

Exercise 2 checks that PyTorch modules can reveal their trainable parameters.

The object-oriented interface is useful only if you can still trace what data is stored and what computation runs.""",
        """Later chapters will rely on module and trainer abstractions. These exercises prepare you to read them without treating them as magic.""",
        """- Calling a method may change object state.
- `parameters()` comes from PyTorch's parent `Module` class.
- The order of method calls matters in training systems.
- If an abstraction feels unclear, write the manual version beside it.""",
    )

    write_nb("Chapter 3.2 - Object-Oriented Design for Implementation.ipynb", cells)



def build_33() -> None:
    cells = [
        title_cell(
            "Chapter 3.3 - Synthetic Regression Data",
            "Synthetic data is data we create ourselves. It lets us test whether a learning algorithm can recover a known pattern before trusting it on messy real data.",
        ),
        imports_cell(),
    ]

    cells += section(
        "3.3.1 Generating the Dataset",
        """Synthetic regression data is generated from a known formula plus noise.

A true parameter is the parameter used to create the fake labels. Noise is random variation added so the task is not perfectly clean.""",
        """If we know the true weights and bias, we can check whether training recovers them. This is a controlled test of the learning pipeline.""",
        [
            ("md", "Manual Python: create three labels from one feature."),
            ("code", """x_values = [1.0, 2.0, 3.0]
true_w = 2.0
true_b = 1.0
labels = []
for x in x_values:
    labels.append(true_w * x + true_b)
labels"""),
            ("md", "PyTorch: generate a small noisy linear dataset."),
            ("code", """torch.manual_seed(0)
X = torch.randn(5, 2)
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
noise = torch.randn(5) * 0.01
y = X @ true_w + true_b + noise
X, y"""),
        ],
        """The manual example uses an exact line with no noise.

`torch.manual_seed(0)` makes the random numbers repeatable for debugging.

`torch.randn(5, 2)` creates 5 examples with 2 features each.

`X @ true_w + true_b` creates clean labels from the true linear rule.

`noise` adds tiny random errors so the data looks more realistic.""",
        """Synthetic data is a unit test for training code. If a model cannot learn a simple generated line, the bug is probably in the implementation rather than the dataset.""",
        """- Synthetic data is useful because the true answer is known.
- Noise makes exact recovery impossible, but close recovery should still happen.
- A random seed helps reproduce debugging results.
- Synthetic success does not guarantee real-world success.""",
    )

    cells += section(
        "3.3.2 Reading the Dataset",
        """Reading a dataset means repeatedly selecting examples and labels for training.

A minibatch is a small group of examples used for one update. Shuffling means changing example order randomly.""",
        """Training on all examples at once can be expensive. Training one example at a time can be noisy. Minibatches balance efficiency and randomness.""",
        [
            ("md", "Manual minibatch reader using shuffled indices."),
            ("code", """indices = list(range(len(y)))
random.shuffle(indices)
batch_size = 2
batch_idx = indices[:batch_size]
batch_X = X[batch_idx]
batch_y = y[batch_idx]
batch_X.shape, batch_y.shape"""),
            ("md", "A tiny generator that yields batches."),
            ("code", """def data_iter(X, y, batch_size):
    indices = list(range(len(y)))
    random.shuffle(indices)
    for start in range(0, len(y), batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]

next(data_iter(X, y, 2))"""),
        ],
        """`range(len(y))` creates one index per label.

`random.shuffle(indices)` changes the order in place.

`batch_idx` selects the first few shuffled positions.

`X[batch_idx]` selects matching input rows.

`yield` makes a generator. A generator returns one item at a time when asked, instead of building all batches at once.""",
        """A DataLoader is the framework version of this idea. It repeatedly returns batches while handling shuffling and batching details.""",
        """- Shuffle indices, not features and labels separately.
- Batch size controls how many examples are in one update.
- The last batch may be smaller than the others.
- A generator pauses after `yield` and resumes when the next item is requested.""",
    )

    cells += section(
        "3.3.3 Concise Implementation of the Data Loader",
        """A concise implementation uses framework data utilities instead of a handwritten batch generator.

In PyTorch, a `TensorDataset` stores aligned tensors. A `DataLoader` groups dataset items into batches.""",
        """Framework data loaders reduce repeated boilerplate and handle common cases correctly, such as shuffling and batching.""",
        [
            ("md", "PyTorch DataLoader version."),
            ("code", """dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
for batch_X, batch_y in loader:
    first_shapes = batch_X.shape, batch_y.shape
    break
first_shapes"""),
        ],
        """`TensorDataset(X, y)` stores features and labels together by row position.

`DataLoader(..., batch_size=2, shuffle=True)` creates an object that can produce shuffled batches.

The `for` loop asks the loader for one batch at a time.

The `break` stops after the first batch so the example stays small.""",
        """Most PyTorch training code uses DataLoader objects. The important mapping is simple: DataLoader returns the same kind of `(batch_X, batch_y)` pairs as our manual generator.""",
        """- DataLoader is an iterator-like object, not the data itself.
- TensorDataset assumes tensors are aligned along the first dimension.
- Shuffling changes order, not values.
- Concise code should still be explainable using the manual version.""",
    )

    cells += section(
        "3.3.4 Summary",
        """Synthetic data gives a controlled setting for debugging linear regression.

The workflow is: choose true parameters, generate features, generate labels, add noise, then read data in batches.""",
        """A known data-generating process makes implementation mistakes easier to find.""",
        [
            ("md", "One compact synthetic-data function."),
            ("code", """def synthetic_data(w, b, n):
    X = torch.randn(n, len(w))
    y = X @ w + b
    y += torch.randn(n) * 0.01
    return X, y

X_small, y_small = synthetic_data(torch.tensor([2.0, -1.0]), 0.5, 4)
X_small.shape, y_small.shape"""),
        ],
        """The function receives true weights, true bias, and number of examples.

It creates feature rows using random normal values.

It computes labels from the linear rule and adds noise.

It returns features and labels as tensors.""",
        """This function can feed later from-scratch and concise implementations, letting both versions learn from the same controlled data.""",
        """- A data generator is not a trained model.
- The true weights create labels; learned weights try to recover them.
- Noise prevents perfect zero loss.
- Always check returned shapes.""",
    )

    cells += section(
        "3.3.5 Exercises",
        """The exercises test whether you can create and read a synthetic regression dataset deliberately.""",
        """Synthetic datasets are most useful when you can change one part and predict the effect.""",
        [
            ("md", "Exercise 1: generate data with three input features."),
            ("code", """true_w = torch.tensor([1.0, -2.0, 0.5])
X3, y3 = synthetic_data(true_w, b=1.5, n=6)
X3.shape, y3.shape"""),
            ("md", "Exercise 2: read one minibatch from the generated data."),
            ("code", """loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X3, y3), batch_size=3, shuffle=True)
batch_X, batch_y = next(iter(loader))
batch_X.shape, batch_y.shape"""),
        ],
        """Exercise 1 checks that feature count controls the second dimension of `X`.

Exercise 2 checks DataLoader batching.

The first dimension is the number of examples, and the second dimension is the number of features.""",
        """These skills support all upcoming training notebooks, where data must arrive as correctly shaped minibatches.""",
        """- `n` controls rows, not columns.
- `len(w)` controls feature count.
- `iter(loader)` creates an iterator over batches.
- `next(...)` asks for one batch from that iterator.""",
    )

    write_nb("Chapter 3.3 - Synthetic Regression Data.ipynb", cells)


def build_34() -> None:
    cells = [
        title_cell(
            "Chapter 3.4 - Linear Regression Implementation from Scratch",
            "This notebook implements linear regression training directly with tensors. The goal is to see every moving part before using concise framework shortcuts.",
        ),
        imports_cell(),
        code("""def data_iter(X, y, batch_size):
    indices = list(range(len(y)))
    random.shuffle(indices)
    for start in range(0, len(y), batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]"""),
    ]

    cells += section(
        "3.4.1 Defining the Model",
        """Defining the model means writing the function that maps input features to predictions.

For linear regression, the model is a weighted sum plus a bias: `X @ w + b`.""",
        """Training cannot begin until the model's forward computation is clear. The forward computation is the path from inputs to predictions.""",
        [
            ("md", "Manual Python prediction for one example."),
            ("code", """x = [1.0, 2.0]
w = [0.1, -0.2]
b = 0.0
pred = x[0] * w[0] + x[1] * w[1] + b
pred"""),
            ("md", "Tensor function for a batch of examples."),
            ("code", """def linreg(X, w, b):
    return X @ w + b

X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
w = torch.tensor([0.1, -0.2])
b = torch.tensor(0.0)
linreg(X, w, b)"""),
        ],
        """The manual code computes one weighted sum.

`linreg` generalizes the same formula to a batch.

`X @ w` gives one prediction per row of `X`.

`+ b` broadcasts the scalar bias to every prediction.""",
        """This model function is the from-scratch version of what `torch.nn.Linear` will later do internally.""",
        """- The model function should not compute loss.
- Weights must match the number of input features.
- The output shape should match the label shape.
- Initialization values are not learned yet; they are just starting points.""",
    )

    cells += section(
        "3.4.2 Defining the Loss Function",
        """A loss function compares predictions with labels and returns a number representing error.

Squared loss is `(prediction - label)^2`, often averaged across a batch.""",
        """The loss defines what training tries to reduce. Different losses mean different learning goals.""",
        [
            ("md", "Manual squared loss for two predictions."),
            ("code", """preds = [1.0, 3.0]
labels = [2.0, 1.0]
losses = [(preds[i] - labels[i]) ** 2 for i in range(2)]
sum(losses) / len(losses)"""),
            ("md", "Tensor squared loss."),
            ("code", """def squared_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

y_hat = torch.tensor([1.0, 3.0])
y = torch.tensor([2.0, 1.0])
squared_loss(y_hat, y)"""),
        ],
        """The manual version computes one squared error per example.

The tensor version returns per-example losses.

The division by 2 is a convenience because it cancels a factor of 2 when differentiating the square.

A training loop usually reduces these losses with `.mean()` or `.sum()`.""",
        """PyTorch loss modules often combine per-example computation and reduction. Writing it manually first makes the reduction choice visible.""",
        """- Per-example losses are not the same as one scalar loss.
- Dividing by 2 changes scale, not the location of the minimum.
- The loss function should compare aligned predictions and labels.
- Shape mismatch can accidentally trigger broadcasting.""",
    )

    cells += section(
        "3.4.3 Defining the Optimization Algorithm",
        """An optimization algorithm updates parameters to reduce loss.

Stochastic gradient descent, or SGD, updates each parameter by moving it opposite its gradient. A gradient is the slope of the loss with respect to a parameter.""",
        """The model learns only when parameters change. Gradients tell us direction. The learning rate controls step size.""",
        [
            ("md", "Manual SGD update for one scalar parameter."),
            ("code", """w = 3.0
grad = 4.0
lr = 0.1
new_w = w - lr * grad
new_w"""),
            ("md", "Tensor SGD update for a list of parameters."),
            ("code", """def sgd(params, lr, batch_size):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()"""),
        ],
        """`new_w = w - lr * grad` moves the parameter opposite the gradient.

`with torch.no_grad()` prevents PyTorch from tracking the update itself.

The loop visits each parameter tensor.

`p.grad.zero_()` clears old gradients so the next backward pass starts fresh.

Dividing by `batch_size` keeps update scale consistent when the loss was summed.""",
        """Framework optimizers automate this same pattern. They store parameter references and update them after `backward()` computes gradients.""",
        """- Optimizers update parameters, not predictions directly.
- Learning rate too large can make training unstable.
- Gradients accumulate unless cleared.
- Parameter updates should usually happen inside `torch.no_grad()`.""",
    )

    cells += section(
        "3.4.4 Training",
        """Training repeats a sequence: get a batch, predict, compute loss, compute gradients, update parameters.

What changes each iteration are the parameter values and the current batch.""",
        """The training loop turns model definition, loss, and optimization into learning. Without the loop, the pieces do not improve.""",
        [
            ("md", "A compact from-scratch training loop on synthetic data."),
            ("code", """X = torch.randn(20, 2)
y = X @ torch.tensor([2.0, -3.4]) + 4.2 + torch.randn(20) * 0.01
w = torch.randn(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(3):
    for batch_X, batch_y in data_iter(X, y, 5):
        loss = squared_loss(linreg(batch_X, w, b), batch_y).sum()
        loss.backward()
        sgd([w, b], lr=0.03, batch_size=5)
w, b"""),
        ],
        """The true parameters create synthetic labels.

`w` and `b` are learnable parameters because they have `requires_grad=True`.

The outer loop repeats epochs.

The inner loop reads one minibatch at a time.

For each minibatch, the code computes loss, calls `backward()`, and updates parameters with SGD.""",
        """This is the essential PyTorch training loop without optimizer classes or module abstractions. Later concise code compresses the same sequence.""",
        """- `backward()` must happen before the SGD update.
- Gradients must be cleared after each update.
- The current batch changes inside the inner loop.
- Learned parameters should move toward true parameters on this synthetic task, but noise prevents exact equality.""",
    )

    cells += section(
        "3.4.5 Summary",
        """From-scratch linear regression has four core parts: model, loss, optimizer, and training loop.""",
        """Writing these parts manually builds the mental model needed to understand framework abstractions.""",
        [
            ("md", "The training sequence as explicit names."),
            ("code", """steps = [
    "read batch",
    "predict",
    "compute loss",
    "backward",
    "update parameters",
]
steps"""),
        ],
        """The list records the order of operations.

Every training iteration follows this order.

If the order changes accidentally, training may fail or silently behave incorrectly.""",
        """Most deep learning systems are variations of this same sequence, even when wrapped in classes and library calls.""",
        """- A clear training loop is more important than a clever one.
- Loss scale affects gradient scale.
- Parameter initialization affects the starting point.
- Debug from-scratch code with tiny synthetic data first.""",
    )

    cells += section(
        "3.4.6 Exercises",
        """These exercises make you inspect and modify the training loop.""",
        """Small changes reveal which parts of training control speed, stability, and correctness.""",
        [
            ("md", "Exercise 1: run one prediction with current parameters."),
            ("code", """sample_X = torch.tensor([[1.0, 2.0]])
linreg(sample_X, w.detach(), b.detach())"""),
            ("md", "Exercise 2: compute one batch loss without updating."),
            ("code", """batch_X, batch_y = next(data_iter(X, y, 5))
y_hat = linreg(batch_X, w.detach(), b.detach())
squared_loss(y_hat, batch_y).mean()"""),
        ],
        """Exercise 1 checks the model function after training.

Exercise 2 separates evaluation from updating. `detach()` gives values without tracking gradients.

This separation helps debug whether prediction and loss work before training changes anything.""",
        """Evaluation code should usually avoid building gradient graphs because it does not update parameters.""",
        """- `detach()` stops gradient tracking for that value.
- Evaluation loss is not a parameter update.
- A single batch loss is noisy.
- Always know whether your code is training or only measuring.""",
    )

    write_nb("Chapter 3.4 - Linear Regression Implementation from Scratch.ipynb", cells)


def build_35() -> None:
    cells = [
        title_cell(
            "Chapter 3.5 - Concise Implementation of Linear Regression",
            "The concise implementation uses PyTorch objects for the model, loss, optimizer, and data loader. The goal is to map each shortcut back to the manual training loop.",
        ),
        imports_cell(),
    ]

    cells += section(
        "3.5.1 Defining the Model",
        """A concise model uses a framework layer instead of manually storing `w` and `b`.

`torch.nn.Linear` is a PyTorch layer that computes a linear transformation: input times weights plus bias.""",
        """Framework layers reduce boilerplate and automatically register parameters so optimizers can find them.""",
        [
            ("md", "Create a one-output linear layer for two input features."),
            ("code", """net = torch.nn.Linear(2, 1)
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_hat = net(X)
y_hat.shape"""),
            ("md", "Inspect the parameter shapes."),
            ("code", """weight_shape = net.weight.shape
bias_shape = net.bias.shape
weight_shape, bias_shape"""),
        ],
        """`torch.nn.Linear(2, 1)` expects 2 input features and produces 1 output per example.

`net(X)` runs the layer's forward computation.

The output has shape `(2, 1)` because there are 2 examples and 1 output per example.

`net.weight` and `net.bias` are learnable parameters registered inside the layer.""",
        """This replaces the manual `linreg(X, w, b)` function and separate parameter tensors.""",
        """- A layer initializes parameters before training.
- Registered parameters can be found by `net.parameters()`.
- Output shape `(batch_size, 1)` may differ from label shape `(batch_size,)`.
- Concise does not mean conceptually different.""",
    )

    cells += section(
        "3.5.2 Defining the Loss Function",
        """A loss module is a framework object that computes loss.

Mean squared error loss measures average squared difference between predictions and labels.""",
        """Using a built-in loss avoids rewriting common formulas and gives consistent reduction options.""",
        [
            ("md", "Use PyTorch's mean squared error loss."),
            ("code", """loss_fn = torch.nn.MSELoss()
y_hat = torch.tensor([[1.0], [3.0]])
y = torch.tensor([[2.0], [1.0]])
loss = loss_fn(y_hat, y)
loss"""),
        ],
        """`torch.nn.MSELoss()` creates a loss object.

Calling `loss_fn(y_hat, y)` compares predictions and labels.

By default, this returns the mean squared error as one scalar.

The prediction and label shapes are both `(2, 1)`, so no accidental broadcasting is needed.""",
        """This replaces the manual squared-loss function and explicit `.mean()` reduction.""",
        """- Built-in losses still need correctly shaped inputs.
- MSE means mean squared error.
- A loss object can hide reduction details, so check defaults.
- The loss is still just a tensor computed from predictions and labels.""",
    )

    cells += section(
        "3.5.3 Defining the Optimization Algorithm",
        """A PyTorch optimizer stores references to parameters and updates them using gradients.

SGD is stochastic gradient descent, the same update idea used in the from-scratch version.""",
        """The optimizer object handles repeated parameter updates and gradient-clearing patterns more reliably than handwritten update code.""",
        [
            ("md", "Create an SGD optimizer for the model parameters."),
            ("code", """net = torch.nn.Linear(2, 1)
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
list(net.parameters())[0].shape"""),
        ],
        """`net.parameters()` returns the trainable parameters stored in the layer.

`torch.optim.SGD(...)` creates an optimizer that will update those parameters.

`lr=0.03` sets the learning rate.

The optimizer does not update anything until `.step()` is called after gradients exist.""",
        """This replaces the manual `sgd([w, b], lr, batch_size)` function.""",
        """- Creating an optimizer does not train the model.
- The optimizer needs parameters, not predictions.
- `.step()` updates parameters.
- `.zero_grad()` clears old gradients before the next backward pass.""",
    )

    cells += section(
        "3.5.4 Training",
        """Concise training still follows the same order: batch, prediction, loss, backward, update.

The difference is that PyTorch objects perform some details for us.""",
        """This version is closer to real PyTorch code while still small enough to trace line by line.""",
        [
            ("md", "Train a linear layer on synthetic data."),
            ("code", """torch.manual_seed(0)
X = torch.randn(20, 2)
y = (X @ torch.tensor([2.0, -3.4]) + 4.2).reshape(-1, 1)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=5, shuffle=True)
net = torch.nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
for Xb, yb in loader:
    trainer.zero_grad(); loss_fn(net(Xb), yb).backward(); trainer.step()
loss_fn(net(X), y)"""),
        ],
        """The synthetic labels follow a known linear rule.

The DataLoader returns minibatches.

`trainer.zero_grad()` clears gradients from the previous batch.

`loss_fn(net(Xb), yb)` computes the scalar batch loss.

`.backward()` computes gradients.

`trainer.step()` updates the layer parameters.""",
        """This is the practical PyTorch version of the from-scratch loop. The conceptual order is unchanged.""",
        """- Semicolons keep this tiny example compact, but production code should usually use separate lines for readability.
- One pass over the loader is one epoch here.
- Built-in optimizers still require explicit gradient clearing.
- A final loss after one epoch may still be high.""",
    )

    cells += section(
        "3.5.5 Summary",
        """The concise implementation replaces manual pieces with PyTorch objects.

Model: `torch.nn.Linear`. Loss: `torch.nn.MSELoss`. Optimizer: `torch.optim.SGD`. Data: `DataLoader`.""",
        """These objects reduce boilerplate once the manual loop is understood.""",
        [
            ("md", "Map concise objects to from-scratch pieces."),
            ("code", """mapping = {
    "torch.nn.Linear": "linreg plus parameters",
    "torch.nn.MSELoss": "squared_loss plus reduction",
    "torch.optim.SGD": "sgd update function",
    "DataLoader": "data_iter function",
}
mapping"""),
        ],
        """Each PyTorch object corresponds to a manual concept.

The concise version is shorter because common patterns are packaged.

The package does not remove the underlying sequence of computation.""",
        """This mapping is the main skill needed to read higher-level training code without losing the execution flow.""",
        """- Framework objects are wrappers around concepts you already saw manually.
- Shorter code can be harder to debug if you cannot map it back.
- Always know which object owns parameters.
- Always know when gradients are computed and cleared.""",
    )

    cells += section(
        "3.5.6 Exercises",
        """These exercises check whether you can modify concise PyTorch training code safely.""",
        """Real work often means changing model dimensions, learning rates, batch sizes, or loss reductions.""",
        [
            ("md", "Exercise 1: create a linear model with three input features."),
            ("code", """net3 = torch.nn.Linear(3, 1)
X3 = torch.randn(4, 3)
y3_hat = net3(X3)
y3_hat.shape"""),
            ("md", "Exercise 2: inspect optimizer parameter groups."),
            ("code", """opt = torch.optim.SGD(net3.parameters(), lr=0.01)
len(opt.param_groups), opt.param_groups[0]["lr"]"""),
        ],
        """Exercise 1 checks input-output dimensions.

Exercise 2 shows that optimizers store parameter groups and settings.

A parameter group is a collection of parameters updated with the same optimizer settings.""",
        """Parameter groups become useful when different parts of a model need different learning rates.""",
        """- The first argument to `Linear` is input feature count.
- The second argument to `Linear` is output count.
- Optimizer settings live in parameter groups.
- Changing a learning rate changes update size, not the model formula.""",
    )

    write_nb("Chapter 3.5 - Concise Implementation of Linear Regression.ipynb", cells)


def build_36() -> None:
    cells = [
        title_cell(
            "Chapter 3.6 - Generalization",
            "Generalization is the ability of a trained model to perform well on new examples, not just the examples it saw during training.",
        ),
        imports_cell(),
    ]

    cells += section(
        "3.6.1 Training Error and Generalization Error",
        """Training error is error measured on the data used for learning.

Generalization error is expected error on new data from the same real problem. Because we cannot see all future data, we estimate it with validation or test data.""",
        """A model that memorizes training data may have low training error but poor real-world performance. Generalization is the reason evaluation data matters.""",
        [
            ("md", "Compute training and validation loss separately."),
            ("code", """train_pred = torch.tensor([1.0, 2.0, 3.0])
train_y = torch.tensor([1.0, 2.0, 4.0])
valid_pred = torch.tensor([1.0, 2.0])
valid_y = torch.tensor([2.0, 2.0])
train_loss = ((train_pred - train_y) ** 2).mean()
valid_loss = ((valid_pred - valid_y) ** 2).mean()
train_loss, valid_loss"""),
        ],
        """The training loss uses predictions and labels from training examples.

The validation loss uses examples not used for parameter updates.

Comparing these values helps diagnose whether the model is only fitting the training data.""",
        """ML systems usually split data into training and validation or test sets. The split gives a less biased estimate of future performance.""",
        """- Low training error alone is not enough.
- Validation data should not be used for gradient updates.
- Test data should be touched as little as possible.
- Generalization error is estimated, not known exactly.""",
    )

    cells += section(
        "3.6.2 Underfitting or Overfitting?",
        """Underfitting means the model is too simple or poorly trained to fit even the training data.

Overfitting means the model fits training data too specifically and performs worse on new data.""",
        """These diagnoses guide what to change. Underfitting and overfitting require different fixes.""",
        [
            ("md", "Classify simple error patterns."),
            ("code", """def diagnose(train_loss, valid_loss):
    if train_loss > 1.0 and valid_loss > 1.0:
        return "likely underfitting"
    if valid_loss > train_loss * 3:
        return "likely overfitting"
    return "roughly balanced"

diagnose(0.1, 0.8), diagnose(3.0, 3.5)"""),
        ],
        """The function is only a toy rule, not a universal test.

High training and validation loss suggests the model cannot fit the pattern well.

Very low training loss with much higher validation loss suggests overfitting.

A balanced gap suggests neither obvious problem from these numbers alone.""",
        """Training dashboards often show training and validation curves. The shape of these curves helps decide whether to train longer, simplify the model, add regularization, or get more data.""",
        """- Overfitting is about the gap between training and new-data performance.
- Underfitting can come from too little training, not only too simple a model.
- A toy threshold is not a scientific diagnosis.
- Noisy validation estimates can mislead when validation data is small.""",
    )

    cells += section(
        "3.6.3 Model Selection",
        """Model selection means choosing between candidate models or settings.

A hyperparameter is a setting chosen before or outside training, such as learning rate, model size, or regularization strength.""",
        """Different choices can produce different generalization behavior. We need a principled way to compare them without using test data repeatedly.""",
        [
            ("md", "Choose the candidate with the lowest validation loss."),
            ("code", """candidates = [
    {"degree": 1, "valid_loss": 0.8},
    {"degree": 2, "valid_loss": 0.3},
    {"degree": 5, "valid_loss": 0.6},
]
best = min(candidates, key=lambda item: item["valid_loss"])
best"""),
        ],
        """Each dictionary stores one candidate setting and its validation loss.

`min(..., key=...)` chooses the item with the smallest selected value.

The lambda function tells Python to compare candidates by `valid_loss`.

The chosen model is selected by validation performance, not training performance.""",
        """Model selection is common in experiments. You try several learning rates, architectures, or regularization settings and choose using validation data.""",
        """- A hyperparameter is not learned by ordinary gradient descent in the same run.
- Validation data guides selection; test data estimates final performance.
- Reusing validation data many times can overfit the validation set.
- Lower validation loss is useful only if the validation set matches the real target problem.""",
    )

    cells += section(
        "3.6.4 Summary",
        """Generalization is the real goal of supervised learning.

Training error tells us how well the model fits known examples. Validation or test error estimates performance on unseen examples.""",
        """A model is useful only if it handles future examples well enough for the task.""",
        [
            ("md", "A compact checklist for generalization debugging."),
            ("code", """checks = [
    "compare train and validation loss",
    "watch for large gaps",
    "choose hyperparameters on validation data",
    "reserve test data for final estimate",
]
checks"""),
        ],
        """The checklist follows the basic evaluation discipline.

First compare train and validation behavior.

Then diagnose underfitting or overfitting.

Then choose settings carefully.

Finally estimate performance on held-out data.""",
        """This discipline becomes more important as models become more flexible and easier to overfit.""",
        """- Generalization cannot be proven by training loss.
- More complex models can help or hurt.
- Evaluation protocol is part of the ML system.
- A single split gives only one estimate.""",
    )

    cells += section(
        "3.6.5 Exercises",
        """These exercises practice reading train/validation behavior.""",
        """Correct diagnosis prevents random trial-and-error changes.""",
        [
            ("md", "Exercise 1: diagnose two loss patterns."),
            ("code", """case_a = diagnose(train_loss=0.05, valid_loss=0.7)
case_b = diagnose(train_loss=2.0, valid_loss=2.5)
case_a, case_b"""),
            ("md", "Exercise 2: select the best learning rate by validation loss."),
            ("code", """runs = [{"lr": 0.1, "valid_loss": 0.9}, {"lr": 0.01, "valid_loss": 0.4}]
best_run = min(runs, key=lambda run: run["valid_loss"])
best_run"""),
        ],
        """Exercise 1 checks underfitting and overfitting language.

Exercise 2 checks validation-based model selection.

The important habit is separating training performance from selection and final evaluation.""",
        """This prepares for weight decay, where regularization changes the tradeoff between fitting and generalizing.""",
        """- Validation loss is an estimate, not an absolute truth.
- The best validation run may not be best on future data.
- Do not choose based only on training loss.
- Record selection criteria before comparing many runs.""",
    )

    write_nb("Chapter 3.6 - Generalization.ipynb", cells)


def build_37() -> None:
    cells = [
        title_cell(
            "Chapter 3.7 - Weight Decay",
            "Weight decay is a regularization method that discourages large weights. It helps control overfitting by adding a size penalty to the training objective.",
        ),
        imports_cell(),
    ]

    cells += section(
        "3.7.1 Norms and Weight Decay",
        """A norm measures the size of a vector. The L2 norm squares each value, sums the squares, and takes a square root.

Weight decay penalizes large weights by adding a term based on weight size to the loss.""",
        """Large weights can make a model overly sensitive to small input changes. Penalizing weight size encourages simpler functions.""",
        [
            ("md", "Manual L2 penalty for a small weight vector."),
            ("code", """w = [3.0, 4.0]
l2_squared = sum(value * value for value in w)
penalty = 0.5 * l2_squared
penalty"""),
            ("md", "PyTorch L2 penalty."),
            ("code", """w = torch.tensor([3.0, 4.0])
penalty = 0.5 * torch.sum(w ** 2)
penalty"""),
        ],
        """The manual code squares each weight and sums the results.

The factor `0.5` is a convenience because it simplifies derivatives.

In PyTorch, `w ** 2` squares each weight elementwise.

`torch.sum(...)` reduces all squared values into one scalar penalty.""",
        """Weight decay modifies the training objective from `data loss` to `data loss plus weight-size penalty`. The model must fit data while keeping weights controlled.""",
        """- Weight decay usually penalizes weights, not the bias.
- The penalty is added during training, not after training.
- A larger penalty strength forces smaller weights more strongly.
- Smaller weights do not automatically mean better predictions.""",
    )

    cells += section(
        "3.7.2 High-Dimensional Linear Regression",
        """High-dimensional data has many features. If the number of features is large compared with the number of examples, overfitting becomes easier.

The model has many ways to fit training noise when it has many weights.""",
        """Weight decay is especially useful when the model has enough flexibility to memorize the training data.""",
        [
            ("md", "Create more features than examples."),
            ("code", """torch.manual_seed(0)
num_train = 5
num_features = 20
X = torch.randn(num_train, num_features)
true_w = torch.zeros(num_features)
true_w[:2] = torch.tensor([2.0, -3.4])
y = X @ true_w + 0.01 * torch.randn(num_train)
X.shape, y.shape"""),
        ],
        """`num_features = 20` means each example has 20 input values.

`num_train = 5` means there are only 5 training examples.

Only the first two true weights are nonzero.

The model does not know this sparse structure unless we guide it with assumptions or regularization.""",
        """Real high-dimensional problems include text features, genomics, recommendation systems, and wide tabular data. Regularization helps reduce reliance on accidental patterns.""",
        """- Many features increase the chance of fitting noise.
- More dimensions are not automatically bad, but they require enough data or regularization.
- True useful features may be sparse.
- Synthetic high-dimensional data is useful for testing overfitting behavior.""",
    )

    cells += section(
        "3.7.3 Implementation from Scratch",
        """From scratch, weight decay is implemented by adding an L2 penalty to the loss before calling `backward()`.

The penalty strength is usually written as lambda. In code, use a name like `wd` because `lambda` is a Python keyword.""",
        """Adding the penalty manually shows exactly how regularization changes the objective and therefore the gradients.""",
        [
            ("md", "Define the penalty and add it to a data loss."),
            ("code", """def l2_penalty(w):
    return 0.5 * torch.sum(w ** 2)

w = torch.randn(20, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
y_hat = X @ w + b
data_loss = ((y_hat - y) ** 2).mean()
loss = data_loss + 0.1 * l2_penalty(w)
loss"""),
            ("md", "Compute gradients from the regularized objective."),
            ("code", """loss.backward()
w.grad.shape, b.grad.shape"""),
        ],
        """`l2_penalty(w)` computes the size penalty for the weight vector.

`data_loss` measures prediction error.

`0.1 * l2_penalty(w)` scales the regularization strength.

The final `loss` combines prediction error and weight penalty.

Calling `backward()` computes gradients for the combined objective.""",
        """This is the transparent version of weight decay. Later optimizer options can apply the same idea more concisely.""",
        """- Penalize `w`, not usually `b`.
- The regularization coefficient controls penalty strength.
- The penalty changes gradients, not just reported loss.
- Clear gradients before another backward pass in a real training loop.""",
    )

    cells += section(
        "3.7.4 Concise Implementation",
        """PyTorch optimizers can apply weight decay directly through an optimizer setting.

This keeps the loss expression focused on prediction error while the optimizer handles the weight penalty.""",
        """The concise option reduces boilerplate and matches common PyTorch practice.""",
        [
            ("md", "Create an optimizer with weight decay."),
            ("code", """net = torch.nn.Linear(20, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.03, weight_decay=0.1)
loss_fn = torch.nn.MSELoss()
y_col = y.reshape(-1, 1)
optimizer.zero_grad()
loss = loss_fn(net(X), y_col)
loss.backward()
optimizer.step()
loss"""),
        ],
        """`torch.nn.Linear(20, 1)` creates a linear model for 20 features.

`weight_decay=0.1` tells the optimizer to include weight decay in parameter updates.

The explicit `loss` is only prediction loss here.

`optimizer.step()` applies the update using gradients and the optimizer's weight-decay rule.""",
        """Most concise PyTorch implementations pass `weight_decay` to the optimizer. Some advanced optimizers handle weight decay slightly differently, so optimizer documentation matters.""",
        """- Built-in weight decay is an optimizer setting, not a separate visible loss term.
- Check whether the optimizer applies weight decay to all parameters or selected groups.
- Bias parameters are often excluded in larger systems.
- Concise code still follows zero-grad, loss, backward, step.""",
    )

    cells += section(
        "3.7.5 Summary",
        """Weight decay is regularization that discourages large weights.

It can be written manually as an L2 penalty or requested through optimizer settings.""",
        """The goal is better generalization, especially when the model can overfit training noise.""",
        [
            ("md", "Compare the two implementation styles."),
            ("code", """styles = {
    "scratch": "add wd * l2_penalty(w) to loss",
    "concise": "pass weight_decay to optimizer",
}
styles"""),
        ],
        """The scratch version makes the penalty visible in the loss.

The concise version moves the penalty into the optimizer.

Both are attempts to control model complexity by discouraging large weights.""",
        """Weight decay is one of the first regularization techniques used in deep learning systems. Later chapters add more regularization tools.""",
        """- Regularization improves expected behavior; it does not guarantee better validation loss every time.
- Too much weight decay can cause underfitting.
- Weight decay strength is a hyperparameter.
- Always compare training and validation performance.""",
    )

    cells += section(
        "3.7.6 Exercises",
        """These exercises practice adding and interpreting weight penalties.""",
        """The goal is to connect the regularization coefficient, weight size, and objective value.""",
        [
            ("md", "Exercise 1: compute L2 penalty for a different weight vector."),
            ("code", """w_test = torch.tensor([1.0, -2.0, 2.0])
l2_penalty(w_test)"""),
            ("md", "Exercise 2: compare regularized losses with two penalty strengths."),
            ("code", """data_loss = torch.tensor(1.5)
penalty = l2_penalty(w_test)
loss_small = data_loss + 0.01 * penalty
loss_large = data_loss + 1.0 * penalty
loss_small, loss_large"""),
        ],
        """Exercise 1 checks the L2 penalty formula.

Exercise 2 checks that larger regularization strength increases the objective more for the same weights.

The model would therefore feel more pressure to reduce weight size.""",
        """This prepares you to tune weight decay using validation performance rather than guessing.""",
        """- A larger penalty coefficient does not always mean better generalization.
- The same coefficient can behave differently with different loss scales.
- Regularization changes training dynamics.
- Tune weight decay on validation data, not test data.""",
    )

    write_nb("Chapter 3.7 - Weight Decay.ipynb", cells)


def main() -> None:
    build_31()
    build_32()
    build_33()
    build_34()
    build_35()
    build_36()
    build_37()


if __name__ == "__main__":
    main()
