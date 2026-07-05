"""Generate Chapter 5 notebooks for the D2L rewrite project."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Chapter 5 - Multilayer Perceptrons"
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



def build_51() -> None:
    cells = [
        title_cell(
            "Chapter 5.1 - Multilayer Perceptrons",
            "A multilayer perceptron, or MLP, is a neural network that stacks linear layers with nonlinear activation functions. It is the first model here that can represent nonlinear patterns.",
        ),
        imports_cell(),
    ]

    cells += section(
        "5.1.1 Hidden Layers",
        """A hidden layer is an intermediate layer between the input and output. It is called hidden because its values are not labels and are not directly observed in the dataset.

A linear layer alone can only build linear decision boundaries. A hidden layer followed by a nonlinear activation lets the model build more flexible functions.""",
        """Many real patterns are not linear. Hidden layers let a model learn intermediate features before making a final prediction.""",
        [
            ("md", "Manual Python: one hidden unit from two inputs."),
            ("code", """x = [1.0, 2.0]
w = [0.5, -1.0]
b = 0.1
hidden_raw = x[0] * w[0] + x[1] * w[1] + b
hidden = max(0.0, hidden_raw)
hidden"""),
            ("md", "PyTorch: a tiny hidden layer maps two features to three hidden values."),
            ("code", """X = torch.tensor([[1.0, 2.0]])
hidden_layer = torch.nn.Linear(2, 3)
raw_hidden = hidden_layer(X)
hidden = torch.relu(raw_hidden)
hidden.shape"""),
        ],
        """The manual code first computes a weighted sum.

`max(0.0, hidden_raw)` is a ReLU activation. It keeps positive values and replaces negative values with zero.

The PyTorch layer computes three hidden weighted sums at once.

`torch.relu` applies the same nonlinear rule elementwise.""",
        """MLPs stack hidden representations. Later layers use these hidden values instead of the original raw inputs.""",
        """- Hidden does not mean mysterious; it means intermediate.
- A hidden layer without a nonlinear activation still collapses into one linear transformation.
- Hidden width is the number of hidden units.
- More hidden units increase flexibility and computational cost.""",
    )

    cells += section(
        "5.1.2 Activation Functions",
        """An activation function is a nonlinear function applied to layer outputs.

Common activation functions include ReLU, sigmoid, and tanh. Nonlinear means the function is not just a weighted sum or constant scaling.""",
        """Without activation functions, stacking linear layers is still equivalent to one linear layer. Activations are what let deep networks represent nonlinear patterns.""",
        [
            ("md", "Compare three activation functions on tiny inputs."),
            ("code", """x = torch.tensor([-2.0, 0.0, 2.0])
relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)
relu, sigmoid, tanh"""),
            ("md", "A small MLP forward pass."),
            ("code", """X = torch.tensor([[1.0, -1.0]])
W1 = torch.randn(2, 3)
b1 = torch.zeros(3)
H = torch.relu(X @ W1 + b1)
H"""),
        ],
        """ReLU outputs zero for negative inputs and keeps positive inputs.

Sigmoid maps inputs into the range from 0 to 1.

Tanh maps inputs into the range from -1 to 1.

The MLP example computes hidden pre-activations with `X @ W1 + b1`, then applies ReLU.""",
        """Modern MLPs commonly use ReLU-like activations because they are simple and train well in many settings.""",
        """- Activation functions are applied elementwise.
- Sigmoid and tanh can saturate when inputs are very large or very small.
- ReLU can output exact zeros.
- Nonlinearity is essential for stacked layers to add expressive power.""",
    )

    cells += section(
        "5.1.3 Summary and Discussion",
        """An MLP combines linear layers and activation functions.

The simplest MLP has an input layer, one hidden layer, an activation, and an output layer.""",
        """MLPs are a general-purpose starting point for learning nonlinear relationships in vector data.""",
        [
            ("md", "A compact two-layer MLP."),
            ("code", """mlp = torch.nn.Sequential(
    torch.nn.Linear(2, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1),
)
mlp(torch.tensor([[1.0, 2.0]])).shape"""),
        ],
        """`Sequential` runs modules in order.

The first linear layer maps 2 inputs to 3 hidden values.

`ReLU` applies nonlinearity.

The final linear layer maps 3 hidden values to 1 output.""",
        """The MLP pattern appears throughout deep learning. Later architectures change the layer type, but the idea of learned representations remains.""",
        """- Layer order matters.
- Hidden size controls representation width.
- Output size should match the task.
- The model is untrained until parameters are fit with data.""",
    )

    cells += section(
        "5.1.4 Exercises",
        """These exercises check whether you can reason about MLP shapes and activations.""",
        """Shape tracking is the first defense against confusing MLP code.""",
        [
            ("md", "Exercise 1: create an MLP that maps four inputs to two outputs."),
            ("code", """net = torch.nn.Sequential(
    torch.nn.Linear(4, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2),
)
net(torch.randn(3, 4)).shape"""),
            ("md", "Exercise 2: inspect ReLU on negative and positive values."),
            ("code", """x = torch.tensor([-1.0, 0.0, 1.0])
torch.relu(x)"""),
        ],
        """Exercise 1 checks batch size, input size, hidden size, and output size.

Exercise 2 checks the ReLU rule directly.

The batch has 3 examples, so the output has 3 rows.""",
        """These checks prepare for implementing MLP training loops.""",
        """- The final layer output count is task-dependent.
- ReLU does not change tensor shape.
- Batch size is independent of layer width.
- Hidden layers store learned intermediate representations.""",
    )

    write_nb("Chapter 5.1 - Multilayer Perceptrons.ipynb", cells)


def build_52() -> None:
    cells = [
        title_cell(
            "Chapter 5.2 - Implementation of Multilayer Perceptrons",
            "This notebook implements an MLP from scratch and then with concise PyTorch modules, keeping the mapping between the two visible.",
        ),
        imports_cell(),
    ]

    cells += section(
        "5.2.1 Implementation from Scratch",
        """From scratch, an MLP stores weight matrices and bias vectors for each layer and defines the forward computation manually.

For classification, the final layer outputs one logit per class.""",
        """Manual implementation exposes what framework layers store and compute internally.""",
        [
            ("md", "Define parameters and a forward function for a tiny MLP."),
            ("code", """W1 = torch.randn(4, 5) * 0.01
b1 = torch.zeros(5)
W2 = torch.randn(5, 3) * 0.01
b2 = torch.zeros(3)
def mlp(X):
    X = X.reshape(X.shape[0], -1)
    H = torch.relu(X @ W1 + b1)
    return H @ W2 + b2"""),
            ("md", "Run the from-scratch MLP on a tiny batch."),
            ("code", """X = torch.randn(2, 1, 2, 2)
logits = mlp(X)
logits.shape"""),
        ],
        """`W1` and `b1` belong to the hidden layer.

`W2` and `b2` belong to the output layer.

The input is flattened into feature vectors.

`torch.relu` creates hidden activations.

The final line returns class logits.""",
        """This is the same structure as a two-layer neural classifier, just without PyTorch module objects.""",
        """- Hidden parameters and output parameters have different shapes.
- The final output is logits, not probabilities.
- ReLU is applied after the hidden linear transformation.
- The model has no learning until gradients update parameters.""",
    )

    cells += section(
        "5.2.2 Concise Implementation",
        """The concise implementation uses `torch.nn.Sequential`, `Flatten`, `Linear`, and `ReLU` modules.

A module is a PyTorch object that stores parameters or computation behavior.""",
        """Concise modules reduce boilerplate and automatically register parameters for optimizers.""",
        [
            ("md", "Define the same MLP with PyTorch modules."),
            ("code", """net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(4, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 3),
)
net(torch.randn(2, 1, 2, 2)).shape"""),
            ("md", "Inspect parameter shapes."),
            ("code", """[p.shape for p in net.parameters()]"""),
        ],
        """`Flatten` converts each image into a vector.

The first `Linear` module is the hidden layer.

`ReLU` applies the activation.

The second `Linear` module returns logits.

`net.parameters()` exposes trainable tensors.""",
        """This is the standard way to define small MLPs in PyTorch.""",
        """- `Sequential` hides no training logic; it only defines forward order.
- `ReLU` has no trainable parameters.
- `Linear` has weights and bias.
- Parameter shapes can be inspected before training.""",
    )

    cells += section(
        "5.2.3 Summary",
        """From-scratch and concise MLPs compute the same kind of function.

The concise version packages parameters and operations into modules.""",
        """Mapping manual tensors to PyTorch modules keeps framework code understandable.""",
        [
            ("md", "Map manual pieces to concise modules."),
            ("code", """mapping = {
    "W1, b1": "first Linear layer",
    "ReLU call": "ReLU module",
    "W2, b2": "second Linear layer",
}
mapping"""),
        ],
        """The dictionary names equivalent concepts.

Manual parameters become `Linear` module parameters.

Manual activation calls become activation modules.""",
        """Later architectures use the same mapping pattern with different layer types.""",
        """- Concise code should be traceable to manual computation.
- Modules can store parameters or just computation.
- Output logits still need a task-appropriate loss.
- Hidden size is a hyperparameter.""",
    )

    cells += section(
        "5.2.4 Exercises",
        """These exercises practice building and inspecting MLPs.""",
        """Small architecture changes should be easy to reason about before training.""",
        [
            ("md", "Exercise 1: create a deeper MLP with two hidden layers."),
            ("code", """deep = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2),
)
deep(torch.randn(5, 3)).shape"""),
            ("md", "Exercise 2: count trainable tensors."),
            ("code", """len(list(deep.parameters()))"""),
        ],
        """Exercise 1 checks layer chaining.

Exercise 2 counts weight and bias tensors.

Three linear layers create six parameter tensors: three weights and three biases.""",
        """Parameter inspection becomes important for debugging larger networks.""",
        """- Every Linear layer usually has two parameter tensors.
- ReLU layers do not add parameters.
- Adjacent layer dimensions must match.
- More layers add expressiveness and training difficulty.""",
    )

    write_nb("Chapter 5.2 - Implementation of Multilayer Perceptrons.ipynb", cells)


def build_53() -> None:
    cells = [
        title_cell(
            "Chapter 5.3 - Forward Propagation, Backward Propagation, and Computational Graphs",
            "This notebook explains how values flow forward through a network and how gradients flow backward through the recorded computation graph.",
        ),
        imports_cell(),
    ]

    cells += section(
        "5.3.1 Forward Propagation",
        """Forward propagation is the process of computing outputs from inputs by running the model operations in order.

Intermediate values are values created between input and output, such as hidden activations.""",
        """The loss cannot be computed until predictions exist, and predictions come from the forward pass.""",
        [
            ("md", "A tiny forward pass through one hidden layer."),
            ("code", """X = torch.tensor([[1.0, 2.0]])
W1 = torch.randn(2, 3)
b1 = torch.zeros(3)
W2 = torch.randn(3, 1)
b2 = torch.zeros(1)
H = torch.relu(X @ W1 + b1)
y_hat = H @ W2 + b2
y_hat"""),
        ],
        """The input `X` is used first.

`X @ W1 + b1` computes hidden pre-activations.

`torch.relu` creates hidden activations.

`H @ W2 + b2` computes the final output.

The forward pass stores values that backpropagation may need later.""",
        """Training loops begin each step with forward propagation, then use the result to compute loss.""",
        """- Forward propagation computes predictions only.
- It does not update parameters by itself.
- Intermediate values matter for gradient computation.
- Layer order determines computation order.""",
    )

    cells += section(
        "5.3.2 Computational Graph of Forward Propagation",
        """A computational graph records how values are produced from other values.

Nodes are values or operations. Edges show dependency: which value was used to make another value.""",
        """The graph lets automatic differentiation apply the chain rule systematically.""",
        [
            ("md", "A small graph-like dependency list."),
            ("code", """graph = [
    "X -> hidden linear",
    "hidden linear -> ReLU",
    "ReLU -> output linear",
    "output linear -> loss",
]
graph"""),
            ("md", "PyTorch records tensor operations when gradients are required."),
            ("code", """x = torch.tensor(2.0, requires_grad=True)
y = x * x
z = y + 3
z.grad_fn is not None"""),
        ],
        """The dependency list is a human-readable graph.

In PyTorch, `requires_grad=True` asks autograd to track operations involving `x`.

`y` and `z` remember how they were computed.

`grad_fn` is PyTorch's stored backward-function reference for non-leaf tensors.""",
        """Deep learning frameworks build these graphs dynamically during ordinary tensor computation.""",
        """- A graph is about dependencies, not visual appearance.
- Leaf tensors are created directly by the user.
- Non-leaf tensors are produced by operations.
- The graph is used for gradients after the loss is computed.""",
    )

    cells += section(
        "5.3.3 Backpropagation",
        """Backpropagation computes gradients by moving backward from the loss through the computational graph.

It applies the chain rule: multiply local sensitivities along dependency paths.""",
        """Parameters can be improved only if we know how changing them would change the loss.""",
        [
            ("md", "PyTorch backpropagation through a small computation."),
            ("code", """x = torch.tensor(2.0, requires_grad=True)
y = x * x
loss = y + 1
loss.backward()
x.grad"""),
            ("md", "Manual chain rule for the same computation."),
            ("code", """x_value = 2.0
dloss_dy = 1.0
dy_dx = 2 * x_value
dloss_dx = dloss_dy * dy_dx
dloss_dx"""),
        ],
        """`loss.backward()` starts from the scalar loss.

PyTorch traces back through `loss = y + 1` and `y = x * x`.

`x.grad` stores the derivative of loss with respect to `x`.

The manual calculation multiplies the local derivative from each step.""",
        """Backpropagation is the core algorithm that makes neural network training practical.""",
        """- Backpropagation computes gradients, not parameter updates.
- The optimizer uses gradients after backpropagation.
- Gradients accumulate unless cleared.
- The chain rule is the mathematical basis.""",
    )

    cells += section(
        "5.3.4 Training Neural Networks",
        """Training a neural network repeats forward propagation, loss computation, backpropagation, and parameter updates.

A parameter update changes weights and biases using an optimizer.""",
        """The full loop is where all pieces connect into learning.""",
        [
            ("md", "One tiny neural-network training step."),
            ("code", """net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
X = torch.randn(4, 2)
y = torch.randn(4, 1)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(net.parameters(), lr=0.1)
opt.zero_grad()
loss = loss_fn(net(X), y)
loss.backward()
opt.step()
loss"""),
        ],
        """The model computes predictions with `net(X)`.

The loss compares predictions with labels.

`zero_grad()` clears old gradients.

`backward()` computes new gradients.

`step()` updates parameters.""",
        """This control flow is shared by MLPs, CNNs, RNNs, and transformers, even though their layers differ.""",
        """- Training is a repeated process, not one forward pass.
- Clearing gradients is necessary before the next backward pass.
- Loss must connect to parameters through the graph.
- Optimizer steps should happen after gradients exist.""",
    )

    cells += section(
        "5.3.5 Summary",
        """Forward propagation computes predictions. Backpropagation computes gradients. The optimizer updates parameters.

The computational graph connects these steps.""",
        """Understanding this flow removes much of the mystery from deep learning training loops.""",
        [
            ("md", "The repeated training sequence."),
            ("code", """sequence = [
    "forward",
    "loss",
    "zero gradients",
    "backward",
    "optimizer step",
]
sequence"""),
        ],
        """The sequence names what happens in one training iteration.

In many PyTorch loops, `zero_grad` appears before forward or before backward.

The important rule is that old gradients should not pollute the new update.""",
        """Later training frameworks package this sequence, but the same execution order remains underneath.""",
        """- The graph records how tensors were computed.
- Backward uses the graph to compute gradients.
- The optimizer changes parameters.
- Debug training by checking each step separately.""",
    )

    cells += section(
        "5.3.6 Exercises",
        """These exercises practice tracing values and gradients.""",
        """Small graphs make it easier to understand large neural networks.""",
        [
            ("md", "Exercise 1: compute a gradient through two operations."),
            ("code", """x = torch.tensor(3.0, requires_grad=True)
y = 2 * x
loss = y * y
loss.backward()
x.grad"""),
            ("md", "Exercise 2: list the training loop steps from memory."),
            ("code", """steps = ["forward", "loss", "backward", "update"]
steps"""),
        ],
        """Exercise 1 checks chain-rule tracking.

Exercise 2 checks control-flow memory.

Both are core skills for reading training code.""",
        """These same ideas appear in all later deep learning chapters.""",
        """- Gradients are local to the current graph and values.
- A scalar loss makes `.backward()` easiest.
- Parameter updates are separate from gradient computation.
- Always know what is stored in memory during training.""",
    )

    write_nb("Chapter 5.3 - Forward Propagation, Backward Propagation, and Computational Graphs.ipynb", cells)


def build_54() -> None:
    cells = [
        title_cell(
            "Chapter 5.4 - Numerical Stability and Initialization",
            "Deep networks can fail because numbers become too small, too large, or poorly scaled. Initialization chooses starting parameter values to make training more stable.",
        ),
        imports_cell(),
    ]

    cells += section(
        "5.4.1 Vanishing and Exploding Gradients",
        """A vanishing gradient becomes extremely small as it moves backward through layers. An exploding gradient becomes extremely large.

Both problems make learning difficult because parameter updates become too tiny or too unstable.""",
        """Deep networks multiply many local derivatives. Repeated multiplication can shrink or grow gradients rapidly.""",
        [
            ("md", "Repeated multiplication can shrink values."),
            ("code", """value = torch.tensor(1.0)
for _ in range(5):
    value = value * 0.2
value"""),
            ("md", "Repeated multiplication can grow values."),
            ("code", """value = torch.tensor(1.0)
for _ in range(5):
    value = value * 3.0
value"""),
        ],
        """The first loop repeatedly multiplies by a number smaller than 1, so the value shrinks.

The second loop repeatedly multiplies by a number larger than 1, so the value grows.

Backpropagation can face similar repeated multiplication across layers.""",
        """Stability problems motivate careful activation choices, initialization schemes, normalization, and gradient clipping.""",
        """- Vanishing gradients slow or stop early-layer learning.
- Exploding gradients can produce unstable updates.
- Depth increases the chance of repeated-scaling problems.
- Stable training is an engineering and mathematical concern.""",
    )

    cells += section(
        "5.4.2 Parameter Initialization",
        """Parameter initialization means choosing starting values for weights and biases before training.

Random initialization breaks symmetry. Symmetry means two units start and behave identically, learning the same thing.""",
        """Bad initialization can make activations or gradients too large or too small before training even has a chance.""",
        [
            ("md", "Compare zero and random weight initialization."),
            ("code", """zeros = torch.zeros(2, 3)
random_weights = torch.randn(2, 3) * 0.01
zeros, random_weights"""),
            ("md", "Use Xavier initialization on a linear layer."),
            ("code", """layer = torch.nn.Linear(4, 3)
torch.nn.init.xavier_uniform_(layer.weight)
torch.nn.init.zeros_(layer.bias)
layer.weight.shape, layer.bias.shape"""),
        ],
        """Zero weights can make hidden units identical.

Small random weights break that identical behavior.

Xavier initialization chooses a scale based on input and output sizes.

The underscore in `xavier_uniform_` means the function modifies the tensor in place.""",
        """PyTorch layers initialize parameters automatically, but explicit initialization is common in research and debugging.""",
        """- Initialization happens before training.
- Random does not mean arbitrary; scale matters.
- Biases are often initialized to zero.
- In-place initialization functions commonly end with `_` in PyTorch.""",
    )

    cells += section(
        "5.4.3 Summary",
        """Numerical stability is about keeping values and gradients in useful ranges.

Initialization is one early decision that affects stability.""",
        """A network that starts in a bad numerical regime may train slowly or fail entirely.""",
        [
            ("md", "A simple stability checklist."),
            ("code", """checks = [
    "activation scale",
    "gradient scale",
    "weight initialization",
    "learning rate",
]
checks"""),
        ],
        """The checklist names common sources of instability.

Activations are forward-pass values.

Gradients are backward-pass values.

Both interact with initialization and learning rate.""",
        """Deep learning practice often starts by checking whether loss is finite and gradients are reasonable.""",
        """- Stable numbers are required for useful learning.
- Initialization is not a replacement for good data or architecture.
- Larger networks need more careful stability checks.
- Watch for `nan` or `inf` values during debugging.""",
    )

    cells += section(
        "5.4.4 Exercises",
        """These exercises practice recognizing scale problems and applying initialization.""",
        """Small experiments reveal why initialization matters before large models obscure the issue.""",
        [
            ("md", "Exercise 1: observe repeated shrinking."),
            ("code", """x = torch.tensor(1.0)
for _ in range(4):
    x = x * 0.5
x"""),
            ("md", "Exercise 2: initialize a layer bias to zero."),
            ("code", """layer = torch.nn.Linear(3, 2)
torch.nn.init.zeros_(layer.bias)
layer.bias"""),
        ],
        """Exercise 1 shows repeated scaling below 1.

Exercise 2 uses an in-place initializer.

Both are tiny versions of stability-related operations.""",
        """These ideas become more important as networks deepen.""",
        """- Repeated multiplication can change scale quickly.
- Initialization functions can modify parameters directly.
- Bias and weight initialization can use different rules.
- Stability should be checked early in training.""",
    )

    write_nb("Chapter 5.4 - Numerical Stability and Initialization.ipynb", cells)


def build_55() -> None:
    cells = [
        title_cell(
            "Chapter 5.5 - Generalization in Deep Learning",
            "Generalization in deep learning asks why large flexible networks can sometimes perform well on new data, and how we reduce overfitting risk in practice.",
        ),
        imports_cell(),
    ]

    cells += section(
        "5.5.1 Revisiting Overfitting and Regularization",
        """Overfitting means a model fits training data too specifically and performs worse on new data.

Regularization is any method that tries to improve generalization by constraining or guiding the learning process.""",
        """Deep networks can have many parameters, so training loss alone is not enough evidence of useful learning.""",
        [
            ("md", "Compare train and validation losses."),
            ("code", """train_loss = torch.tensor([1.0, 0.5, 0.2])
valid_loss = torch.tensor([1.1, 0.7, 0.9])
gap = valid_loss - train_loss
gap"""),
        ],
        """Training loss decreases across epochs.

Validation loss first decreases and then increases.

The widening gap can indicate overfitting.

Regularization methods try to reduce this problem.""",
        """Common regularization methods include weight decay, dropout, data augmentation, early stopping, and model-size control.""",
        """- Low training loss is not the final goal.
- Regularization can increase training loss while improving validation performance.
- Overfitting is diagnosed with held-out data.
- No regularization method is universally best.""",
    )

    cells += section(
        "5.5.2 Inspiration from Nonparametrics",
        """Nonparametric methods can grow in complexity with the amount of data.

The term does not mean no parameters at all. It means the model capacity is not fixed in the same simple way as a small linear model.""",
        """This perspective helps explain why flexibility is not automatically bad. Flexible models need enough data, suitable structure, and careful evaluation.""",
        [
            ("md", "A tiny nearest-neighbor style prediction."),
            ("code", """train_x = torch.tensor([0.0, 1.0, 3.0])
train_y = torch.tensor([0.0, 2.0, 6.0])
query = torch.tensor(1.2)
dist = torch.abs(train_x - query)
pred = train_y[torch.argmin(dist)]
pred"""),
        ],
        """The query is compared with stored training inputs.

`torch.abs(train_x - query)` computes distances.

`argmin` finds the nearest stored example.

The prediction reuses that nearest example's label.""",
        """Large neural networks can also use data-rich flexibility, though their mechanism differs from nearest neighbors.""",
        """- Nonparametric does not mean simple.
- Memorization and useful flexibility can look similar without proper evaluation.
- More data can support more flexible models.
- This section is conceptual motivation, not a training recipe.""",
    )

    cells += section(
        "5.5.3 Early Stopping",
        """Early stopping means stopping training when validation performance stops improving.

It treats the number of training epochs as a regularization choice.""",
        """A model may begin to overfit if training continues after validation loss starts worsening.""",
        [
            ("md", "Select the epoch with the lowest validation loss."),
            ("code", """valid_loss = torch.tensor([0.9, 0.6, 0.5, 0.7])
best_epoch = int(torch.argmin(valid_loss))
best_loss = valid_loss[best_epoch]
best_epoch, best_loss"""),
        ],
        """The validation-loss tensor stores one value per epoch.

`argmin` finds the position of the smallest validation loss.

That position is the chosen stopping point in this toy example.""",
        """Training systems often save the best validation checkpoint instead of simply keeping the final epoch.""",
        """- Early stopping uses validation data, not test data.
- The best epoch is a hyperparameter chosen by validation performance.
- Noisy validation curves can make stopping decisions unstable.
- Checkpointing is needed if the best epoch is not the final epoch.""",
    )

    cells += section(
        "5.5.4 Classical Regularization Methods for Deep Networks",
        """Classical regularization methods include weight decay, dropout, data augmentation, and model-size control.

Data augmentation means creating modified training examples that preserve the label, such as cropping or flipping images.""",
        """Each method limits overfitting in a different way: by penalizing parameters, adding noise, increasing data variety, or reducing model capacity.""",
        [
            ("md", "A small regularization-method map."),
            ("code", """methods = {
    "weight_decay": "penalize large weights",
    "dropout": "randomly hide activations",
    "augmentation": "add label-preserving variants",
    "smaller_model": "reduce capacity",
}
methods"""),
        ],
        """The dictionary maps method names to plain-English effects.

Weight decay changes the objective.

Dropout changes training-time activations.

Augmentation changes the training data.

Smaller models reduce the function class.""",
        """Real systems often combine several regularization methods and choose strengths using validation data.""",
        """- Regularization is not a guarantee.
- Some methods are task-specific.
- Data augmentation must preserve labels.
- Validation performance should guide regularization strength.""",
    )

    cells += section(
        "5.5.5 Summary",
        """Deep-learning generalization is managed through evaluation discipline and regularization methods.

The practical goal is not minimum training loss, but reliable validation and test performance.""",
        """Deep networks are flexible, so careful model selection and regularization are essential.""",
        [
            ("md", "A generalization checklist."),
            ("code", """checks = [
    "track validation loss",
    "compare train/validation gap",
    "use regularization",
    "keep test set final",
]
checks"""),
        ],
        """The checklist follows the evaluation discipline.

Training and validation curves diagnose behavior.

Regularization methods are chosen using validation data.

The test set remains final.""",
        """This mindset applies to every later deep model.""",
        """- Generalization is measured on unseen data.
- Regularization can trade training fit for validation performance.
- Flexible models require careful evaluation.
- Keep records of hyperparameter choices.""",
    )

    cells += section(
        "5.5.6 Exercises",
        """These exercises practice generalization diagnostics.""",
        """The point is to read training curves as evidence, not decoration.""",
        [
            ("md", "Exercise 1: find the best validation epoch."),
            ("code", """losses = torch.tensor([0.8, 0.55, 0.58, 0.7])
int(torch.argmin(losses))"""),
            ("md", "Exercise 2: identify a growing train-validation gap."),
            ("code", """train = torch.tensor([0.9, 0.4])
valid = torch.tensor([1.0, 0.9])
valid - train"""),
        ],
        """Exercise 1 checks early stopping logic.

Exercise 2 checks gap computation.

A growing gap can signal overfitting.""",
        """These are simple versions of real experiment tracking.""",
        """- Choose epochs using validation data.
- Do not tune repeatedly on the test set.
- A gap is evidence, not a full diagnosis.
- Curves should be interpreted with domain context.""",
    )

    write_nb("Chapter 5.5 - Generalization in Deep Learning.ipynb", cells)


def build_56() -> None:
    cells = [
        title_cell(
            "Chapter 5.6 - Dropout",
            "Dropout is a regularization method that randomly sets some hidden activations to zero during training. It makes the network less dependent on any single hidden unit.",
        ),
        imports_cell(),
    ]

    cells += section(
        "5.6.1 Dropout in Practice",
        """Dropout randomly removes activations during training. The dropout probability is the chance that an activation is set to zero.

During evaluation, dropout is turned off so predictions are deterministic.""",
        """Dropout discourages hidden units from co-adapting too strongly. Co-adapting means relying on each other in a brittle way.""",
        [
            ("md", "Apply PyTorch dropout to a small tensor."),
            ("code", """torch.manual_seed(0)
drop = torch.nn.Dropout(p=0.5)
drop.train()
X = torch.ones(6)
drop(X)"""),
            ("md", "Switch dropout to evaluation mode."),
            ("code", """drop.eval()
drop(X)"""),
        ],
        """`p=0.5` means each activation has a 50 percent chance of being dropped during training.

`drop.train()` enables training behavior.

`drop.eval()` enables evaluation behavior.

In evaluation mode, the shown dropout layer leaves values unchanged.""",
        """PyTorch modules switch dropout behavior when the model uses `.train()` or `.eval()` mode.""",
        """- Dropout behaves differently during training and evaluation.
- Dropout is usually applied to hidden activations, not labels.
- Higher dropout means stronger noise.
- Dropout is a regularization method, not an optimizer.""",
    )

    cells += section(
        "5.6.2 Implementation from Scratch",
        """From scratch, dropout creates a random mask of zeros and ones and multiplies activations by that mask.

A mask is a tensor used to keep or remove values.""",
        """Manual dropout reveals that the method is just randomized elementwise filtering with scaling.""",
        [
            ("md", "Implement dropout for a small tensor."),
            ("code", """def dropout_layer(X, p):
    keep_prob = 1 - p
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob

X = torch.ones(5)
dropout_layer(X, 0.4)"""),
        ],
        """`keep_prob` is the probability of keeping an activation.

`torch.rand(X.shape)` creates random values between 0 and 1.

The comparison creates a Boolean mask.

Dividing by `keep_prob` keeps the expected activation scale similar.""",
        """Framework dropout layers implement this idea and handle training/evaluation mode automatically.""",
        """- The mask is random during training.
- Scaling prevents average activation size from shrinking too much.
- `p=0` means no dropout.
- `p=1` would drop everything and is not useful here.""",
    )

    cells += section(
        "5.6.3 Concise Implementation",
        """Concise dropout uses `torch.nn.Dropout` as a module inside a network.

It is usually placed after activation functions in hidden layers.""",
        """The module version handles random masking and train/eval behavior consistently.""",
        [
            ("md", "Build a small MLP with dropout."),
            ("code", """net = torch.nn.Sequential(
    torch.nn.Linear(4, 5),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(5, 2),
)
net.train()
net(torch.randn(3, 4)).shape"""),
        ],
        """The first linear layer creates hidden values.

ReLU applies nonlinearity.

Dropout randomly masks hidden activations during training mode.

The final linear layer produces outputs.""",
        """Dropout is common in fully connected networks, though modern architectures may rely on other regularization methods too.""",
        """- Dropout modules should be inside the model if they are part of training behavior.
- `.train()` and `.eval()` matter.
- Dropout changes outputs randomly during training.
- Do not use dropout as a substitute for validation.""",
    )

    cells += section(
        "5.6.4 Summary",
        """Dropout randomly hides activations during training and is disabled during evaluation.

It regularizes by making the network less dependent on particular hidden units.""",
        """Dropout is one practical tool for reducing overfitting in flexible neural networks.""",
        [
            ("md", "A dropout behavior summary."),
            ("code", """behavior = {
    "train_mode": "randomly mask activations",
    "eval_mode": "use all activations",
}
behavior"""),
        ],
        """The dictionary contrasts training and evaluation behavior.

This mode difference is the most important operational detail.

Forgetting it can make evaluation noisy or training ineffective.""",
        """Model training scripts should call `.train()` for training and `.eval()` for validation or testing.""",
        """- Dropout is stochastic during training.
- Evaluation should be deterministic unless intentionally sampling.
- Dropout probability is a hyperparameter.
- Too much dropout can cause underfitting.""",
    )

    cells += section(
        "5.6.5 Exercises",
        """These exercises practice dropout masks and modes.""",
        """Understanding mode-dependent behavior prevents subtle evaluation bugs.""",
        [
            ("md", "Exercise 1: apply scratch dropout to a vector."),
            ("code", """torch.manual_seed(1)
X = torch.ones(4)
dropout_layer(X, 0.5)"""),
            ("md", "Exercise 2: compare train and eval mode shape."),
            ("code", """net.eval()
out = net(torch.randn(2, 4))
out.shape"""),
        ],
        """Exercise 1 checks random masking.

Exercise 2 checks that evaluation mode still preserves output shape.

The values may change by mode, but the shape should not.""",
        """These checks are useful when adding dropout to existing networks.""",
        """- Random masks differ unless the seed is fixed.
- Mode changes behavior, not tensor shape.
- Dropout is normally used only during training.
- Always set evaluation mode before validation metrics.""",
    )

    write_nb("Chapter 5.6 - Dropout.ipynb", cells)


def build_57() -> None:
    cells = [
        title_cell(
            "Chapter 5.7 - Predicting House Prices on Kaggle",
            "This notebook teaches the Kaggle-style workflow using tiny offline tabular data. It avoids downloads while preserving the real steps: data access, preprocessing, validation, model selection, and submission formatting.",
        ),
        imports_cell(),
    ]

    cells += section(
        "5.7.1 Downloading Data",
        """Kaggle is a platform for data science competitions. Competitions provide training data, test data, and a required submission format.

Downloading data means obtaining those files locally before training.""",
        """A reproducible project needs to know where data came from and how it was stored. This notebook uses toy offline data instead of network downloads.""",
        [
            ("md", "Represent downloaded train and test tables as Python rows."),
            ("code", """train_rows = [
    {"id": 1, "rooms": 2.0, "area": 80.0, "price": 200.0},
    {"id": 2, "rooms": 3.0, "area": 120.0, "price": 300.0},
]
test_rows = [{"id": 3, "rooms": 2.0, "area": 100.0}]
len(train_rows), len(test_rows)"""),
        ],
        """Each dictionary is one row.

Training rows include the label `price`.

Test rows omit `price` because that is what the competition asks us to predict.

The toy data stands in for downloaded CSV files.""",
        """In a real Kaggle run, this step would read files such as `train.csv` and `test.csv` from disk after downloading them.""",
        """- Do not assume network access inside teaching notebooks.
- Training data has labels; competition test data usually does not.
- Keep file paths and data versions documented.
- Toy data is for workflow understanding, not leaderboard performance.""",
    )

    cells += section(
        "5.7.2 Kaggle",
        """A Kaggle competition evaluates predictions submitted in a required file format.

A leaderboard ranks submissions by a chosen metric. A metric is a rule for scoring predictions.""",
        """Kaggle adds practical constraints: train locally, predict on hidden-label test data, submit in the expected format, and avoid overfitting the public leaderboard.""",
        [
            ("md", "A tiny submission row has an ID and prediction."),
            ("code", """submission = [{"Id": 3, "SalePrice": 250.0}]
required_columns = ["Id", "SalePrice"]
list(submission[0].keys()) == required_columns"""),
        ],
        """The submission dictionary has exactly two keys.

`Id` identifies the test example.

`SalePrice` stores the predicted target value.

The column-order check mirrors the idea of matching a competition format.""",
        """Real Kaggle submissions are usually CSV files with exactly the required columns and one row per test example.""",
        """- The leaderboard score is not the same as local validation performance.
- Public leaderboard feedback can be overused.
- Submission format mistakes can fail even if the model is reasonable.
- Competition test labels are hidden from participants.""",
    )

    cells += section(
        "5.7.3 Accessing and Reading the Dataset",
        """Accessing data means loading it into memory. Reading a dataset means separating features, labels, and IDs.

For house prices, features might include rooms and area. The label is the sale price.""",
        """The model should train on features and labels, while IDs are kept for submission but not used as predictive features by default.""",
        [
            ("md", "Extract feature and label tensors from toy rows."),
            ("code", """X = torch.tensor([[row["rooms"], row["area"]] for row in train_rows])
y = torch.tensor([row["price"] for row in train_rows]).reshape(-1, 1)
test_X = torch.tensor([[row["rooms"], row["area"]] for row in test_rows])
test_ids = [row["id"] for row in test_rows]
X.shape, y.shape, test_X.shape, test_ids"""),
        ],
        """The list comprehension reads selected feature columns.

`y` stores training prices as a column tensor.

`test_X` stores test features.

`test_ids` stores IDs for later submission.""",
        """Tabular ML pipelines usually track IDs separately from model inputs.""",
        """- IDs identify rows; they are not automatically useful features.
- Feature order must be consistent between train and test data.
- Labels are available for training rows only.
- Shape checks catch many data-loading errors.""",
    )

    cells += section(
        "5.7.4 Data Preprocessing",
        """Preprocessing prepares raw features for learning.

Common tabular preprocessing includes filling missing values, standardizing numerical features, and encoding categorical features.""",
        """Features with very different scales can make optimization harder. Missing or categorical values must be handled before tensor training.""",
        [
            ("md", "Standardize numerical features."),
            ("code", """mean = X.mean(dim=0, keepdim=True)
std = X.std(dim=0, keepdim=True)
X_scaled = (X - mean) / std
test_scaled = (test_X - mean) / std
X_scaled, test_scaled"""),
        ],
        """`mean` stores the training feature averages.

`std` stores the training feature standard deviations.

Training and test features use the same training mean and standard deviation.

Using test statistics would leak information from the test set.""",
        """Real Kaggle preprocessing often combines numerical standardization and one-hot encoding for categorical columns.""",
        """- Fit preprocessing values on training data only.
- Apply the same transformation to validation and test data.
- Categorical features need encoding.
- Preprocessing choices should be reproducible.""",
    )

    cells += section(
        "5.7.5 Error Measure",
        """House-price competitions often use root mean squared log error, or RMSLE.

A logarithm reduces the dominance of large prices and measures relative error more than absolute error.""",
        """Predicting 10 percent too high on a small house and 10 percent too high on a large house should be treated more similarly than raw dollar errors would.""",
        [
            ("md", "Compute log-RMSE for positive predictions and labels."),
            ("code", """pred = torch.tensor([[210.0], [280.0]])
label = torch.tensor([[200.0], [300.0]])
log_error = torch.log(pred) - torch.log(label)
rmse_log = torch.sqrt((log_error ** 2).mean())
rmse_log"""),
        ],
        """`torch.log` takes the natural logarithm.

The difference compares log predictions with log labels.

Squaring makes errors positive.

The mean and square root produce root mean squared log error.""",
        """The evaluation metric should guide validation and model selection.""",
        """- Log metrics require positive values.
- The competition metric may differ from the training loss.
- Relative error matters more under log metrics.
- Clamp or constrain predictions if they might become nonpositive.""",
    )

    cells += section(
        "5.7.6 k-Fold Cross-Validation",
        """k-fold cross-validation splits training data into `k` parts. Each part becomes validation data once while the others are used for training.

The final validation estimate averages across folds.""",
        """Cross-validation uses limited data more efficiently than one fixed validation split.""",
        [
            ("md", "Create fold indices for a tiny dataset."),
            ("code", """n = 6
k = 3
indices = torch.arange(n)
fold_size = n // k
folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]
folds"""),
        ],
        """`n` is the number of examples.

`k` is the number of folds.

`fold_size` is the number of examples per fold in this simple case.

Each slice becomes one validation fold.""",
        """Kaggle solutions often use cross-validation to compare model settings before producing test predictions.""",
        """- Folds should preserve row-label alignment.
- Real datasets may not divide evenly by `k`.
- Cross-validation is more expensive than one split.
- Test data should not be part of cross-validation.""",
    )

    cells += section(
        "5.7.7 Model Selection",
        """Model selection means choosing hyperparameters or model designs using validation performance.

A hyperparameter is a setting chosen outside gradient training, such as hidden size, learning rate, or weight decay.""",
        """Different settings can produce different validation scores, so selection should be systematic.""",
        [
            ("md", "Choose the model setting with the lowest validation score."),
            ("code", """runs = [
    {"hidden": 8, "score": 0.18},
    {"hidden": 16, "score": 0.15},
    {"hidden": 32, "score": 0.17},
]
best = min(runs, key=lambda run: run["score"])
best"""),
        ],
        """Each dictionary stores one candidate setting and validation score.

`min` chooses the smallest score because lower error is better.

The selected hidden size is chosen by validation performance.""",
        """Good Kaggle workflows record tried settings and avoid repeatedly tuning on public leaderboard feedback.""",
        """- Use validation or cross-validation for selection.
- Keep the test submission process separate.
- Hyperparameters are not ordinary learned weights.
- Record experiments so results are traceable.""",
    )

    cells += section(
        "5.7.8 Submitting Predictions on Kaggle",
        """Submitting predictions means creating a file with test IDs and predicted target values in the required format.

The model predicts labels for test rows whose true labels are hidden.""",
        """A correct submission format is required before scoring can happen.""",
        [
            ("md", "Create toy submission rows from IDs and predictions."),
            ("code", """test_pred = torch.tensor([[260.0]])
submission = []
for row_id, pred_value in zip(test_ids, test_pred.reshape(-1)):
    submission.append({"Id": row_id, "SalePrice": float(pred_value)})
submission"""),
        ],
        """`test_pred` stores predicted prices for test examples.

`zip(test_ids, test_pred.reshape(-1))` pairs each ID with one prediction.

Each output dictionary has the required submission fields.

Real code would write these rows to a CSV file.""",
        """The submission step is deployment-like: the model produces outputs for examples without known labels.""",
        """- The number of predictions must match the number of test IDs.
- Submission column names must match the competition requirement.
- Predictions should be in the metric's valid range.
- Do not include training labels in the submission.""",
    )

    cells += section(
        "5.7.9 Summary and Discussion",
        """A Kaggle workflow includes data access, preprocessing, local validation, model selection, test prediction, and submission formatting.""",
        """The competition setting rewards both modeling and careful engineering discipline.""",
        [
            ("md", "A compact Kaggle workflow checklist."),
            ("code", """workflow = [
    "read train/test data",
    "preprocess using train statistics",
    "validate models locally",
    "predict test labels",
    "write submission",
]
workflow"""),
        ],
        """The checklist names the required workflow order.

Preprocessing must be fit on training data.

Validation should guide model choices.

Submission happens after choosing the model.""",
        """This workflow is also useful outside Kaggle whenever test labels are unavailable at prediction time.""",
        """- Kaggle score is only as meaningful as the competition setup.
- Public leaderboard tuning can overfit.
- Reproducibility matters.
- Offline toy examples teach workflow, not leaderboard strategy.""",
    )

    cells += section(
        "5.7.10 Exercises",
        """These exercises practice the mechanics of a Kaggle-style tabular workflow.""",
        """Small offline examples help separate workflow understanding from competition logistics.""",
        [
            ("md", "Exercise 1: standardize a new feature matrix."),
            ("code", """X2 = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
mean2 = X2.mean(dim=0, keepdim=True)
std2 = X2.std(dim=0, keepdim=True)
(X2 - mean2) / std2"""),
            ("md", "Exercise 2: build a two-row submission list."),
            ("code", """ids = [10, 11]
preds = torch.tensor([100.0, 120.0])
[{"Id": i, "SalePrice": float(p)} for i, p in zip(ids, preds)]"""),
        ],
        """Exercise 1 checks numerical preprocessing.

Exercise 2 checks submission formatting.

Both operations are common in tabular competitions.""",
        """These mechanics support the larger model-training workflow.""",
        """- Use training statistics for scaling.
- Preserve row ID order for submissions.
- Predictions should be numeric.
- Keep competition logistics separate from model concepts.""",
    )

    write_nb("Chapter 5.7 - Predicting House Prices on Kaggle.ipynb", cells)


def main() -> None:
    build_51()
    build_52()
    build_53()
    build_54()
    build_55()
    build_56()
    build_57()


if __name__ == "__main__":
    main()
