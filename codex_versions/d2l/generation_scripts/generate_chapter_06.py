"""Generate Chapter 6 notebooks for the D2L rewrite project.

These notebooks are intentionally notebook-native. They should run top to
bottom in a cloud notebook without creating external files or relying on an IDE.
The rigor comes from tiny drills, shape contracts, assertions, captured failure
cases, and checkpoint prompts inside each notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Chapter 6 - Builders' Guide"
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


def setup_cell(extra: str = "") -> dict:
    body = """import math
from pathlib import Path
import tempfile

import torch
from torch import nn

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

def shape(x):
    return tuple(x.shape)

def count_scalars(parameters):
    return sum(p.numel() for p in parameters)
"""
    if extra.strip():
        body += "\n" + extra.strip() + "\n"
    return code(body)


def title_cell(title: str, purpose: str, outcomes: list[str]) -> dict:
    outcome_lines = "\n".join(f"- {item}" for item in outcomes)
    return md(
        f"""# {title}

{purpose.strip()}

## How to use this notebook

Run the notebook from top to bottom. Every code block is designed to be cloud-runnable and self-contained inside this notebook. The drills are intentionally small: predict the shape or behavior first, run the cell, then read the assertion as the contract you must understand.

## You are done when you can

{outcome_lines}
"""
    )


def checkpoint(title: str, questions: list[str]) -> dict:
    lines = "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, start=1))
    return md(
        f"""## {title} Checkpoint

Answer these before moving on. You do not need a separate notes file for chapters; short answers in markdown cells or in your own study notes are enough.

{lines}
"""
    )


def build_61() -> None:
    cells = [
        title_cell(
            "Chapter 6.1 - Layers and Modules",
            "Chapter 5 treated networks mostly as computations: tensors go in, predictions come out, loss produces gradients, and parameters update. Chapter 6 starts the software step: how do we package those computations so they can be inspected, reused, nested, saved, moved to a GPU, and trained without losing track of the model's state? A PyTorch module is the boundary where a mathematical idea becomes a reliable software object.",
            [
                "explain why deep learning code needs modules instead of loose tensor functions everywhere",
                "explain what `forward` owns and what `Module.__call__` adds around it",
                "trace tensor shapes through a module stack",
                "read a module as a representation pipeline, not just a list of APIs",
                "distinguish simple chains from custom forward logic",
                "debug a shape mismatch inside a module stack",
            ],
        ),
        setup_cell(),
        md(
            """## 6.1.0 The Problem This Notebook Solves

Before this point, it is possible to understand neural networks as loose pieces:

- a weight tensor
- a bias tensor
- a forward computation
- a loss
- a backward pass
- an optimizer step

That is enough for tiny examples, but it does not scale. Once a model has many layers, you need a way to answer basic engineering questions without manually remembering everything:

- Which tensors are part of the model?
- Which pieces are trainable?
- Which computation runs first, second, and third?
- What shape does each layer expect?
- How do we move the whole model to another device?
- How do we save and reload the model's learned state?
- How do we reuse the same block in a larger architecture?

`nn.Module` exists to solve that organization problem. It is not just decoration around a function. It gives PyTorch a standard way to treat model components as objects with structure, state, and behavior.

The mental model for this chapter is:

```text
__init__ defines what the component owns
forward defines what the component does to inputs
module(X) runs the component through PyTorch's normal call path
```

That separation matters. If a tensor is created inside `forward` and not stored on `self`, it is usually temporary computation. If a layer, parameter, or buffer is assigned to `self` in `__init__`, PyTorch can discover it later. Chapter 6.2 will build on this by asking exactly which parameters a module owns. For now, the goal is to understand the module as the basic boundary of model code.
"""
        ),
        md(
            """## 6.1.1 A Module Is a Callable Object With a Forward Contract

Start with the smallest possible module: one that has computation but no trainable parameters. This is important because it separates two ideas that beginners often merge together:

- A module is a reusable computation boundary.
- A parameter is a learned tensor.

A module can have parameters, but it does not need them. A centering layer is still a meaningful model component because it changes the representation passed to later layers. It receives a tensor, computes the mean, subtracts that mean, and returns another tensor with the same shape.

The theoretical meaning is simple: some layers transform the coordinate system of the data without learning anything. They can normalize, reshape, clip, center, mask, or route values. These operations affect what later trainable layers see, so they belong in the model pipeline even when they have no weights.

Before running the cell, predict:

- The output shape should match the input shape.
- The output mean should be zero or extremely close to zero.
- `layer(X)` should call `forward(X)` indirectly.

In normal PyTorch code, call `layer(X)`, not `layer.forward(X)`. The direct `forward` call skips PyTorch's normal module call path. That path is where hooks, wrappers, instrumentation, and other framework features can attach.
"""
        ),
        code(
            """class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


X = torch.tensor([1.0, 2.0, 3.0, 4.0])
layer = CenteredLayer()
Y = layer(X)

print("input:", X)
print("output:", Y)
print("output mean:", Y.mean())

assert shape(Y) == shape(X)
assert torch.allclose(Y.mean(), torch.tensor(0.0))"""
        ),
        md(
            """## 6.1.2 The Call Path Preserves State on the Module

A neural network layer is not only a pure mathematical expression. In software, it is also an object that can remember things. That object state is what lets larger systems inspect the model, save it, move it between devices, and configure training.

There are several kinds of state a module might own:

- ordinary Python attributes, such as a debug flag or the last input shape
- submodules, such as a linear layer inside a larger block
- parameters, which are tensors learned by gradient descent
- buffers, which are persistent tensors that are not usually learned by gradients

This cell uses ordinary Python state only: `last_input_shape`. That makes the idea visible without introducing parameter registration yet.

The discipline is:

- `__init__` should create the structure and long-lived state.
- `forward` should describe the computation for one call.
- temporary tensors created inside `forward` are part of the current computation, not permanent model ownership.

The graceful handoff is that this simple logger prepares you for parameterized modules. A `Linear` layer is not conceptually different in shape: it is also an object that owns state and defines a forward computation. The difference is that some of its state is trainable.
"""
        ),
        code(
            """class ShapeLogger(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def forward(self, X):
        self.last_input_shape = shape(X)
        return X * 2


logger = ShapeLogger()
X = torch.ones(2, 3)
Y = logger(X)

print("last input shape:", logger.last_input_shape)
print("output shape:", shape(Y))

assert logger.last_input_shape == (2, 3)
assert torch.equal(Y, torch.full((2, 3), 2.0))"""
        ),
        md(
            """## 6.1.3 Sequential Modules Are Shape Pipelines

Most beginner networks are representation pipelines. Each layer takes the current representation and produces the next one.

For an MLP, this usually means:

```text
raw features -> hidden representation -> activated hidden representation -> output scores
```

The word "representation" matters. A hidden layer is not just a tensor with a different size. It is the model's current internal description of the input. The first `Linear(4, 5)` layer maps each example from 4 input features into 5 hidden features. `ReLU` keeps the same shape but changes the values by suppressing negative activations. The final `Linear(5, 2)` maps the 5 hidden features into 2 output values.

`nn.Sequential` is appropriate when this story is a straight chain. The key theoretical constraint becomes a mechanical shape handshake:

```text
each layer's output feature count must match the next layer's expected input feature count
```

The shape trace in the next cell is not busywork. It is how you verify that the representation pipeline you intended is the pipeline the code actually built.
"""
        ),
        code(
            """net = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
)

X = torch.randn(3, 4)
shape_trace = []
current = X
for name, module in net.named_children():
    current = module(current)
    shape_trace.append((name, type(module).__name__, shape(current)))

for row in shape_trace:
    print(row)

assert shape_trace == [
    ("0", "Linear", (3, 5)),
    ("1", "ReLU", (3, 5)),
    ("2", "Linear", (3, 2)),
]"""
        ),
        md(
            """## 6.1.4 Custom Forward Logic Is for Branches, Reuse, and Control Flow

`nn.Sequential` is a chain. Real architectures are often not pure chains. They may reuse a block, branch into multiple paths, add a skip connection, or make a decision based on the input.

This is where custom modules become more than syntax. They let you express the actual structure of the model.

The example below is a residual MLP block. Its idea is:

```text
keep the original representation
compute an update to that representation
add the update back to the original
```

The theoretical meaning is that the block does not have to reinvent the whole representation from scratch. It can learn a correction. This is one of the central ideas behind residual networks: if the best thing for a layer to do is "mostly preserve what came in, with a learned adjustment", the architecture makes that easy to express.

The important shape rule is strict: if you add two tensors, they must be compatible shapes. Here the block must return the same shape as its input because `X + update` is the core operation.

Before running the cell, predict:

- Input shape: batch size 2, feature width 4.
- Output shape: still batch size 2, feature width 4.
- Parameter count: two `Linear(4, 4)` layers, each with a 4 by 4 weight and a 4-value bias.
"""
        ),
        code(
            """class ResidualMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, X):
        update = self.block(X)
        return X + update


block = ResidualMLPBlock(width=4)
X = torch.randn(2, 4)
Y = block(X)

print("input shape:", shape(X))
print("output shape:", shape(Y))
print("parameter scalars:", count_scalars(block.parameters()))

assert shape(Y) == shape(X)
assert count_scalars(block.parameters()) == 4 * 4 + 4 + 4 * 4 + 4"""
        ),
        md(
            """## 6.1.5 Break It Deliberately: Bad Shape Handoff

Shape errors are not random PyTorch annoyances. They are failed contracts between representation stages.

In the bad network below, the first linear layer maps 4 features to 5 hidden features. After `ReLU`, the tensor still has 5 features. The final linear layer incorrectly expects 4 features. That means the code is trying to multiply tensors whose inner dimensions do not agree.

The theory-level mistake is: the architecture story is inconsistent. One block says "I produce a 5-feature representation"; the next block says "I consume a 4-feature representation." The runtime error is just PyTorch discovering that contradiction.

This cell catches the error so the notebook keeps running. Read the first line of the error and connect it back to the broken handoff.
"""
        ),
        code(
            """bad_net = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(4, 2),  # wrong: previous layer outputs 5 features, not 4
)

try:
    bad_net(torch.randn(3, 4))
except RuntimeError as err:
    print(type(err).__name__)
    print(str(err).splitlines()[0])
else:
    raise AssertionError("The bad network should have failed.")"""
        ),
        checkpoint(
            "6.1",
            [
                "What problem does `nn.Module` solve once networks become larger than a few loose tensors?",
                "What belongs in `__init__`, and what belongs in `forward`?",
                "Why should normal code call `module(X)` instead of `module.forward(X)`?",
                "Why can a layer with no parameters still be a legitimate module?",
                "In the sequential example, why does the second `Linear` layer need `in_features=5`?",
                "Why is a hidden layer better understood as a representation than as just a matrix output?",
                "What shape must a residual block return if it adds its output back to its input?",
                "When is `nn.Sequential` too limited?",
            ],
        ),
    ]
    write_nb("Chapter 6.1 - Layers and Modules.ipynb", cells)


def build_62() -> None:
    cells = [
        title_cell(
            "Chapter 6.2 - Parameter Management",
            "Chapter 6.1 established that modules are the boundaries of model code. Chapter 6.2 asks what those boundaries own. In deep learning, the learned behavior of a model lives in its parameters, so serious model work requires being able to find them, name them, count them, share them intentionally, freeze them intentionally, and detect when they were never registered at all.",
            [
                "explain why parameters are model state rather than temporary forward-pass values",
                "inspect parameter names, shapes, and scalar counts",
                "connect parameter names to the module tree that owns them",
                "explain how tied parameters differ from copied parameters",
                "show that optimizers only update parameters they receive",
                "catch an unregistered tensor mistake",
            ],
        ),
        setup_cell(),
        md(
            """## 6.2.0 The Problem This Notebook Solves

After you can write modules, the next question is ownership.

A neural network is not only a forward computation. It is also a collection of learned tensors. During training, the optimizer needs a reliable answer to a simple question:

```text
which tensors should I update?
```

PyTorch answers that question through module registration. If a trainable tensor is stored correctly inside a module, it appears in `parameters()` and `named_parameters()`. If it is hidden inside an ordinary list, created only inside `forward`, or stored as a plain tensor when it should be a parameter, PyTorch may not find it.

This is why parameter management is theoretical and mechanical at the same time:

- The theory says the model's learned function is encoded in parameters.
- The software says those parameters must be discoverable by the module tree.
- The optimizer says it can only update tensors it was given.
- The debugger says names and shapes are how you locate the source of a mistake.

The handoff from 6.1 is direct: modules define ownership boundaries; parameter management inspects what those boundaries actually own.
"""
        ),
        md(
            """## 6.2.1 Parameters Live Inside a Module Tree

When a model is nested, parameter names become paths through the module tree. In a `Sequential`, numeric names such as `0.weight` and `3.0.weight` are not arbitrary. They tell you where the parameter lives.

The conceptual point is that a large model is not a bag of unnamed arrays. It is a hierarchy of components, and each trainable tensor has a location in that hierarchy. This matters when you:

- debug a layer that has the wrong shape
- freeze only part of a model
- initialize only certain layer types
- give different optimizer settings to different parameter groups
- compare model sizes

Before running the cell, predict the scalar count:

```text
Linear(4, 6): weight 6 by 4, bias 6
Linear(6, 3): weight 3 by 6, bias 3
Linear(3, 1): weight 1 by 3, bias 1
```

The code prints both tensor shapes and scalar counts because both are useful. A single weight matrix is one parameter tensor, but it can contain many scalar values.
"""
        ),
        code(
            """net = nn.Sequential(
    nn.Linear(4, 6),
    nn.ReLU(),
    nn.Sequential(
        nn.Linear(6, 3),
        nn.ReLU(),
    ),
    nn.Linear(3, 1),
)

for name, param in net.named_parameters():
    print(f"{name:10s}", shape(param), "scalars=", param.numel())

total = count_scalars(net.parameters())
print("total trainable scalars:", total)

assert total == (6 * 4 + 6) + (3 * 6 + 3) + (1 * 3 + 1)"""
        ),
        md(
            """## 6.2.2 `state_dict` Is Saved Tensor State, Not Just Trainable Parameters

`named_parameters()` answers "what trainable tensors does this model expose?" `state_dict()` answers a broader save/load question: "what tensor state would I need to serialize to recreate this model's current behavior?"

For a simple MLP, those two lists look the same because the model only has trainable weights and biases. Later, they can differ. Batch normalization, for example, has learned parameters but also running mean and variance buffers. Those buffers affect inference even though they are not usually updated by gradients.

The theoretical distinction is:

- parameters are learned by optimization
- buffers are persistent model state
- both can affect the model's output

This section prepares the handoff to Chapter 6.6, where saving and loading depends on understanding `state_dict`.
"""
        ),
        code(
            """state = net.state_dict()
print("state_dict keys:")
for key, value in state.items():
    print(f"{key:10s}", shape(value))

parameter_names = [name for name, _ in net.named_parameters()]
state_names = list(state.keys())

assert parameter_names == state_names"""
        ),
        md(
            """## 6.2.3 Tied Parameters Share One Object

Parameter sharing is an architectural claim, not just a memory trick.

If two parts of a model use equal copied weights, they start with the same values but can drift apart during training. If they use the same parameter object, there is only one learned tensor. Every use contributes gradient to that same tensor, and every optimizer step updates the one shared object.

This matters in real models. Weight tying can express a belief like:

```text
these two computations should use the same learned rule
```

The most important mechanical test is object identity, not value equality. `a == b` asks whether values match. `a is b` asks whether both names point to the exact same object.

Before running the cell, predict:

- The shared layer should appear only once in `named_parameters()`.
- The two positions in the model should point to the same weight object.
- The shared weight should receive a gradient after backward.
"""
        ),
        code(
            """shared = nn.Linear(2, 2, bias=False)
tied_net = nn.Sequential(shared, nn.ReLU(), shared)

print("same weight object:", tied_net[0].weight is tied_net[2].weight)
print("named parameters:", [name for name, _ in tied_net.named_parameters()])

X = torch.ones(1, 2)
loss = tied_net(X).sum()
loss.backward()

print("shared gradient:")
print(shared.weight.grad)

assert tied_net[0].weight is tied_net[2].weight
assert shared.weight.grad is not None
assert torch.isfinite(shared.weight.grad).all()"""
        ),
        md(
            """## 6.2.4 Freezing Means `requires_grad=False`

Freezing is the opposite of learning for a selected part of the model. The frozen tensor still participates in the forward pass, so it still affects predictions. The difference is that autograd does not compute gradients for it, and the optimizer should not update it.

This is useful when:

- using a pretrained feature extractor
- training only a new classification head
- comparing what happens when part of a model is held fixed
- preventing a stable component from being disturbed

The theory-level distinction is important:

```text
frozen does not mean removed
frozen means used in forward computation but not adapted by gradient descent
```

The cell freezes the first linear layer. The final linear layer remains trainable, so its gradient should exist after backward.
"""
        ),
        code(
            """model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))

for param in model[0].parameters():
    param.requires_grad_(False)

trainable = [(name, shape(p)) for name, p in model.named_parameters() if p.requires_grad]
frozen = [(name, shape(p)) for name, p in model.named_parameters() if not p.requires_grad]

print("trainable:", trainable)
print("frozen:", frozen)

optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)

X = torch.randn(5, 3)
y = torch.randn(5, 1)
loss = ((model(X) - y) ** 2).mean()
optimizer.zero_grad()
loss.backward()

assert model[0].weight.grad is None
assert model[2].weight.grad is not None"""
        ),
        md(
            """## 6.2.5 Break It Deliberately: Plain Tensor Is Not Registered

This failure is subtle because the forward computation can look mathematically valid. The layer has a tensor called `weight`, and that tensor even has `requires_grad=True`. But PyTorch's module machinery does not treat every tensor attribute as a registered parameter.

For optimizers and save/load tools to discover a trainable tensor, it should be an `nn.Parameter` assigned to `self`. That assignment is the registration event.

The conceptual mistake is ownership ambiguity. The code creates a tensor, but it does not clearly tell PyTorch:

```text
this tensor is part of the learned model state
```

That is why the optimizer rejects the model. It sees no registered parameters to update.
"""
        ),
        code(
            """class BadLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(3, 1, requires_grad=True)  # not registered

    def forward(self, X):
        return X @ self.weight


bad = BadLinear()
print("named parameters:", list(bad.named_parameters()))

try:
    torch.optim.SGD(bad.parameters(), lr=0.1)
except ValueError as err:
    print(type(err).__name__)
    print(str(err))
else:
    raise AssertionError("An optimizer should reject an empty parameter list.")"""
        ),
        checkpoint(
            "6.2",
            [
                "Why is parameter ownership a module-tree question rather than only a tensor question?",
                "What is the difference between counting parameter tensors and counting scalar parameter values?",
                "Why can `state_dict()` include tensor state beyond trainable parameters in larger models?",
                "Why does a tied layer appear once in `named_parameters()` even if it is used twice?",
                "Why is object identity more important than value equality for tied parameters?",
                "What exactly changes when a parameter is frozen?",
                "Why did `BadLinear` have no registered parameters?",
            ],
        ),
    ]
    write_nb("Chapter 6.2 - Parameter Management.ipynb", cells)


def build_63() -> None:
    cells = [
        title_cell(
            "Chapter 6.3 - Parameter Initialization",
            "Training does not begin from nowhere. Before the first batch, every learned tensor already has values, and those starting values shape the early forward signals and backward gradients. Initialization is the bridge between architecture design and optimization behavior.",
            [
                "explain why starting values affect optimization even before learning begins",
                "inspect initialized weights and biases",
                "explain why all-zero weights create symmetry problems",
                "connect activation scale to gradient usability",
                "compare tiny, huge, Xavier, and Kaiming-style initializations",
                "apply an initializer across a module tree",
            ],
        ),
        setup_cell(),
        md(
            """## 6.3.0 The Problem This Notebook Solves

A model architecture says what computations are possible. Initialization decides where optimization starts inside that space of possible computations.

For a linear layer, the weights and biases are trainable, but before training they are just initial guesses. Bad guesses can make learning unnecessarily hard:

- If weights are too small, signals can shrink as they pass through many layers.
- If weights are too large, activations and gradients can explode.
- If many units start exactly the same, they can receive the same gradients and learn redundant features.

The goal is not to find magical starting values. The goal is to start in a regime where different units can learn different things and gradients remain numerically useful.

This connects back to Chapter 5's stability discussion. There, instability was about forward and backward propagation through deep networks. Here, we look at the software mechanism that chooses the initial tensor values.
"""
        ),
        md(
            """## 6.3.1 Inspect Before You Trust

Before reasoning about initialization, look at the actual tensor statistics. The exact random values do not matter. The scale, shape, and finiteness matter.

The weight shape of `nn.Linear(8, 4)` is `(4, 8)` because the layer produces 4 output features from 8 input features. Each output feature owns one row of 8 incoming weights. The bias shape is `(4,)` because each output feature gets one bias.

The mean and standard deviation are quick sanity checks:

- mean near zero usually means the layer does not start with a strong systematic positive or negative preference
- standard deviation tells you the typical weight size
- finite values rule out broken initialization

This is the first habit: inspect before trusting an initializer.
"""
        ),
        code(
            """layer = nn.Linear(8, 4)

print("weight shape:", shape(layer.weight))
print("bias shape:", shape(layer.bias))
print("weight mean:", float(layer.weight.mean()))
print("weight std:", float(layer.weight.std()))

assert shape(layer.weight) == (4, 8)
assert shape(layer.bias) == (4,)
assert torch.isfinite(layer.weight).all()"""
        ),
        md(
            """## 6.3.2 Zero Weights Are Mechanically Bad for Hidden Units

All-zero weights are not just "small." They create symmetry.

If several hidden units start with the same incoming weights and the same bias, they compute the same output for the same input. If they also receive the same gradient pattern, they can keep moving together. The model wastes capacity because multiple units behave like copies of each other.

Biases are different. A zero bias does not usually force two units to have identical incoming feature detectors if their weights are different. That is why zero bias initialization is common, while all-zero weight initialization is usually a bad default for hidden layers.

In this tiny example, every output unit produces the exact same value: zero. The point is not that zero output is always bad; the point is that the hidden units are indistinguishable.
"""
        ),
        code(
            """layer = nn.Linear(3, 4)
with torch.no_grad():
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)

X = torch.tensor([[1.0, 2.0, 3.0]])
Y = layer(X)

print(Y)

assert torch.equal(Y, torch.zeros(1, 4))
assert torch.equal(Y[:, 0], Y[:, 1])"""
        ),
        md(
            """## 6.3.3 Initialization Scale Changes Activation Scale

Deep networks repeatedly transform representations. Even before training, each layer changes the scale of the tensor it receives.

If each layer shrinks the representation too much, later layers receive almost no signal. If each layer amplifies the representation too much, values can become huge. In both cases, gradients can become hard to use.

This cell does not train. That is intentional. It isolates one question:

```text
what happens to activation scale during a forward pass through several randomly initialized layers?
```

The tiny initialization should tend to shrink activations across layers. The huge initialization should tend to expand them. This gives a concrete reason fan-aware initializers exist.
"""
        ),
        code(
            """def activation_std_after_stack(init_std):
    torch.manual_seed(0)
    layers = []
    for _ in range(5):
        layer = nn.Linear(64, 64)
        with torch.no_grad():
            nn.init.normal_(layer.weight, mean=0.0, std=init_std)
            nn.init.zeros_(layer.bias)
        layers += [layer, nn.ReLU()]

    X = torch.randn(256, 64)
    stats = []
    for layer in layers:
        X = layer(X)
        if isinstance(layer, nn.Linear):
            stats.append(float(X.std()))
    return stats


tiny_stats = activation_std_after_stack(0.01)
huge_stats = activation_std_after_stack(1.0)

print("tiny init stds:", tiny_stats)
print("huge init stds:", huge_stats)

assert tiny_stats[-1] < tiny_stats[0]
assert huge_stats[-1] > huge_stats[0]"""
        ),
        md(
            """## 6.3.4 Use Fan-Aware Initializers for Common Layers

Fan-aware initializers choose weight scale using the layer's width.

Plain-English idea:

- A layer with many inputs adds together many weighted values.
- If the weights are not scaled carefully, the sum can become too large or too small.
- The right scale depends on how many values flow into or out of the layer.

Xavier initialization is commonly associated with activations that are roughly symmetric around zero. Kaiming initialization is commonly used with ReLU-like activations because ReLU discards negative values and changes signal flow.

You do not need to memorize the derivation here. The rigor is knowing what problem these initializers solve: they try to preserve usable signal scale through the network at the start of training.
"""
        ),
        code(
            """xavier_layer = nn.Linear(128, 64)
kaiming_layer = nn.Linear(128, 64)

with torch.no_grad():
    nn.init.xavier_uniform_(xavier_layer.weight)
    nn.init.zeros_(xavier_layer.bias)
    nn.init.kaiming_uniform_(kaiming_layer.weight, nonlinearity="relu")
    nn.init.zeros_(kaiming_layer.bias)

print("xavier std:", float(xavier_layer.weight.std()))
print("kaiming std:", float(kaiming_layer.weight.std()))

assert torch.isfinite(xavier_layer.weight).all()
assert torch.isfinite(kaiming_layer.weight).all()"""
        ),
        md(
            """## 6.3.5 Apply Initialization Across a Module Tree

Real models contain many modules. You do not want to manually initialize each layer one at a time, and you definitely do not want to accidentally initialize a `ReLU` as if it had weights.

`net.apply(fn)` walks through the module tree. The initializer receives each module, checks what type it is, and acts only when appropriate.

This is where Chapters 6.1 and 6.2 connect:

- 6.1 gave you the module tree.
- 6.2 gave you parameter ownership.
- 6.3 now uses the module tree to mutate parameters deliberately.

The type check is not optional ceremony. It is the guardrail that keeps model-wide initialization from touching objects that do not own the expected parameter shapes.
"""
        ),
        code(
            """def init_mlp(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


net = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

net.apply(init_mlp)

for name, param in net.named_parameters():
    print(name, shape(param), float(param.std()) if param.ndim > 1 else float(param.sum()))

assert torch.equal(net[0].bias, torch.zeros_like(net[0].bias))
assert torch.equal(net[2].bias, torch.zeros_like(net[2].bias))"""
        ),
        md(
            """## 6.3.6 Break It Deliberately: Initialize Too Late

Initialization is supposed to happen before learning. Reinitialization after training is not a harmless refresh; it overwrites the learned function.

The theory-level mistake is confusing setup with training. During training, the optimizer gradually changes parameters to reduce loss. If you call an initializer afterward, you replace those learned values with a new starting point. The model may still have the same architecture, but its learned behavior is gone.

This cell demonstrates the mutation directly by saving a copy of the weight, overwriting it, and proving the value changed.
"""
        ),
        code(
            """layer = nn.Linear(2, 1)
before = layer.weight.detach().clone()

with torch.no_grad():
    nn.init.ones_(layer.weight)

after = layer.weight.detach().clone()

print("before:", before)
print("after:", after)

assert not torch.allclose(before, after)
assert torch.equal(after, torch.ones_like(after))"""
        ),
        checkpoint(
            "6.3",
            [
                "Why is initialization part of optimization theory, not just software setup?",
                "Why are zero biases usually less dangerous than zero weights?",
                "What symmetry problem do all-zero hidden weights create?",
                "What did the tiny and huge initialization drill show mechanically?",
                "What problem are Xavier and Kaiming initializers trying to solve?",
                "Why should an initializer check `isinstance(module, nn.Linear)` before touching `.weight`?",
                "Why is reinitializing after training equivalent to discarding learned parameters?",
            ],
        ),
    ]
    write_nb("Chapter 6.3 - Parameter Initialization.ipynb", cells)


def build_64() -> None:
    cells = [
        title_cell(
            "Chapter 6.4 - Lazy Initialization",
            "Most layers need to know their parameter shapes before they can store weights. Lazy initialization is a controlled exception: it lets a layer postpone part of its shape decision until real input reveals the missing dimension.",
            [
                "explain why parameter shape depends on input shape",
                "identify uninitialized lazy parameters",
                "explain when lazy layer shapes become known",
                "distinguish convenience from a different model type",
                "use lazy layers in a clean-restartable notebook",
                "debug a lazy layer after it has locked onto an input size",
            ],
        ),
        setup_cell(),
        md(
            """## 6.4.0 The Problem This Notebook Solves

In a normal linear layer, the weight shape is known immediately:

```text
Linear(input_features, output_features)
weight shape: output_features by input_features
```

That means the layer designer must know the input feature count while writing the model. Sometimes this is easy. In an MLP over a fixed 10-feature table, the number is obvious. Sometimes it is annoying. After convolutions, pooling, flattening, or other shape-changing operations, the final feature count may take careful tracing.

Lazy initialization lets the model say:

```text
I know how many output features I want.
I will infer the input feature count when I see the first batch.
```

The theory is not that lazy layers learn differently. They do not. The theory is about delaying a software commitment. Before the first forward pass, the layer does not yet know enough to create an ordinary weight matrix. After the first forward pass, it has a real parameter shape and behaves like a regular layer.

The handoff from 6.3 is direct: initialization chooses parameter values, but parameter values require parameter shapes. Lazy modules postpone shape creation, then initialize once the missing shape is known.
"""
        ),
        md(
            """## 6.4.1 Before the First Forward Pass, Some Shapes Are Unknown

`nn.LazyLinear(3)` knows that it should produce 3 output features. It does not know how many input features it will receive. Without that input feature count, it cannot create a complete weight matrix.

This is why inspecting a lazy weight too early is different from inspecting a normal parameter. The layer is not broken; it is incomplete by design.

Before running the cell, predict:

- Before the first forward pass, the weight shape should not be fully available.
- If the first input has shape `(2, 4)`, the layer sees 4 input features.
- The final weight shape should become `(3, 4)`.
- The output shape should become `(2, 3)`.
"""
        ),
        code(
            """lazy = nn.LazyLinear(3)

print("weight object type before forward:", type(lazy.weight).__name__)
try:
    print(lazy.weight.shape)
except RuntimeError as err:
    print("shape unavailable before forward:")
    print(str(err).splitlines()[0])

X = torch.randn(2, 4)
Y = lazy(X)

print("output shape:", shape(Y))
print("weight shape after forward:", shape(lazy.weight))

assert shape(Y) == (2, 3)
assert shape(lazy.weight) == (3, 4)"""
        ),
        md(
            """## 6.4.2 Lazy Layers Are Useful After Shape-Changing Layers

Flattening image-like tensors is a common place where lazy initialization is convenient.

An image batch shaped `(5, 1, 4, 4)` contains:

```text
5 examples
1 channel per example
4 rows
4 columns
```

After flattening, each example has `1 * 4 * 4 = 16` features. You can compute that by hand here, but in deeper convolutional networks the spatial size may change several times. A lazy linear layer can infer the flattened feature count from a sample forward pass.

The conceptual tradeoff is:

- lazy layer: easier to write when shape arithmetic is annoying
- explicit layer: clearer contract when teaching or debugging

The cell below uses lazy initialization only as a convenience. Once initialized, the layer owns ordinary parameters.
"""
        ),
        code(
            """net = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(10),
)

X = torch.randn(5, 1, 4, 4)
Y = net(X)

print("flattened feature count:", net[1].weight.shape[1])
print("output shape:", shape(Y))

assert shape(Y) == (5, 10)
assert shape(net[1].weight) == (10, 16)"""
        ),
        md(
            """## 6.4.3 Explicit Dimensions Are Often Clearer in Teaching Code

Explicit dimensions make the architecture contract visible in the code. That is why teaching notebooks often prefer them even when lazy layers would work.

This comparison is important because lazy initialization can hide shape reasoning if used too early. You should still be able to explain why the flattened feature count is 16. Lazy layers should reduce boilerplate, not replace understanding.

The handoff to Chapter 7 is worth noticing: CNNs create many shape-changing steps. Lazy layers can make prototypes easier, but serious CNN debugging still requires tracing height, width, and channel count.
"""
        ),
        code(
            """explicit = nn.Sequential(
    nn.Flatten(),
    nn.Linear(16, 10),
)

X = torch.randn(5, 1, 4, 4)
Y = explicit(X)

print("explicit weight shape:", shape(explicit[1].weight))
print("lazy weight shape:", shape(net[1].weight))

assert shape(Y) == (5, 10)
assert shape(explicit[1].weight) == shape(net[1].weight)"""
        ),
        md(
            """## 6.4.4 Break It Deliberately: Change Feature Count After Initialization

Lazy does not mean permanently flexible.

After the first forward pass, the layer has committed to a weight matrix. If the first input had 4 features, the weight expects 4 features from then on. Passing 5 features later is not a new lazy inference event; it is a shape mismatch.

This is the central warning:

```text
lazy initialization delays the first shape decision
it does not remove shape contracts
```
"""
        ),
        code(
            """lazy = nn.LazyLinear(3)
lazy(torch.randn(2, 4))

try:
    lazy(torch.randn(2, 5))
except RuntimeError as err:
    print(type(err).__name__)
    print(str(err).splitlines()[0])
else:
    raise AssertionError("The initialized lazy layer should reject a new feature count.")"""
        ),
        checkpoint(
            "6.4",
            [
                "Why can a linear layer not create its full weight matrix without knowing input feature count?",
                "Which dimension is unknown in `nn.LazyLinear(3)` before the first forward pass?",
                "What input shape made the lazy weight become `(3, 4)`?",
                "Why is lazy initialization a software convenience rather than a new learning rule?",
                "Why can a lazy layer still fail later with a shape mismatch?",
                "When would you prefer explicit dimensions over lazy initialization?",
            ],
        ),
    ]
    write_nb("Chapter 6.4 - Lazy Initialization.ipynb", cells)


def build_65() -> None:
    cells = [
        title_cell(
            "Chapter 6.5 - Custom Layers",
            "Custom layers are where you extend PyTorch without leaving PyTorch's model system. The goal is to write new computations that still behave like normal modules: inspectable, trainable when appropriate, movable across devices, and saveable through `state_dict`.",
            [
                "explain when a new operation deserves to become a module",
                "write parameter-free and parameterized layers",
                "register trainable tensors with `nn.Parameter`",
                "register persistent non-trainable tensor state as a buffer",
                "prove a custom layer parameter updates after one optimizer step",
            ],
        ),
        setup_cell(),
        md(
            """## 6.5.0 The Problem This Notebook Solves

Built-in layers cover common operations, but research code and serious engineering often need custom behavior. The mistake is to think custom behavior means abandoning PyTorch conventions.

A good custom layer should still answer the same questions as a built-in layer:

- What computation runs in `forward`?
- Which tensors are trainable parameters?
- Which tensors are persistent but not trainable?
- What input and output shapes does the layer promise?
- Will an optimizer, device move, or save/load call discover the right state?

This notebook builds three categories:

```text
parameter-free layer: computation only
parameterized layer: computation plus learned tensors
buffer-owning layer: computation plus persistent non-learned tensor state
```

The handoff from earlier Chapter 6 sections is deliberate: 6.1 gave modules, 6.2 gave parameter ownership, 6.3 gave initialization, and 6.4 gave shape creation. Custom layers combine all of those ideas.
"""
        ),
        md(
            """## 6.5.1 Parameter-Free Layers Still Belong in Module Pipelines

Some transformations have no learned weights, but they still change the representation passed to later layers. That makes them legitimate model components.

`RowCenter` subtracts each row's own mean. If each row is one example, the layer removes that example's average feature level while preserving its relative feature differences.

The theory-level point is that not every useful model operation is learned. Some operations impose a fixed transformation that changes what the learned layers have to handle.

Before running the cell, predict:

- Output shape should match input shape.
- Each row's mean should become zero.
- The layer should expose no trainable parameters.
"""
        ),
        code(
            """class RowCenter(nn.Module):
    def forward(self, X):
        return X - X.mean(dim=1, keepdim=True)


layer = RowCenter()
X = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
Y = layer(X)

print(Y)
print("row means:", Y.mean(dim=1))
print("parameters:", list(layer.parameters()))

assert shape(Y) == shape(X)
assert torch.allclose(Y.mean(dim=1), torch.zeros(2))
assert list(layer.parameters()) == []"""
        ),
        md(
            """## 6.5.2 Trainable Custom Layers Use `nn.Parameter`

This custom layer is a small linear layer written by hand. The computation is familiar:

```text
input representation @ weight + bias -> output representation
```

The important part is not the matrix multiply itself. The important part is how the learnable tensors are owned. Assigning an `nn.Parameter` to `self.weight` or `self.bias` tells PyTorch:

```text
this tensor is learned model state
include it in parameters()
include it in state_dict()
move it when the module moves devices
compute gradients for it during backward
```

That is the difference between a custom layer that merely computes and a custom layer that participates correctly in training.
"""
        ),
        code(
            """class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        return X @ self.weight + self.bias


layer = MyLinear(3, 2)
X = torch.randn(4, 3)
Y = layer(X)

print("output shape:", shape(Y))
print("named parameters:", [(name, shape(p)) for name, p in layer.named_parameters()])

assert shape(Y) == (4, 2)
assert [name for name, _ in layer.named_parameters()] == ["weight", "bias"]"""
        ),
        md(
            """## 6.5.3 Prove the Custom Parameters Actually Update

Never trust a custom trainable layer just because the forward pass runs. A correct custom layer must satisfy three contracts:

```text
forward produces the expected shape
backward creates gradients for registered parameters
optimizer.step mutates those parameters
```

This cell runs one synthetic update only. It is not trying to learn a meaningful task. It is a wiring test. If this test fails, a longer training experiment would only hide the bug under more code.

This style should feel familiar from serious integration work, but it stays inside the notebook: tiny proof first, larger use later.
"""
        ),
        code(
            """torch.manual_seed(0)
layer = MyLinear(3, 1)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

X = torch.randn(8, 3)
y = torch.randn(8, 1)

before = layer.weight.detach().clone()
pred = layer(X)
loss = ((pred - y) ** 2).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()

after = layer.weight.detach().clone()

print("loss:", float(loss.detach()))
print("grad shape:", shape(layer.weight.grad))
print("weight changed:", not torch.allclose(before, after))

assert shape(layer.weight.grad) == shape(layer.weight)
assert not torch.allclose(before, after)"""
        ),
        md(
            """## 6.5.4 Buffers Are Persistent State, Not Learned Parameters

Some tensor state belongs to the module but should not be learned by gradient descent. That is what buffers are for.

A buffer is appropriate when a tensor should:

- move with the module when calling `.to(device)`
- appear in `state_dict`
- persist as part of the model's behavior
- not appear in `parameters()`
- not be updated by the optimizer

Running statistics in normalization layers are the classic real example. The `FixedScale` example is simpler: the scale tensor is persistent state, but it is not something the optimizer should learn.

The theoretical distinction is clean:

```text
parameter: learned state
buffer: persistent non-learned state
ordinary attribute: Python-side state not automatically treated as model tensor state
```
"""
        ),
        code(
            """class FixedScale(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.register_buffer("scale", torch.ones(features))

    def forward(self, X):
        return X * self.scale


layer = FixedScale(3)
print("parameters:", list(layer.named_parameters()))
print("buffers:", [(name, shape(buf)) for name, buf in layer.named_buffers()])
print("state_dict:", list(layer.state_dict().keys()))

assert list(layer.named_parameters()) == []
assert list(layer.state_dict().keys()) == ["scale"]"""
        ),
        md(
            """## 6.5.5 Break It Deliberately: Forget `nn.Parameter`

This layer looks plausible because the tensor has `requires_grad=True`. But it is still not registered as a module parameter.

The issue is not whether autograd can theoretically compute a gradient for the tensor. The issue is whether PyTorch's model tools can discover that tensor as part of the model's learned state.

The optimizer asks the module for registered parameters. Since there are none, the optimizer has nothing to update. This is the same failure mode as 6.2, now inside the custom-layer context.
"""
        ),
        code(
            """class AlmostLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(3, 1, requires_grad=True)

    def forward(self, X):
        return X @ self.weight


layer = AlmostLinear()
print("parameter list:", list(layer.named_parameters()))

try:
    torch.optim.SGD(layer.parameters(), lr=0.1)
except ValueError as err:
    print(type(err).__name__)
    print(str(err))
else:
    raise AssertionError("The optimizer should reject an empty parameter list.")"""
        ),
        checkpoint(
            "6.5",
            [
                "What makes a custom layer compatible with the rest of PyTorch's model system?",
                "Why can `RowCenter` be used in a model even though it has no parameters?",
                "What does `nn.Parameter` change compared with a plain tensor?",
                "How did the optimizer-step drill prove the custom layer is wired correctly?",
                "What is the difference between a parameter, a buffer, and an ordinary attribute?",
                "When should tensor state be a buffer instead of a parameter?",
            ],
        ),
    ]
    write_nb("Chapter 6.5 - Custom Layers.ipynb", cells)


def build_66() -> None:
    cells = [
        title_cell(
            "Chapter 6.6 - File I/O",
            "Training changes model state over time. File I/O is how that state survives beyond the current notebook session. This chapter stays notebook-native, but it teaches the same save/load concepts needed for reliable experiments.",
            [
                "explain why saving values is different from saving architecture code",
                "save and load tensors in notebook-safe temporary paths",
                "save and restore a model with `state_dict`",
                "resume optimizer state from a checkpoint dictionary",
                "recognize architecture mismatch errors while loading",
            ],
        ),
        setup_cell(
            """def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)"""
        ),
        md(
            """## 6.6.0 The Problem This Notebook Solves

So far, every notebook cell has lived in memory. Once the kernel restarts, ordinary Python variables disappear. That is fine for tiny drills, but learned model state should not vanish just because a session ends.

File I/O answers several different needs:

- save a tensor so it can be reused
- save model weights so predictions can be reproduced
- save optimizer state so training can resume with its momentum or other internal state
- save metadata so a checkpoint is interpretable later

The most important theoretical distinction is:

```text
architecture: the code that defines the computation
state: the tensor values currently inside that computation
checkpoint: state plus enough context to resume or interpret work
```

This notebook uses temporary paths so it remains cloud-safe. The concept transfers directly to persistent paths later.
"""
        ),
        md(
            """## 6.6.1 Save and Load Tensors

The smallest save/load unit is a tensor. Saving a tensor writes its values, dtype, shape, and enough PyTorch metadata to reconstruct it.

This is not a model checkpoint yet. It is just serialization of one object. Starting here keeps the mechanism visible before model state dictionaries add naming and architecture compatibility.

Temporary directories keep this notebook cloud-safe. The files exist only while the cell runs.
"""
        ),
        code(
            """with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "tensor.pt"
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save(original, path)
    loaded = torch_load(path)

print("loaded tensor:")
print(loaded)

assert torch.equal(original, loaded)
assert shape(loaded) == (2, 3)"""
        ),
        md(
            """## 6.6.2 Save and Restore Model Weights

Saving a model's `state_dict` saves tensor state, not the architecture definition. That means loading has two steps:

```text
recreate the same architecture in code
load the saved tensor values into that architecture
```

The theory is that the learned function depends on both pieces:

- architecture decides how tensors are used
- state decides the learned values used by that architecture

If both match, the restored model should produce the same predictions for the same input. The cell proves that by comparing predictions before and after loading.
"""
        ),
        code(
            """torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
X = torch.randn(5, 3)
before = model(X).detach()

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "model_state.pt"
    torch.save(model.state_dict(), path)

    restored = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    restored.load_state_dict(torch_load(path, map_location="cpu"))
    after = restored(X).detach()

print("max prediction difference:", float((before - after).abs().max()))

assert torch.allclose(before, after)"""
        ),
        md(
            """## 6.6.3 Checkpoints Usually Need More Than Model Weights

A model `state_dict` is enough for inference. It is often not enough for training resume.

Optimizers can have their own state. Momentum is the simplest example: the optimizer remembers a running update direction. Adam-style optimizers remember even more. If you reload only model weights and create a fresh optimizer, you may not truly resume the same training process.

A checkpoint is usually a dictionary with several pieces:

```text
model_state: learned model tensor values
optimizer_state: optimizer internal state
epoch or step: where training stopped
metadata: short context for humans and future code
```

This is still a notebook drill, not a persistent experiment artifact. The purpose is to understand what belongs in a checkpoint before larger training runs depend on it.
"""
        ),
        code(
            """torch.manual_seed(1)
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

X = torch.randn(4, 2)
y = torch.randn(4, 1)
loss = ((model(X) - y) ** 2).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": 1,
            "description": "one synthetic regression step",
        },
        path,
    )

    loaded = torch_load(path, map_location="cpu")
    new_model = nn.Linear(2, 1)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1, momentum=0.9)
    new_model.load_state_dict(loaded["model_state"])
    new_optimizer.load_state_dict(loaded["optimizer_state"])

print("checkpoint keys:", sorted(loaded.keys()))
print("loaded epoch:", loaded["epoch"])

assert loaded["epoch"] == 1
assert "state" in loaded["optimizer_state"]"""
        ),
        md(
            """## 6.6.4 Break It Deliberately: Load Into the Wrong Architecture

A saved state dictionary is not magic. The receiving architecture must have compatible parameter names and shapes.

If the saved layer has weight shape for 3 input features and the new layer expects 4 input features, PyTorch refuses to load it. That refusal is good. Silent loading would mean the model's learned state no longer matches the computation that uses it.

The theory-level mistake is mixing state from one function family with architecture code from another. The shape mismatch is the mechanical evidence.
"""
        ),
        code(
            """source = nn.Linear(3, 1)

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "linear.pt"
    torch.save(source.state_dict(), path)

    wrong = nn.Linear(4, 1)
    try:
        wrong.load_state_dict(torch_load(path, map_location="cpu"))
    except RuntimeError as err:
        print(type(err).__name__)
        print(str(err).splitlines()[0])
    else:
        raise AssertionError("Loading should fail because the input feature count changed.")"""
        ),
        checkpoint(
            "6.6",
            [
                "What is the difference between architecture and state?",
                "What does a model `state_dict` save, and what does it not save?",
                "Why did predictions match after loading into the same architecture?",
                "Why can optimizer state matter when resuming training?",
                "What does a size mismatch during loading usually mean?",
            ],
        ),
    ]
    write_nb("Chapter 6.6 - File I-O.ipynb", cells)


def build_67() -> None:
    cells = [
        title_cell(
            "Chapter 6.7 - GPUs",
            "Deep learning computation happens on devices: usually CPU or GPU. Device management is the discipline of keeping tensors, parameters, buffers, losses, and logging values in the right place at the right time.",
            [
                "explain why hardware placement is part of the tensor contract",
                "choose a CPU/GPU device safely",
                "move tensors and modules to the same device",
                "understand device mismatch errors",
                "move tensors back to CPU for logging or NumPy conversion",
            ],
        ),
        setup_cell(),
        md(
            """## 6.7.0 The Problem This Notebook Solves

A tensor is not only a shape and dtype. It also lives somewhere.

On a CPU-only machine, all computation happens on CPU. On a GPU machine, large tensor operations can be much faster on GPU, but only if the relevant tensors and model parameters are on that GPU. PyTorch generally does not hide this from you because moving data across devices has real cost and can create ambiguity.

The core rule is:

```text
tensors that participate in one operation usually must live on the same device
```

This matters for model code:

- model parameters live on a device
- input batches live on a device
- losses and outputs live where they were computed
- logging often wants CPU scalar values

This notebook is CPU-safe. If a GPU exists, it uses it. If not, it still teaches the same placement logic.
"""
        ),
        md(
            """## 6.7.1 Choose a Device Without Assuming a GPU Exists

Cloud notebooks vary. Some sessions have CUDA. Some do not. Good teaching code should not crash just because no GPU is available.

The pattern is:

```text
if CUDA is available, use cuda
otherwise, use cpu
```

This is not only convenience. It makes the rest of the notebook express a single device contract: every tensor and module should be moved to `device`, whatever that device happens to be.
"""
        ),
        code(
            """has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

print("cuda available:", has_cuda)
print("selected device:", device)
print("visible cuda devices:", torch.cuda.device_count())

assert device.type in {"cpu", "cuda"}"""
        ),
        md(
            """## 6.7.2 Tensors Live on One Device at a Time

A tensor's device is part of its runtime identity. Two tensors with the same values and shapes are still not directly compatible for most operations if one is on CPU and the other is on GPU.

This is because CPU memory and GPU memory are physically different places. Adding tensors requires the operation to access both operands. PyTorch expects them to be colocated unless an operation explicitly handles transfer.

The cell creates both operands directly on the selected device, performs the addition there, then moves the result to CPU only for an equality check.
"""
        ),
        code(
            """x = torch.tensor([1.0, 2.0, 3.0], device=device)
y = torch.ones(3, device=device)
z = x + y

print("x device:", x.device)
print("z:", z)

assert z.device == x.device
assert torch.equal(z.cpu(), torch.tensor([2.0, 3.0, 4.0]))"""
        ),
        md(
            """## 6.7.3 Modules Move Through Their Parameters and Buffers

Calling `.to(device)` on a module moves registered parameters and buffers. It does not create a permanent force field that pulls all future input tensors onto the same device.

That separation is important:

```text
model.to(device) moves model state
X.to(device) or tensor creation with device=device moves data
```

The forward pass succeeds only when model state and input batch are colocated. The cell checks the set of parameter devices to prove that the module state moved.
"""
        ),
        code(
            """model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1)).to(device)
X = torch.randn(5, 3, device=device)
Y = model(X)

parameter_devices = {p.device.type for p in model.parameters()}

print("parameter devices:", parameter_devices)
print("output device:", Y.device)

assert parameter_devices == {device.type}
assert Y.device.type == device.type"""
        ),
        md(
            """## 6.7.4 Logging Usually Wants CPU Values

Training code often computes values on GPU but logs values in ordinary Python. That creates two separate issues:

- gradient history: the loss tensor is part of the computation graph
- device placement: the loss tensor may live on GPU

`detach()` says "I want the value, not the graph history." `cpu()` says "move the value to CPU memory." Converting to `float` then gives a plain Python number that is safe for logging.

This distinction prevents accidental graph retention and avoids device-specific logging surprises.
"""
        ),
        code(
            """loss = (Y ** 2).mean()
loss_value = float(loss.detach().cpu())

print("loss tensor device:", loss.device)
print("plain Python loss:", loss_value)

assert isinstance(loss_value, float)"""
        ),
        md(
            """## 6.7.5 Break It Deliberately: Device Mismatch

Device mismatch errors are not mysterious. They usually mean one part of the computation is on CPU while another part is on GPU.

This cell creates that failure only when CUDA exists. On CPU-only machines, it skips the failure because there is no GPU to mismatch against.

The conceptual mistake is splitting one forward computation across devices without an explicit transfer plan.
"""
        ),
        code(
            """if torch.cuda.is_available():
    cpu_x = torch.randn(2, 3)
    gpu_model = nn.Linear(3, 1).to("cuda")
    try:
        gpu_model(cpu_x)
    except RuntimeError as err:
        print(type(err).__name__)
        print(str(err).splitlines()[0])
    else:
        raise AssertionError("A CPU tensor should not run through a CUDA model.")
else:
    print("Skipped mismatch demo because CUDA is not available in this runtime.")"""
        ),
        md(
            """## 6.7.6 Minimal Device-Safe Training Step

This final cell is the device-safe pattern to carry forward:

```text
choose device
move model state to device
create or move batch tensors to device
compute predictions and loss on that device
backpropagate
update parameters
move small logging values back to CPU
```

The training logic is not different because a GPU exists. The difference is placement. The same conceptual training loop from earlier chapters still applies; Chapter 6.7 adds the hardware contract.
"""
        ),
        code(
            """model = nn.Linear(3, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

X = torch.randn(8, 3, device=device)
y = torch.randn(8, 1, device=device)

pred = model(X)
loss = ((pred - y) ** 2).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("loss:", float(loss.detach().cpu()))
print("model device:", next(model.parameters()).device)

assert next(model.parameters()).device.type == device.type
assert torch.isfinite(loss).item()"""
        ),
        checkpoint(
            "6.7",
            [
                "Why is device part of a tensor's runtime contract?",
                "Why is device selection written with a CPU fallback?",
                "What does `model.to(device)` move, and what does it not move?",
                "Why does moving the model not automatically move future input batches?",
                "Why should logged scalar values often use `detach().cpu()`?",
                "What does a device mismatch error mean mechanically?",
            ],
        ),
    ]
    write_nb("Chapter 6.7 - GPUs.ipynb", cells)


def main() -> None:
    build_61()
    build_62()
    build_63()
    build_64()
    build_65()
    build_66()
    build_67()


if __name__ == "__main__":
    main()
