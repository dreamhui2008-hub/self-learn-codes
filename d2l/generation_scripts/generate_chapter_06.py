"""Generate Chapter 6 notebooks for the D2L rewrite project."""

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
    base = "from pathlib import Path\nimport tempfile\nimport torch"
    return code(base + ("\n" + extra.strip() if extra.strip() else ""))



def build_61() -> None:
    cells = [
        title_cell(
            "Chapter 6.1 - Layers and Modules",
            "A module is the basic building block PyTorch uses to organize neural network layers, parameters, and forward computations.",
        ),
        imports_cell(),
    ]

    cells += section(
        "6.1.1 A Custom Module",
        """A custom module is a class you write by inheriting from `torch.nn.Module`.

Inheritance means your class reuses behavior from a parent class. `self` is the current object. `forward` is the method that defines how inputs become outputs.""",
        """Custom modules let you package parameters and computation into reusable model components.""",
        [
            ("md", "Define a tiny custom module."),
            ("code", """class CenteredLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.tensor([1.0, 2.0, 3.0]))"""),
        ],
        """`class CenteredLayer(torch.nn.Module)` creates a child class of `Module`.

`super().__init__()` initializes PyTorch's module machinery.

`forward` receives input `X` and returns centered values.

Calling `layer(...)` automatically runs `forward`.""",
        """Research code often defines custom modules for model blocks that are not available as one built-in layer.""",
        """- `forward` is called indirectly by `layer(X)`.
- `super().__init__()` is required for PyTorch module internals.
- A module can have no trainable parameters.
- Custom modules should keep computation readable.""",
    )

    cells += section(
        "6.1.2 The Sequential Module",
        """A sequential module runs several modules in order.

The output of one module becomes the input to the next module.""",
        """Many networks are simple chains of layers. `Sequential` expresses that chain without writing a custom `forward` method.""",
        [
            ("md", "Build a small sequential network."),
            ("code", """net = torch.nn.Sequential(
    torch.nn.Linear(2, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1),
)
X = torch.randn(4, 2)
net(X).shape"""),
        ],
        """The first linear layer maps 2 features to 3 hidden values.

ReLU applies a nonlinear activation.

The second linear layer maps 3 hidden values to 1 output.

`Sequential` runs these modules in listed order.""",
        """`Sequential` is useful for straightforward feed-forward networks and simple model prototypes.""",
        """- `Sequential` is ordered.
- It is less convenient when the forward pass needs branches or loops.
- Each layer must output a shape the next layer accepts.
- It defines model structure, not training logic.""",
    )

    cells += section(
        "6.1.3 Executing Code in the Forward Propagation Method",
        """The `forward` method can contain ordinary Python control flow, such as `if` statements and loops.

Control flow means code that decides what runs and how often it repeats.""",
        """Some models need computations that are not a simple chain. Custom `forward` methods allow that flexibility.""",
        [
            ("md", "Use a loop inside `forward`."),
            ("code", """class RepeatDouble(torch.nn.Module):
    def forward(self, X):
        for _ in range(2):
            X = X * 2
        return X

module = RepeatDouble()
module(torch.tensor([1.0, 2.0]))"""),
        ],
        """`forward` starts with input `X`.

The loop runs twice.

Each loop iteration doubles `X`.

The returned value is four times the original input.""",
        """PyTorch records the tensor operations that actually run, so dynamic forward logic can still work with automatic differentiation.""",
        """- Python control flow runs during the forward pass.
- Autograd tracks tensor operations, not comments or intentions.
- Complex `forward` methods should be documented with clear shapes.
- Debug dynamic modules with tiny inputs first.""",
    )

    cells += section(
        "6.1.4 Summary",
        """Modules organize neural network computation.

A module may contain parameters, submodules, and a `forward` method.""",
        """PyTorch models are module trees. Understanding modules makes model code easier to inspect and modify.""",
        [
            ("md", "Inspect submodules inside a sequential model."),
            ("code", """names = []
for name, module in net.named_children():
    names.append((name, type(module).__name__))
names"""),
        ],
        """`named_children()` returns immediate submodules.

Each pair contains a name and module object.

`type(module).__name__` shows the module class name.""",
        """Large models are nested module trees; inspection tools help navigate them.""",
        """- A module can contain other modules.
- Not every module has parameters.
- Model structure is separate from optimizer state.
- Module inspection is a core debugging skill.""",
    )

    cells += section(
        "6.1.5 Exercises",
        """These exercises practice creating and inspecting modules.""",
        """Module fluency is required before reading larger research model definitions.""",
        [
            ("md", "Exercise 1: create a module that adds one."),
            ("code", """class AddOne(torch.nn.Module):
    def forward(self, X):
        return X + 1

AddOne()(torch.tensor([2.0]))"""),
            ("md", "Exercise 2: inspect a sequential model's children."),
            ("code", """[(n, type(m).__name__) for n, m in net.named_children()]"""),
        ],
        """Exercise 1 checks custom `forward` behavior.

Exercise 2 checks module-tree inspection.

Both are small versions of real PyTorch model work.""",
        """Custom modules and inspection utilities become essential when models are no longer simple chains.""",
        """- `forward` should return a tensor or structure of tensors.
- `named_children` shows immediate children, not every nested module.
- Calling a module runs `forward`.
- Keep module examples small while learning.""",
    )

    write_nb("Chapter 6.1 - Layers and Modules.ipynb", cells)


def build_62() -> None:
    cells = [
        title_cell(
            "Chapter 6.2 - Parameter Management",
            "Parameter management means finding, inspecting, sharing, and organizing the trainable tensors stored inside modules.",
        ),
        imports_cell(),
    ]

    cells += section(
        "6.2.1 Parameter Access",
        """A parameter is a tensor that a model learns during training.

PyTorch stores parameters inside modules and exposes them with methods such as `parameters()` and `named_parameters()`.""",
        """Before debugging or customizing training, you need to know where parameters live and what shapes they have.""",
        [
            ("md", "Access parameter names and shapes."),
            ("code", """net = torch.nn.Sequential(
    torch.nn.Linear(2, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1),
)
[(name, p.shape) for name, p in net.named_parameters()]"""),
            ("md", "Access one layer's weight and bias directly."),
            ("code", """layer = net[0]
layer.weight.shape, layer.bias.shape"""),
        ],
        """`named_parameters()` returns parameter names and tensors.

The names include the layer position inside `Sequential`.

`net[0]` selects the first submodule.

A `Linear` layer has a weight matrix and bias vector.""",
        """Parameter inspection is used in debugging, freezing layers, custom initialization, and optimizer configuration.""",
        """- Parameters are tensors wrapped for learning.
- ReLU has no parameters.
- A module can expose parameters from nested submodules.
- Shape mismatches often show up clearly in parameter inspection.""",
    )

    cells += section(
        "6.2.2 Tied Parameters",
        """Tied parameters are shared by multiple parts of a model.

Sharing means two computations use the same parameter object, not just equal copied values.""",
        """Parameter tying can enforce structure, reduce parameter count, or make two model parts learn together.""",
        [
            ("md", "Use the same layer object twice."),
            ("code", """shared = torch.nn.Linear(2, 2)
net = torch.nn.Sequential(shared, torch.nn.ReLU(), shared)
first_id = id(net[0].weight)
second_id = id(net[2].weight)
first_id == second_id"""),
            ("md", "Equal object identity means actual sharing."),
            ("code", """net[0].weight is net[2].weight"""),
        ],
        """`shared` is one `Linear` layer object.

The same object is inserted twice into `Sequential`.

`id(...)` returns an identity number for the object.

The two positions refer to the same weight parameter.""",
        """Tied parameters appear in some language models, autoencoders, and architectures where symmetry or reuse is intentional.""",
        """- Equal values are not the same as shared objects.
- Shared parameters receive gradient contributions from every use.
- Accidental sharing can create confusing behavior.
- Use `is` or `id` to check object identity.""",
    )

    cells += section(
        "6.2.3 Summary",
        """Parameter management is about knowing which tensors are learned, where they are stored, and whether they are shared.""",
        """This knowledge is required for initialization, freezing, saving, loading, and custom optimization.""",
        [
            ("md", "Count trainable parameter tensors."),
            ("code", """count = 0
for p in net.parameters():
    count += 1
count"""),
        ],
        """The loop visits every parameter tensor exposed by the model.

The count measures tensors, not individual scalar values.

A weight matrix counts as one parameter tensor.""",
        """Real models can have thousands of parameter tensors and millions or billions of scalar parameter values.""",
        """- Tensor count and scalar count are different.
- Shared parameters should not be double-counted conceptually.
- Parameter names help locate tensors.
- Optimizers update parameters they are given.""",
    )

    cells += section(
        "6.2.4 Exercises",
        """These exercises practice finding and comparing parameters.""",
        """Parameter inspection is a practical debugging habit.""",
        [
            ("md", "Exercise 1: list parameter names."),
            ("code", """[name for name, _ in net.named_parameters()]"""),
            ("md", "Exercise 2: count scalar parameter values."),
            ("code", """total = 0
for p in net.parameters():
    total += p.numel()
total"""),
        ],
        """Exercise 1 checks naming.

Exercise 2 uses `numel`, which counts scalar elements inside a tensor.

The scalar count is usually what people mean by model size.""",
        """Counting parameters helps compare model capacity and memory use.""",
        """- `numel()` counts scalar entries.
- Names depend on module structure.
- Parameter access does not update parameters.
- Inspect before customizing optimizer behavior.""",
    )

    write_nb("Chapter 6.2 - Parameter Management.ipynb", cells)


def build_63() -> None:
    cells = [
        title_cell(
            "Chapter 6.3 - Parameter Initialization",
            "Parameter initialization chooses starting values for learned tensors before training begins.",
        ),
        imports_cell(),
    ]

    cells += section(
        "6.3.1 Built-in Initialization",
        """Built-in initialization uses PyTorch helper functions to fill parameter tensors.

An in-place function changes an existing tensor directly. In PyTorch, many in-place functions end with an underscore.""",
        """Initialization scale affects training stability. Built-in functions provide standard, tested choices.""",
        [
            ("md", "Initialize a layer with normal weights and zero bias."),
            ("code", """layer = torch.nn.Linear(3, 2)
torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
torch.nn.init.zeros_(layer.bias)
layer.weight.shape, layer.bias"""),
            ("md", "Initialize with Xavier uniform."),
            ("code", """layer = torch.nn.Linear(4, 3)
torch.nn.init.xavier_uniform_(layer.weight)
torch.nn.init.zeros_(layer.bias)
layer.weight.shape"""),
        ],
        """`normal_` fills the weight tensor with random normal values.

`zeros_` fills the bias tensor with zeros.

`xavier_uniform_` chooses a uniform range based on layer input and output sizes.

The underscore signals direct modification of existing tensors.""",
        """Custom initialization is common when reproducing papers, debugging unstable training, or matching architecture conventions.""",
        """- Initialization happens before training.
- In-place functions should be used deliberately.
- Bias and weight initialization can differ.
- Good initialization helps but does not guarantee good training.""",
    )

    cells += section(
        "6.3.2 Summary",
        """PyTorch provides initialization functions under `torch.nn.init`.

These functions modify parameters before optimization starts.""",
        """Initialization is one of the earliest choices that can affect optimization behavior.""",
        [
            ("md", "A reusable initialization helper."),
            ("code", """def init_small(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, std=0.01)
        torch.nn.init.zeros_(module.bias)

net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
net.apply(init_small)"""),
        ],
        """`isinstance` checks whether a module is a `Linear` layer.

`net.apply(init_small)` visits modules inside the network.

The helper initializes every linear layer it sees.

This is a common pattern for model-wide initialization.""",
        """Model-wide initialization helpers are useful when a network has many layers.""",
        """- `apply` visits modules, not raw tensors only.
- Check module type before assuming it has weights.
- Initialization helpers should be deterministic if reproducibility matters.
- Do not initialize after training unless you intend to reset learning.""",
    )

    cells += section(
        "6.3.3 Exercises",
        """These exercises practice initializing parameters and inspecting the result.""",
        """Initialization is easier to trust when you can verify tensor shapes and values.""",
        [
            ("md", "Exercise 1: set all weights to one."),
            ("code", """layer = torch.nn.Linear(2, 2)
torch.nn.init.ones_(layer.weight)
layer.weight"""),
            ("md", "Exercise 2: initialize only biases to zero."),
            ("code", """torch.nn.init.zeros_(layer.bias)
layer.bias"""),
        ],
        """Exercise 1 uses an in-place initializer on weights.

Exercise 2 uses an in-place initializer on bias.

Both examples directly modify stored parameters.""",
        """These operations are building blocks for custom model setup.""",
        """- In-place initialization changes the module immediately.
- Weight shapes come from layer dimensions.
- Bias shape equals output size for a linear layer.
- Simple initializers are useful for debugging.""",
    )

    write_nb("Chapter 6.3 - Parameter Initialization.ipynb", cells)


def build_64() -> None:
    cells = [
        title_cell(
            "Chapter 6.4 - Lazy Initialization",
            "Lazy initialization delays parameter shape creation until the layer sees input data. This is useful when input dimensions are not known while writing the model.",
        ),
        imports_cell(),
    ]

    cells += section(
        "6.4.1 Summary",
        """Lazy initialization means a layer waits to infer parameter shapes from the first input.

A lazy layer has incomplete parameter shapes before its first forward pass.""",
        """Lazy layers reduce the need to manually compute input sizes, especially after complex feature extractors.""",
        [
            ("md", "Use a lazy linear layer."),
            ("code", """lazy = torch.nn.LazyLinear(3)
X = torch.randn(2, 4)
y = lazy(X)
weight_shape = lazy.weight.shape
y.shape, weight_shape"""),
            ("md", "A non-lazy layer needs the input size upfront."),
            ("code", """regular = torch.nn.Linear(4, 3)
regular(torch.randn(2, 4)).shape"""),
        ],
        """`LazyLinear(3)` knows the output size but not the input size.

The first forward pass receives `X` with 4 features.

After that pass, the weight shape becomes known.

A regular `Linear(4, 3)` requires the input size immediately.""",
        """Lazy initialization appears in model-building utilities where input shapes are discovered by running sample data through the network.""",
        """- Lazy layers need a first forward pass before full parameter inspection.
- Lazy initialization is convenience, not a different model type.
- Shape errors may appear later with lazy layers.
- Once initialized, lazy parameters behave like ordinary parameters.""",
    )

    cells += section(
        "6.4.2 Exercises",
        """These exercises practice lazy layer behavior.""",
        """The key habit is knowing when shapes become available.""",
        [
            ("md", "Exercise 1: initialize a lazy layer with a sample input."),
            ("code", """layer = torch.nn.LazyLinear(2)
out = layer(torch.randn(5, 7))
out.shape, layer.weight.shape"""),
            ("md", "Exercise 2: compare with regular Linear."),
            ("code", """regular = torch.nn.Linear(7, 2)
regular.weight.shape"""),
        ],
        """Exercise 1 shows the lazy layer learning its input feature count from data.

Exercise 2 shows the regular layer already has known shape.

Both produce weights compatible with 7 input features and 2 outputs.""",
        """Lazy modules are helpful for quick prototyping, but explicit dimensions are often clearer in teaching code.""",
        """- Lazy does not mean untrained.
- Lazy means shape creation is delayed.
- A sample forward pass initializes parameters.
- Inspect lazy layers after they see input.""",
    )

    write_nb("Chapter 6.4 - Lazy Initialization.ipynb", cells)


def build_65() -> None:
    cells = [
        title_cell(
            "Chapter 6.5 - Custom Layers",
            "Custom layers let you define reusable computations that fit into PyTorch models like built-in layers.",
        ),
        imports_cell(),
    ]

    cells += section(
        "6.5.1 Layers without Parameters",
        """A layer without parameters performs computation but has no learned weights or biases.

It still inherits from `torch.nn.Module` so it can be placed inside larger models.""",
        """Some useful transformations do not need learned parameters, such as centering, reshaping, or clipping.""",
        [
            ("md", "Define a parameter-free layer."),
            ("code", """class CenteredLayer(torch.nn.Module):
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
Y = layer(torch.tensor([1.0, 2.0, 3.0]))
Y.mean()"""),
        ],
        """The class inherits from `Module`.

It defines only `forward` because no parameters need initialization.

The output subtracts the mean from every input value.

The output mean is zero apart from floating-point rounding.""",
        """Parameter-free layers are common for deterministic transformations inside a model pipeline.""",
        """- A module does not need parameters.
- `forward` can use ordinary tensor operations.
- Parameter-free does not mean useless.
- The layer can still be part of `Sequential`.""",
    )

    cells += section(
        "6.5.2 Layers with Parameters",
        """A layer with parameters stores learned tensors as `torch.nn.Parameter` objects.

`Parameter` tells PyTorch that a tensor should be treated as trainable model state.""",
        """Custom parameterized layers are needed when built-in layers do not match the computation you want.""",
        [
            ("md", "Define a custom linear layer."),
            ("code", """class MyLinear(torch.nn.Module):
    def __init__(self, in_units, out_units):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_units, out_units) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(out_units))
    def forward(self, X):
        return X @ self.weight + self.bias

layer = MyLinear(2, 3)
layer(torch.randn(4, 2)).shape"""),
        ],
        """`Parameter` wraps tensors so PyTorch registers them as learnable.

`self.weight` and `self.bias` are stored as module attributes.

The forward method computes a linear transformation.

Calling the layer runs that forward computation.""",
        """Custom parameterized layers are central to research code and new architectures.""",
        """- Use `Parameter` for tensors that should be learned.
- Store parameters on `self` so the module can find them.
- Shape choices must match the forward computation.
- Custom layers should be tested with tiny inputs.""",
    )

    cells += section(
        "6.5.3 Summary",
        """Custom layers can be parameter-free or parameterized.

Both forms use `forward` to define computation.""",
        """Custom layers let you extend PyTorch while keeping compatibility with modules, optimizers, and model inspection tools.""",
        [
            ("md", "Inspect custom layer parameters."),
            ("code", """layer = MyLinear(2, 3)
[(name, p.shape) for name, p in layer.named_parameters()]"""),
        ],
        """`named_parameters()` finds the custom layer's registered parameters.

The names come from the attributes `weight` and `bias`.

The shapes match the custom initialization.""",
        """If a custom layer registers parameters correctly, optimizers can update them like built-in layer parameters.""",
        """- Registration happens by assigning `Parameter` objects to `self`.
- Plain tensors on `self` are not automatically trainable parameters.
- Custom layers should expose predictable shapes.
- Test forward output before training.""",
    )

    cells += section(
        "6.5.4 Exercises",
        """These exercises practice custom layer creation.""",
        """Writing tiny layers makes module mechanics concrete.""",
        [
            ("md", "Exercise 1: create a layer that multiplies by two."),
            ("code", """class DoubleLayer(torch.nn.Module):
    def forward(self, X):
        return 2 * X

DoubleLayer()(torch.tensor([3.0]))"""),
            ("md", "Exercise 2: list parameters of a parameter-free layer."),
            ("code", """list(DoubleLayer().parameters())"""),
        ],
        """Exercise 1 defines a parameter-free computation.

Exercise 2 confirms there are no trainable parameters.

This distinction matters for optimizer behavior.""",
        """Custom layers are safe to add to larger models when their input/output behavior is clear.""",
        """- Parameter-free layers return an empty parameter list.
- Parameterized layers need `Parameter` objects.
- The forward method should be deterministic unless randomness is intended.
- Test with one small tensor first.""",
    )

    write_nb("Chapter 6.5 - Custom Layers.ipynb", cells)


def build_66() -> None:
    cells = [
        title_cell(
            "Chapter 6.6 - File I/O",
            "File I/O means saving data to disk and loading it later. PyTorch can save tensors and model parameters for reuse.",
        ),
        imports_cell(),
    ]

    cells += section(
        "6.6.1 Loading and Saving Tensors",
        """Saving a tensor writes its values to a file. Loading reads those values back into memory.

A file path tells the program where the saved object lives.""",
        """Training can take time, and intermediate data may need to be reused. Saving avoids recomputing or losing results.""",
        [
            ("md", "Save and load one tensor in a temporary directory."),
            ("code", """with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "x.pt"
    x = torch.tensor([1.0, 2.0])
    torch.save(x, path)
    loaded = torch.load(path)
loaded"""),
            ("md", "Save and load a dictionary of tensors."),
            ("code", """with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "data.pt"
    torch.save({"a": torch.ones(2)}, path)
    loaded = torch.load(path)
loaded["a"]"""),
        ],
        """`TemporaryDirectory` creates a short-lived folder.

`torch.save(x, path)` writes the tensor.

`torch.load(path)` reads it back.

A dictionary can store multiple named tensors in one file.""",
        """Tensor saving is useful for cached data, embeddings, checkpoints, and debugging artifacts.""",
        """- Save to known paths in real projects.
- Temporary directories disappear after the block.
- `torch.load` reads Python objects saved by PyTorch.
- Be careful loading files from untrusted sources.""",
    )

    cells += section(
        "6.6.2 Loading and Saving Model Parameters",
        """A model's `state_dict` is a dictionary containing parameter and buffer tensors.

A buffer is persistent tensor state that is not usually trained by gradients.""",
        """Saving only parameters lets you recreate the model architecture in code and restore learned values from disk.""",
        [
            ("md", "Save and load a model state dictionary."),
            ("code", """net = torch.nn.Linear(2, 1)
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "model.pt"
    torch.save(net.state_dict(), path)
    new_net = torch.nn.Linear(2, 1)
    new_net.load_state_dict(torch.load(path))
new_net.weight.shape"""),
        ],
        """`state_dict()` returns the model's saved tensor state.

A new model with the same architecture is created.

`load_state_dict` copies saved values into the new model.

The architecture must match the saved parameter shapes.""",
        """Model checkpoints usually store `state_dict`s, optimizer state, epoch number, and metadata.""",
        """- Saving parameters is not the same as saving the class definition.
- The loading model must have compatible architecture.
- Optimizer state is separate from model state.
- Checkpoints should include enough metadata to resume work.""",
    )

    cells += section(
        "6.6.3 Summary",
        """PyTorch file I/O commonly uses `torch.save`, `torch.load`, `state_dict`, and `load_state_dict`.""",
        """Saving and loading make experiments reproducible and training resumable.""",
        [
            ("md", "A checkpoint dictionary structure."),
            ("code", """checkpoint = {
    "model": "state_dict goes here",
    "epoch": 3,
    "note": "tiny example",
}
checkpoint"""),
        ],
        """The checkpoint dictionary names what would be saved.

Real code would store actual tensor states.

Metadata such as epoch helps resume training.""",
        """Well-designed checkpoint files are part of reliable ML engineering.""",
        """- Save enough information to reproduce results.
- Keep architecture code versioned.
- Do not trust arbitrary serialized files.
- Verify loaded models with a tiny prediction.""",
    )

    cells += section(
        "6.6.4 Exercises",
        """These exercises practice local saving and loading.""",
        """File I/O should be tested with tiny artifacts before it is trusted for large checkpoints.""",
        [
            ("md", "Exercise 1: save and load a scalar tensor."),
            ("code", """with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "scalar.pt"
    torch.save(torch.tensor(7.0), path)
    value = torch.load(path)
value"""),
            ("md", "Exercise 2: inspect state dictionary keys."),
            ("code", """net = torch.nn.Linear(3, 2)
list(net.state_dict().keys())"""),
        ],
        """Exercise 1 checks tensor serialization.

Exercise 2 checks model state naming.

Linear layers usually expose `weight` and `bias` in their state dictionary.""",
        """These checks prepare for saving larger models and checkpoints.""",
        """- Temporary files are only for examples.
- State dictionary keys depend on module names.
- Loading requires compatible shapes.
- Keep save paths organized in real projects.""",
    )

    write_nb("Chapter 6.6 - File I-O.ipynb", cells)


def build_67() -> None:
    cells = [
        title_cell(
            "Chapter 6.7 - GPUs",
            "GPUs accelerate tensor computation when available. This notebook explains devices with code that remains inspectable and safe on machines without a GPU.",
        ),
        imports_cell(),
    ]

    cells += section(
        "6.7.1 Computing Devices",
        """A computing device is hardware where tensors and operations live. Common devices are CPU and GPU.

A CPU is general-purpose. A GPU is designed for many parallel numerical operations.""",
        """Deep learning can be much faster on GPUs because tensor operations often contain many independent arithmetic tasks.""",
        [
            ("md", "Check whether CUDA GPU support is available."),
            ("code", """has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")
has_cuda, device"""),
            ("md", "Count visible CUDA devices safely."),
            ("code", """count = torch.cuda.device_count()
count"""),
        ],
        """`torch.cuda.is_available()` returns whether PyTorch can use CUDA.

`torch.device(...)` creates a device object.

The conditional chooses GPU only when available.

`device_count()` reports how many CUDA devices PyTorch sees.""",
        """Training scripts often choose a device at startup and move tensors and models there.""",
        """- CUDA is NVIDIA's GPU computing platform.
- A machine can have PyTorch installed without GPU support.
- Device checks should be conditional, not assumed.
- CPU examples should still run conceptually without GPU hardware.""",
    )

    cells += section(
        "6.7.2 Tensors and GPUs",
        """A tensor lives on one device at a time. Operations usually require tensors to be on the same device.

Moving a tensor means copying its data to another device.""",
        """Device mismatch is a common PyTorch error. A model on GPU cannot directly compute with input tensors left on CPU.""",
        [
            ("md", "Create a tensor on the selected device."),
            ("code", """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0], device=device)
x.device"""),
            ("md", "Move a tensor back to CPU."),
            ("code", """x_cpu = x.to("cpu")
x_cpu.device"""),
        ],
        """The device is chosen safely based on availability.

`torch.tensor(..., device=device)` creates the tensor directly on that device.

`.to("cpu")` moves or keeps the tensor on CPU.

The `.device` attribute reports where the tensor lives.""",
        """Data batches must be moved to the same device as the model before the forward pass.""",
        """- Device movement can copy data and cost time.
- Tensor device and tensor dtype are different concepts.
- Operations across different devices usually fail.
- Keep device handling explicit in training code.""",
    )

    cells += section(
        "6.7.3 Neural Networks and GPUs",
        """A neural network module also lives on a device through its parameters.

Moving a module to a device moves its parameters and buffers.""",
        """Model parameters and input tensors must be on the same device for computation.""",
        [
            ("md", "Move a small network and input to the selected device."),
            ("code", """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.nn.Linear(2, 1).to(device)
X = torch.randn(3, 2).to(device)
y_hat = net(X)
y_hat.device"""),
        ],
        """The selected device is GPU if available, otherwise CPU.

`.to(device)` moves the module parameters.

The input tensor is moved to the same device.

The output is produced on that device.""",
        """Training loops usually move each minibatch to the model's device before computing predictions.""",
        """- Moving the model is not enough; move input data too.
- Outputs are usually on the same device as the computation.
- Convert to CPU before NumPy conversion.
- Device-safe code should not assume GPU availability.""",
    )

    cells += section(
        "6.7.4 Summary",
        """Device management controls where tensors and models are stored and computed.

Correct device placement is required for GPU acceleration and for avoiding mismatch errors.""",
        """As models grow, hardware placement becomes part of the training system design.""",
        [
            ("md", "A device-safe training setup checklist."),
            ("code", """checklist = [
    "choose device",
    "move model",
    "move each batch",
    "move outputs to CPU for logging if needed",
]
checklist"""),
        ],
        """The checklist names common device-handling steps.

Choosing a device happens once near startup.

Batches are moved during training.

Logging sometimes needs CPU values.""",
        """Good device handling lets the same code work on CPU laptops and GPU servers.""",
        """- Hardware availability varies by machine.
- Device checks should be explicit.
- CPU fallback is useful for teaching and debugging.
- GPU speedups require enough work to offset transfer costs.""",
    )

    cells += section(
        "6.7.5 Exercises",
        """These exercises practice device-safe code without requiring a GPU.""",
        """The habit is to write code that adapts to available hardware.""",
        [
            ("md", "Exercise 1: choose a safe device."),
            ("code", """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device"""),
            ("md", "Exercise 2: create a tensor on that device."),
            ("code", """x = torch.ones(2, device=device)
x.device"""),
        ],
        """Exercise 1 chooses GPU only when available.

Exercise 2 creates a tensor directly on the chosen device.

Both examples run conceptually on CPU-only machines.""",
        """Device-safe code is the starting point for portable training scripts.""",
        """- Never assume every machine has a GPU.
- Always check device placement when errors mention device mismatch.
- Move data and model together.
- Keep examples hardware-safe when teaching.""",
    )

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
