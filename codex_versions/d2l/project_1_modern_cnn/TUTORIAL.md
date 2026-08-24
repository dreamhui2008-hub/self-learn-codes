# Project 1: Modern CNN Training System

This project turns the Chapter 8 CNN material into a production-shaped image classification system.

The goal is not to worship a specific architecture name. The goal is to understand why each design decision exists, implement it cleanly, test the tensor contracts, train real models, save artifacts, and explain the results using evidence.

The main model is a CIFAR-10 sized ConvNeXt-style CNN.

The comparison models are:

- MLP baseline
- LeNet-CIFAR
- AlexNet-small
- ResNet-small
- ConvNeXt-tiny

## 0. What This Project Is

This is Project 1 after the Chapter 8 modern CNN study.

It is deliberately more serious than a notebook-only exercise:

```text
notebook-only learning:
    fast for exploration
    easy to accidentally duplicate state
    hard to test as a system
    hard to reproduce on another machine

project-shaped learning:
    slower at first
    code has clear ownership
    tests catch broken contracts
    CLI commands run locally or in the cloud
    notebook becomes explanation and inspection
```

The project still stays small enough to read completely.

This tutorial is written as:

```text
theory
-> small syntax/mechanics drills
-> packaged file edits
-> checks
-> notebook reflection
-> experiment report
```

## 1. Learning Contract

You are done when you can:

- explain why convolution is a useful image prior
- explain why convolution windows slide instead of using one full-image matrix
- explain why pooling and downsampling exist
- explain why channels often increase while spatial dimensions shrink
- explain why ReLU replaced sigmoid for deeper CNNs
- explain why BatchNorm changes the train/eval contract
- explain why residual paths help optimization
- explain why depthwise convolution and pointwise mixing appear in ConvNeXt-style blocks
- trace shapes through every model
- count parameters and explain where they live
- train one batch and prove parameters changed
- run full training from a CLI command
- save metrics, configs, checkpoints, plots, and error examples
- compare architectures and ablations without changing training code

## 2. Production-Adjacent Scope

This project includes:

- package source under `src/modern_cnn`
- typed TOML configs
- deterministic seeds where practical
- train/val/test data pipeline
- model registry
- unit tests
- CLI commands
- run directories
- JSON and JSONL artifacts
- checkpoints
- confusion matrix plots
- misclassification grids
- notebook drills and artifact inspection

This project skips:

- Docker
- CI/CD
- distributed training
- model serving APIs
- model registry services
- external experiment trackers

Those are skipped only for this project. The design should make them easy to add later.

## 3. Core System

The core system is:

```text
config -> data -> model -> train -> evaluate -> checkpoint -> artifacts -> CLI
```

Meaning:

- `config.py` defines what run is being executed
- `data.py` builds datasets and dataloaders
- `models.py` builds architectures from a registry
- `train.py` performs optimization
- `evaluate.py` measures behavior without training
- `checkpoint.py` saves and loads model state
- `artifacts.py` writes evidence to disk
- `cli.py` makes the project runnable outside notebooks

The notebook imports these modules.

The notebook does not redefine the system.

## 4. Exact Run Order

Use this order:

```text
1. Create the project folders.
2. Type pyproject.toml and package init.
3. Type the config files.
4. Type config.py.
5. Run the config loader drill.
6. Type data.py.
7. Run the data shape drills.
8. Type metrics.py.
9. Run the metric drills.
10. Type blocks.py.
11. Run block shape drills.
12. Type models.py.
13. Run model shape and parameter drills.
14. Type evaluate.py, checkpoint.py, and artifacts.py.
15. Run checkpoint and artifact drills.
16. Type train.py.
17. Run the one-batch update drill.
18. Type cli.py.
19. Type tests.
20. Run pytest.
21. Run smoke CLI training.
22. Build the notebook.
23. Run real CIFAR-10 experiments.
24. Run ablations.
25. Write the final report.
```

Do not run real CIFAR-10 training until smoke tests pass.

## 5. Reference Map

Use the Chapter 8 architectures this way:

| Model | Core idea | What this project uses it for |
|---|---|---|
| MLP | no image-specific prior | baseline that flattens pixels |
| LeNet | early CNN pipeline | convolution, sigmoid, average pooling, dense head |
| AlexNet-small | ReLU, max pooling, deeper conv stack | older high-capacity CNN baseline scaled for CIFAR-10 |
| ResNet-small | residual blocks | optimization comparison with identity paths |
| ConvNeXt-tiny | modernized CNN block | depthwise conv, pointwise mixing, GELU, LayerNorm, global pooling |

The comparison question:

```text
Which architectural biases improve CIFAR-10 performance under the same training pipeline?
```

The engineering question:

```text
Can the same production-shaped code train, evaluate, and inspect all of them?
```

## 6. Theory: Why Convolution Exists

An image is not just a list of numbers. It is a grid.

A CIFAR-10 image arrives as:

```text
[channels, height, width] = [3, 32, 32]
```

The height and width dimensions encode adjacency. Neighboring pixels often belong to the same local object part, edge, texture, or color region.

An MLP baseline flattens the image:

```text
[3, 32, 32] -> [3072]
```

That can still work, but the model no longer receives locality as an architectural bias. It must learn from dense weights that nearby pixels often interact.

A convolution keeps spatial structure:

```text
input image grid
-> local window
-> shared detector weights
-> feature map
```

The important biases are:

- locality: a small kernel sees nearby pixels together
- weight sharing: the same detector is reused at every spatial position
- translation tolerance: the same feature can be detected in different positions
- parameter efficiency: a 3x3 kernel has far fewer weights than a full-image dense mapping

Mechanical contrast:

```text
dense layer:
    each output unit has separate weights for every input pixel

convolution:
    each output channel reuses one small kernel across the grid
```

Tradeoff:

- convolution assumes local spatial structure matters
- that is good for natural images
- it is not always right for arbitrary tabular features

This is why the project keeps an MLP baseline. The baseline is not there because it is expected to win. It is there to reveal what the convolutional prior buys.

## 7. Theory: Why Convolution Windows Are Iterated

A convolution kernel is small, such as 3x3 or 5x5.

Question:

```text
If the image is 32x32, why not use one huge 32x32 detector?
```

Answer:

```text
because small reusable detectors are cheaper and more general
```

A 3x3 kernel asks local questions:

- is there a vertical edge here?
- is there a color transition here?
- is there a corner here?
- is there a small texture here?

The same learned question is asked at every location.

After many layers, the model can represent larger patterns because the receptive field grows.

Example:

```text
layer 1:
    one output depends on a 3x3 input neighborhood

layer 2:
    one output depends on several layer-1 neighborhoods

layer 3:
    one output depends on an even wider input region
```

So the model gets local parameter efficiency early and larger context later.

## 8. Theory: Why Pooling And Downsampling Exist

Pooling summarizes spatial neighborhoods.

Max pooling asks:

```text
Was a strong activation present in this region?
```

Average pooling asks:

```text
What was the average activation in this region?
```

Why summarize?

- reduce compute
- reduce memory
- enlarge effective receptive field
- tolerate small shifts
- force later layers to operate on more abstract spatial maps

Mechanical example:

```text
[batch, 64, 32, 32]
MaxPool2d(kernel=2, stride=2)
-> [batch, 64, 16, 16]
```

The channel count is unchanged. The spatial dimensions are halved.

Tradeoff:

- downsampling saves compute
- downsampling loses exact location detail
- max pooling keeps strongest evidence but discards weaker signals
- average pooling smooths signals but may blur sharp evidence

Modern CNNs often use strided convolutions instead of explicit pooling in some stages. The purpose is still controlled spatial reduction.

## 9. Theory: Why Channels Grow While Space Shrinks

CNNs often follow this pattern:

```text
[3, 32, 32]
-> [64, 32, 32]
-> [128, 16, 16]
-> [256, 8, 8]
-> [512, 4, 4]
```

Spatial detail decreases.

Feature variety increases.

Early layers preserve many positions because low-level visual evidence is local. Later layers need more channels because they represent more possible patterns:

- textures
- parts
- object fragments
- class-relevant combinations

The network trades:

```text
where exactly is the signal?
```

for:

```text
what kinds of signals exist?
```

This is why the classifier head often uses global average pooling at the end. By then, exact final position is often less important than whether each learned feature appeared.

## 10. Theory: Why ReLU Replaced Sigmoid

LeNet used sigmoid-style activations.

Sigmoid maps:

```text
large negative -> near 0
0 -> 0.5
large positive -> near 1
```

This is easy to understand but has a problem: saturation.

When sigmoid is near 0 or 1, its gradient is small. In deeper networks, small gradients make optimization difficult.

ReLU maps:

```text
negative -> 0
positive -> unchanged
```

Why ReLU helped deep CNNs:

- cheaper operation
- stronger gradients for positive activations
- sparse activations
- less saturation than sigmoid on the positive side

Tradeoff:

- negative information is zeroed
- units can become inactive if they stay negative
- ReLU output is not normalized

Modern ConvNeXt-style blocks often use GELU instead of ReLU:

```text
GELU is smoother and gates values more gradually
```

The tutorial uses LeNet with sigmoid to preserve the historical idea, AlexNet/ResNet with ReLU, and ConvNeXt with GELU.

## 11. Theory: Why Dense Heads Become Expensive

Flattening turns a feature map into a vector.

Example:

```text
[batch, 256, 4, 4]
-> [batch, 4096]
```

A dense layer from 4096 to 512 has:

```text
4096 * 512 + 512 = 2,097,664 parameters
```

That is only one layer.

LeNet and AlexNet used dense classifier heads. This made sense historically, but it can create many parameters.

Modern CNNs often use:

```text
AdaptiveAvgPool2d((1, 1))
flatten
Linear(channels, classes)
```

Mechanical meaning:

```text
[batch, 256, 4, 4]
-> [batch, 256, 1, 1]
-> [batch, 256]
-> [batch, 10]
```

Tradeoff:

- dense heads preserve final spatial layout information
- global average pooling discards final location layout
- global average pooling dramatically reduces classifier parameters
- for classification, "feature exists" is often enough

## 12. Theory: Why BatchNorm Exists

BatchNorm2d normalizes each channel using statistics from:

```text
batch, height, width
```

For each channel, it computes a mean and variance during training.

Why this helps:

- activation scales become less fragile
- deeper networks often train more reliably
- larger learning rates often become possible
- batch noise can regularize slightly

Mechanical contract:

```text
model.train():
    BatchNorm uses current batch statistics
    BatchNorm updates running_mean and running_var

model.eval():
    BatchNorm uses stored running_mean and running_var
    BatchNorm does not update those statistics
```

This is why training and evaluation functions must explicitly switch modes.

Failure pattern:

```text
validation accuracy changes unpredictably
```

Possible cause:

```text
evaluating while the model is still in train mode
```

## 13. Theory: Why Residual Connections Exist

A residual block computes:

```text
output = block(x) + x
```

If shape changes:

```text
output = block(x) + projection(x)
```

The identity path makes it easier for a deep block to learn a correction rather than an entirely new representation.

Optimization meaning:

```text
without residual:
    every layer must carry useful signal through the main path

with residual:
    signal and gradients have a simpler path through addition
```

Shape rule:

```text
main path output shape must equal skip path output shape
```

If channels or spatial size change, raw `x` cannot be added. A projection, usually a 1x1 convolution, fixes the shape.

This project tests residual shape changes because many ResNet bugs are not conceptual. They are plain tensor-shape mismatches.

## 14. Theory: Why Depthwise Convolution Exists

A regular convolution mixes spatial and channel information at the same time.

For:

```text
in_channels = 64
out_channels = 64
kernel = 7x7
```

regular convolution weights:

```text
64 * 64 * 7 * 7 = 200,704
```

Depthwise convolution uses one spatial filter per channel:

```text
groups = channels
```

depthwise weights:

```text
64 * 1 * 7 * 7 = 3,136
```

Then a pointwise 1x1 convolution mixes channels.

Mechanical split:

```text
depthwise conv:
    spatial mixing within each channel

pointwise conv:
    channel mixing at each spatial location
```

ConvNeXt uses this idea as part of a modern CNN block:

```text
depthwise spatial conv
-> normalization
-> pointwise channel expansion
-> GELU
-> pointwise channel contraction
-> residual addition
```

The project includes ablations so this does not remain a fact to memorize.

## 15. Project Layout

Create:

```text
codex_versions/d2l/project_1_modern_cnn/
```

Target files:

```text
project_1_modern_cnn/
  README.md
  TUTORIAL.md
  pyproject.toml
  .gitignore
  configs/
    smoke.toml
    cifar10_mlp.toml
    cifar10_lenet.toml
    cifar10_alexnet_small.toml
    cifar10_resnet_small.toml
    cifar10_convnext_tiny.toml
    ablation_convnext_no_residual.toml
    ablation_convnext_small_kernel.toml
    ablation_convnext_no_augmentation.toml
  notebooks/
    project_1_modern_cnn.ipynb
  src/
    modern_cnn/
      __init__.py
      artifacts.py
      blocks.py
      checkpoint.py
      cli.py
      config.py
      data.py
      evaluate.py
      metrics.py
      models.py
      reference_notes.py
      train.py
  tests/
    test_blocks.py
    test_checkpoint.py
    test_config.py
    test_data.py
    test_shapes.py
    test_train_step.py
```

## 16. Phase 0: Skeleton And Metadata

### 16.1 Goal

Build the package shell so Python can import `modern_cnn`.

This is not busywork. Importability is what lets the same code work in notebooks, tests, and CLI commands.

### 16.2 Exact Run Order

Use this order:

```text
1. Create folders.
2. Type pyproject.toml.
3. Type .gitignore.
4. Type src/modern_cnn/__init__.py.
5. Install editable package.
6. Run import check.
```

### 16.3 Directory Drill

Terminal: project root shell, not `experiments.ipynb`.

Action:

Run from `codex_versions/d2l/project_1_modern_cnn`.

```bash
mkdir -p configs notebooks src/modern_cnn tests artifacts data
```

Mechanical meaning:

- `src/modern_cnn` is importable package code.
- `configs` contains experiment definitions.
- `artifacts` contains generated run outputs.
- `data` contains downloaded datasets.
- `tests` contains system contract checks.

### 16.4 File Edit: `pyproject.toml`

Action:

Create `pyproject.toml`.

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "modern-cnn"
version = "0.1.0"
description = "Project 1 modern CNN training system"
requires-python = ">=3.10"
dependencies = [
  "torch",
  "torchvision",
  "matplotlib",
  "tomli; python_version < '3.11'",
]

[project.optional-dependencies]
dev = [
  "pytest",
  "ipykernel",
]

[project.scripts]
modern-cnn = "modern_cnn.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Mechanical meaning:

- `dependencies` are needed at runtime.
- `dev` dependencies are needed for tests and notebooks.
- `[project.scripts]` creates the `modern-cnn` command after install.
- `pythonpath = ["src"]` lets tests import local source.

### 16.5 File Edit: `.gitignore`

Action:

Create `.gitignore`.

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ipynb_checkpoints/
.venv/
data/
artifacts/
*.pt
*.pth
```

Mechanical meaning:

- source code is tracked
- generated data is not tracked
- model checkpoints are not tracked
- temporary Python cache files are not tracked

### 16.6 File Edit: `src/modern_cnn/__init__.py`

Action:

Create `src/modern_cnn/__init__.py`.

```python
__version__ = "0.1.0"
```

### 16.7 Install And Import Check

Terminal: project root shell, not `experiments.ipynb`.

Action:

Run:

```bash
python -m pip install -e ".[dev]"
python -c "import modern_cnn; print(modern_cnn.__version__)"
```

Expected output:

```text
0.1.0
```

Failure checks:

- `ModuleNotFoundError: modern_cnn`: editable install did not run, or current directory is wrong.
- `modern-cnn: command not found`: reinstall after adding `[project.scripts]`.

## 17. Phase 1: Configs

### 17.1 Goal

Configs make experiments reproducible.

The training code should not hard-code:

- model name
- batch size
- learning rate
- epoch count
- dataset mode
- output directory

Those belong in config files.

### 17.2 Exact Run Order

Use this order:

```text
1. Run the nested dictionary preflight.
2. Run the dataclass preflight.
3. Type all TOML config files.
4. Type config.py in two chunks.
5. Run the config loader drill.
6. Run failure checks.
```

### 17.3 Syntax Preflight: Nested Dictionaries

Notebook: `experiments.ipynb`.

Purpose:

TOML files load into nested dictionaries. Before using TOML, understand the shape.

Action:

Run this in Python or the notebook.

```python
raw = {
    "run_name": "smoke_convnext_tiny",
    "data": {"dataset": "fake_cifar10", "batch_size": 16},
    "model": {"name": "convnext_tiny", "width": 32},
}

print(raw["run_name"])
print(raw["data"]["batch_size"])
print(raw["model"]["name"])
```

Expected output:

```text
smoke_convnext_tiny
16
convnext_tiny
```

Mechanical meaning:

- each `[section]` in TOML becomes one nested dictionary
- `data.batch_size` in the final dataclass comes from `raw["data"]["batch_size"]`

### 17.4 Syntax Preflight: Dataclass Construction

Notebook: `experiments.ipynb`.

Purpose:

See how a dictionary becomes a typed object.

Action:

Run this in Python or the notebook.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoDataConfig:
    dataset: str = "fake_cifar10"
    batch_size: int = 64


raw_section = {"dataset": "cifar10", "batch_size": 128}
cfg = DemoDataConfig(**raw_section)

print(cfg.dataset)
print(cfg.batch_size)
```

Expected output:

```text
cifar10
128
```

Mechanical meaning:

- `**raw_section` expands dictionary keys into constructor arguments
- `frozen=True` makes the config immutable after creation
- immutable config prevents accidental mutation during training

### 17.5 File Edit: `configs/smoke.toml`

Action:

Create `configs/smoke.toml`.

```toml
run_name = "smoke_convnext_tiny"

[data]
dataset = "fake_cifar10"
data_dir = "data"
batch_size = 16
num_workers = 0
val_fraction = 0.25
augment = false
image_size = 32
num_classes = 10
subset_train = 64
subset_val = 32

[model]
name = "convnext_tiny"
num_classes = 10
width = 32
depth = 1
dropout = 0.0
kernel_size = 7
use_residual = true

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.01

[experiment]
seed = 0
epochs = 1
device = "auto"
output_dir = "artifacts"
log_interval = 1
```

Mechanical meaning:

- fake data avoids downloads during system checks
- tiny subsets make the run fast
- `num_workers = 0` is simplest and notebook-safe
- `device = "auto"` uses CUDA if available

### 17.6 File Edit: Full CIFAR-10 Configs

Action:

Create these files.

File: `configs/cifar10_mlp.toml`

```toml
run_name = "cifar10_mlp"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = true
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "mlp"
num_classes = 10
width = 512
depth = 2
dropout = 0.1
kernel_size = 3
use_residual = false

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.01

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

File: `configs/cifar10_lenet.toml`

```toml
run_name = "cifar10_lenet"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = true
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "lenet"
num_classes = 10
width = 0
depth = 0
dropout = 0.0
kernel_size = 5
use_residual = false

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.01

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

File: `configs/cifar10_alexnet_small.toml`

```toml
run_name = "cifar10_alexnet_small"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = true
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "alexnet_small"
num_classes = 10
width = 64
depth = 0
dropout = 0.2
kernel_size = 3
use_residual = false

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.01

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

File: `configs/cifar10_resnet_small.toml`

```toml
run_name = "cifar10_resnet_small"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = true
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "resnet_small"
num_classes = 10
width = 64
depth = 2
dropout = 0.0
kernel_size = 3
use_residual = true

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.01

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

File: `configs/cifar10_convnext_tiny.toml`

```toml
run_name = "cifar10_convnext_tiny"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = true
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "convnext_tiny"
num_classes = 10
width = 64
depth = 2
dropout = 0.1
kernel_size = 7
use_residual = true

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.05

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

### 17.7 File Edit: Ablation Configs

Purpose:

Ablations ask what a specific design choice contributed.

Action:

Create these configs by copying `cifar10_convnext_tiny.toml` and changing only the named fields.

File: `configs/ablation_convnext_no_residual.toml`

```toml
run_name = "ablation_convnext_no_residual"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = true
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "convnext_tiny"
num_classes = 10
width = 64
depth = 2
dropout = 0.1
kernel_size = 7
use_residual = false

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.05

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

File: `configs/ablation_convnext_small_kernel.toml`

```toml
run_name = "ablation_convnext_small_kernel"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = true
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "convnext_tiny"
num_classes = 10
width = 64
depth = 2
dropout = 0.1
kernel_size = 3
use_residual = true

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.05

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

File: `configs/ablation_convnext_no_augmentation.toml`

```toml
run_name = "ablation_convnext_no_augmentation"

[data]
dataset = "cifar10"
data_dir = "data"
batch_size = 128
num_workers = 2
val_fraction = 0.1
augment = false
image_size = 32
num_classes = 10
subset_train = 0
subset_val = 0

[model]
name = "convnext_tiny"
num_classes = 10
width = 64
depth = 2
dropout = 0.1
kernel_size = 7
use_residual = true

[optim]
name = "adamw"
lr = 0.001
weight_decay = 0.05

[experiment]
seed = 0
epochs = 20
device = "auto"
output_dir = "artifacts"
log_interval = 50
```

Mechanical meaning:

- no residual asks whether identity paths help optimization
- small kernel asks whether wider spatial context helps
- no augmentation asks whether training-time variation helps validation

### 17.8 File Edit: `config.py`

This section is the packaged config implementation.

Block grammar:

```text
imports:
    load TOML and dataclass tools

section dataclasses:
    one dataclass per config section

Config:
    top-level object used by the rest of the project

load_config:
    read TOML
    validate section shape
    construct typed config
```

Action:

Create `src/modern_cnn/config.py`.

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    # Python 3.11 and newer include tomllib.
    import tomllib
except ModuleNotFoundError:
    # Python 3.10 needs the tomli dependency from pyproject.toml.
    import tomli as tomllib


@dataclass(frozen=True)
class DataConfig:
    # "fake_cifar10" is for smoke tests; "cifar10" downloads real CIFAR-10.
    dataset: str = "fake_cifar10"

    # Root directory for downloaded datasets.
    data_dir: str = "data"

    # Number of examples per optimizer/evaluation batch.
    batch_size: int = 64

    # DataLoader workers. Use 0 for notebooks and smoke runs.
    num_workers: int = 0

    # Fraction of the training set reserved for validation.
    val_fraction: float = 0.1

    # Whether training data gets random crop/flip augmentation.
    augment: bool = False

    # CIFAR-10 is naturally 32x32, but keeping this explicit documents the contract.
    image_size: int = 32

    # CIFAR-10 has 10 classes.
    num_classes: int = 10

    # Optional subset limits for fast runs. 0 means use all available examples.
    subset_train: int = 0
    subset_val: int = 0


@dataclass(frozen=True)
class ModelConfig:
    # Registry key used by models.build_model.
    name: str = "convnext_tiny"

    # Final classifier output size.
    num_classes: int = 10

    # Base channel width for scalable models.
    width: int = 64

    # Number of repeated blocks in each stage for scalable models.
    depth: int = 2

    # Dropout probability in classifier heads.
    dropout: float = 0.1

    # ConvNeXt depthwise kernel size.
    kernel_size: int = 7

    # ConvNeXt ablation switch.
    use_residual: bool = True


@dataclass(frozen=True)
class OptimConfig:
    # Supported values: "adamw" and "sgd".
    name: str = "adamw"

    # Optimizer step size.
    lr: float = 1e-3

    # L2-style regularization. AdamW decouples this from gradient scaling.
    weight_decay: float = 0.01


@dataclass(frozen=True)
class ExperimentConfig:
    # Random seed for repeatability.
    seed: int = 0

    # Number of full passes through the training loader.
    epochs: int = 1

    # "auto" chooses CUDA when available.
    device: str = "auto"

    # Root directory where run artifacts are written.
    output_dir: str = "artifacts"

    # Print frequency measured in batches.
    log_interval: int = 50


@dataclass(frozen=True)
class Config:
    # Human-readable run name used in artifact directory names.
    run_name: str

    # Each nested table becomes one typed section.
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    experiment: ExperimentConfig

    def to_dict(self) -> dict[str, Any]:
        # Convert nested dataclasses to plain dicts for JSON artifact writing.
        return asdict(self)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    # Missing sections become empty dicts so dataclass defaults can fill them.
    value = raw.get(key, {})

    # If a user writes data = "bad" instead of [data], fail early with a clear error.
    if not isinstance(value, dict):
        raise TypeError(f"Config section {key!r} must be a TOML table.")

    return value


def load_config(path: str | Path) -> Config:
    # Normalize to Path so callers can pass strings or Path objects.
    path = Path(path)

    # tomllib reads bytes.
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    # Build the immutable typed config used by the rest of the project.
    return Config(
        run_name=str(raw.get("run_name", path.stem)),
        data=DataConfig(**_section(raw, "data")),
        model=ModelConfig(**_section(raw, "model")),
        optim=OptimConfig(**_section(raw, "optim")),
        experiment=ExperimentConfig(**_section(raw, "experiment")),
    )
```

### 17.9 Config Loader Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from modern_cnn.config import load_config

cfg = load_config("configs/smoke.toml")

print("run:", cfg.run_name)
print("dataset:", cfg.data.dataset)
print("batch_size:", cfg.data.batch_size)
print("model:", cfg.model.name)
print("device:", cfg.experiment.device)
```

Expected output:

```text
run: smoke_convnext_tiny
dataset: fake_cifar10
batch_size: 16
model: convnext_tiny
device: auto
```

Failure checks:

- `FileNotFoundError`: run from project root or fix path.
- `TypeError: unexpected keyword`: a TOML key does not match the dataclass field name.
- `Config section 'data' must be a TOML table`: TOML syntax used `data = ...` instead of `[data]`.

## 18. Phase 2: Data Pipeline

### 18.1 Goal

Create a data pipeline that gives every model the same input contract:

```text
images: [batch, 3, 32, 32]
labels: [batch]
```

The data module owns:

- CIFAR-10 class names
- transforms
- train/validation split
- fake-data smoke mode
- real CIFAR-10 mode
- dataloader construction

### 18.2 Exact Run Order

Use this order:

```text
1. Run the tensor contract preflight.
2. Run the transform preflight.
3. Run the split-index preflight.
4. Type data.py.
5. Run the fake-loader shape drill.
6. Run the class-name drill.
```

### 18.3 Syntax Preflight: Image Tensor Contract

Notebook: `experiments.ipynb`.

Purpose:

Make the batch shape concrete before touching datasets.

Action:

Run:

```python
import torch

batch = torch.randn(16, 3, 32, 32)
labels = torch.randint(0, 10, (16,))

print("batch:", batch.shape)
print("labels:", labels.shape)
print("one image:", batch[0].shape)
```

Expected output:

```text
batch: torch.Size([16, 3, 32, 32])
labels: torch.Size([16])
one image: torch.Size([3, 32, 32])
```

Mechanical meaning:

- dimension 0 is batch
- dimension 1 is channel
- dimensions 2 and 3 are spatial axes
- labels have one integer class per image

### 18.4 Syntax Preflight: Train Transform vs Eval Transform

Notebook: `experiments.ipynb`.

Purpose:

Understand why training and evaluation transforms differ.

Action:

Run:

```python
from torchvision import transforms

image_size = 32

train_transform = transforms.Compose([
    transforms.RandomCrop(image_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])

print(train_transform)
print(eval_transform)
```

Mechanical meaning:

- training transform can change examples to improve robustness
- evaluation transform should be stable so metrics are comparable
- never use random augmentation for validation/test metrics

### 18.5 Syntax Preflight: Deterministic Split Indices

Notebook: `experiments.ipynb`.

Purpose:

See how validation split is produced without leaking examples.

Action:

Run:

```python
import torch

n = 10
val_fraction = 0.2
seed = 0

generator = torch.Generator().manual_seed(seed)
indices = torch.randperm(n, generator=generator).tolist()
val_size = int(n * val_fraction)

val_idx = indices[:val_size]
train_idx = indices[val_size:]

print("train_idx:", train_idx)
print("val_idx:", val_idx)
print("overlap:", set(train_idx).intersection(set(val_idx)))
```

Expected pattern:

```text
overlap: set()
```

Mechanical meaning:

- `randperm` shuffles example IDs
- first slice becomes validation
- remaining slice becomes training
- the split is reproducible because the generator has a seed

### 18.6 File Edit: `data.py`

Block grammar:

```text
constants:
    CIFAR-10 class names

LoaderBundle:
    named return object for train, val, test loaders

split_indices:
    deterministic train/validation split

build_transforms:
    train-time random transforms and eval-time stable transforms

_make_base_dataset:
    fake data or real CIFAR-10

build_dataloaders:
    assemble datasets, subsets, loaders, and class names
```

Action:

Create `src/modern_cnn/data.py`.

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import Config


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class LoaderBundle:
    # Training loader uses shuffle=True and training transforms.
    train: DataLoader

    # Validation loader uses shuffle=False and evaluation transforms.
    val: DataLoader

    # Test loader uses the official test split when dataset="cifar10".
    test: DataLoader

    # Human-readable names for confusion matrices and error grids.
    class_names: tuple[str, ...]


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    # The validation fraction must leave examples for both train and validation.
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    # Use a local generator so this split is reproducible and does not depend on global RNG state.
    generator = torch.Generator().manual_seed(seed)

    # randperm returns shuffled integer example IDs from 0 to n - 1.
    indices = torch.randperm(n, generator=generator).tolist()

    # At least one validation example, even for tiny fake datasets.
    val_size = max(1, int(n * val_fraction))

    # Validation gets the first slice; training gets the rest.
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def build_transforms(augment: bool, image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # Evaluation transform is deterministic. Metrics should not depend on random crop/flip.
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # If augment is disabled, train and eval transforms are intentionally identical.
    if not augment:
        return eval_transform, eval_transform

    # Training transform adds random crop and horizontal flip.
    # This teaches the model not to depend too much on exact pixel placement.
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_transform, eval_transform


def _make_base_dataset(cfg: Config, train: bool, transform: transforms.Compose):
    # FakeData mimics the CIFAR tensor contract without downloading data.
    if cfg.data.dataset == "fake_cifar10":
        image_shape = (3, cfg.data.image_size, cfg.data.image_size)
        if train:
            size = max(cfg.data.subset_train, 1)
        else:
            size = max(cfg.data.subset_val, 32)
        return datasets.FakeData(
            size=size,
            image_size=image_shape,
            num_classes=cfg.data.num_classes,
            transform=transform,
        )

    # Real CIFAR-10 downloads into cfg.data.data_dir.
    if cfg.data.dataset == "cifar10":
        return datasets.CIFAR10(
            root=cfg.data.data_dir,
            train=train,
            download=True,
            transform=transform,
        )

    raise ValueError(f"Unknown dataset: {cfg.data.dataset}")


def _maybe_subset(dataset, limit: int):
    # limit=0 means use the whole dataset.
    if limit <= 0 or limit >= len(dataset):
        return dataset

    # Use the first limit examples after the split/subset object has been created.
    return Subset(dataset, list(range(limit)))


def build_dataloaders(cfg: Config) -> LoaderBundle:
    # Training and evaluation transforms are built together to keep the policy explicit.
    train_transform, eval_transform = build_transforms(cfg.data.augment, cfg.data.image_size)

    # Use a seeded generator for reproducible shuffle order in the training DataLoader.
    generator = torch.Generator().manual_seed(cfg.experiment.seed)

    # Build two train=True base datasets so train and val can use different transforms.
    train_base = _make_base_dataset(cfg, train=True, transform=train_transform)
    val_base = _make_base_dataset(cfg, train=True, transform=eval_transform)

    # Test set is separate for real CIFAR-10 and fake for smoke mode.
    test_base = _make_base_dataset(cfg, train=False, transform=eval_transform)

    # Split by example indices so train and validation do not overlap.
    train_idx, val_idx = split_indices(len(train_base), cfg.data.val_fraction, cfg.experiment.seed)

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)
    test_ds = test_base

    # Optional fast subsets are useful for local checks and debugging.
    train_ds = _maybe_subset(train_ds, cfg.data.subset_train)
    val_ds = _maybe_subset(val_ds, cfg.data.subset_val)

    loader_kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return LoaderBundle(
        train=DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs),
        val=DataLoader(val_ds, shuffle=False, **loader_kwargs),
        test=DataLoader(test_ds, shuffle=False, **loader_kwargs),
        class_names=CIFAR10_CLASSES,
    )
```

### 18.7 Data Shape Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from modern_cnn.config import load_config
from modern_cnn.data import build_dataloaders

cfg = load_config("configs/smoke.toml")
loaders = build_dataloaders(cfg)

images, labels = next(iter(loaders.train))

print("images:", images.shape)
print("labels:", labels.shape)
print("classes:", loaders.class_names)
```

Expected output pattern:

```text
images: torch.Size([16, 3, 32, 32])
labels: torch.Size([16])
```

Failure checks:

- `RuntimeError` from worker process: set `num_workers = 0` for notebook runs.
- wrong image shape: check `image_size` and transform order.
- labels shape `[16, 1]`: labels should be integer class IDs with shape `[batch]`.

## 19. Phase 3: Metrics

### 19.1 Goal

Metrics turn logits into evidence.

The model returns logits:

```text
[batch, num_classes]
```

Metrics compare those logits to labels:

```text
[batch]
```

This phase implements:

- accuracy
- confusion matrix
- class accuracy
- average meter
- parameter count

### 19.2 Exact Run Order

Use this order:

```text
1. Run argmax drill.
2. Run confusion-matrix indexing drill.
3. Run AverageMeter drill.
4. Type metrics.py.
5. Run final metric import drill.
```

### 19.3 Syntax Preflight: `argmax(dim=1)`

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch

logits = torch.tensor([
    [0.1, 3.0, 0.2],
    [2.0, 0.5, 0.1],
])

predicted = logits.argmax(dim=1)

print("logits:", logits.shape)
print("predicted:", predicted)
```

Expected output:

```text
logits: torch.Size([2, 3])
predicted: tensor([1, 0])
```

Mechanical meaning:

- `dim=1` compares classes within each example
- `dim=0` would compare examples against each other, which is wrong for classification

### 19.4 Syntax Preflight: Confusion Matrix Flattening

Notebook: `experiments.ipynb`.

Purpose:

See how `(true_class, predicted_class)` pairs become row/column counts.

Action:

Run:

```python
import torch

labels = torch.tensor([0, 1, 1, 2])
predicted = torch.tensor([0, 2, 1, 2])
num_classes = 3

flat = labels * num_classes + predicted
counts = torch.bincount(flat, minlength=num_classes * num_classes)
cm = counts.reshape(num_classes, num_classes)

print("flat:", flat)
print(cm)
```

Expected output:

```text
flat: tensor([0, 5, 4, 8])
tensor([[1, 0, 0],
        [0, 1, 1],
        [0, 0, 1]])
```

Mechanical meaning:

- rows are true classes
- columns are predicted classes
- row 1 column 2 means true class 1 predicted as class 2

### 19.5 Syntax Preflight: AverageMeter

Notebook: `experiments.ipynb`.

Purpose:

Training loss must be averaged by number of examples, not number of batches, because the last batch can be smaller.

Action:

Run:

```python
total = 0.0
count = 0

batch_loss = 2.0
batch_size = 16
total += batch_loss * batch_size
count += batch_size

batch_loss = 1.0
batch_size = 4
total += batch_loss * batch_size
count += batch_size

print(total / count)
```

Expected output:

```text
1.8
```

Mechanical meaning:

- the large batch contributes more examples
- simple average of `2.0` and `1.0` would incorrectly give `1.5`

### 19.6 File Edit: `metrics.py`

Action:

Create `src/modern_cnn/metrics.py`.

```python
from __future__ import annotations

import torch
from torch import nn


class AverageMeter:
    def __init__(self) -> None:
        # Running weighted sum of metric values.
        self.total = 0.0

        # Number of examples represented by the total.
        self.count = 0

    def update(self, value: float, n: int) -> None:
        # Multiply by n so a metric from a larger batch contributes more.
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        # Empty meters appear before any update; return 0.0 instead of crashing.
        if self.count == 0:
            return 0.0
        return self.total / self.count


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # logits shape: [batch, classes]
    # labels shape: [batch]
    predicted = logits.argmax(dim=1)

    # Convert boolean matches to float and average over examples.
    return float((predicted == labels).float().mean().item())


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # predicted shape: [batch]
    predicted = logits.argmax(dim=1)

    # Encode each (true, predicted) pair as one integer.
    # true class controls the row; predicted class controls the column.
    flat = labels * num_classes + predicted

    # Count all true/predicted pairs, including pairs that did not occur.
    counts = torch.bincount(flat, minlength=num_classes * num_classes)

    # Reshape back into [true_class, predicted_class].
    return counts.reshape(num_classes, num_classes)


def class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    # Correct predictions are on the diagonal.
    correct = cm.diag()

    # Row totals are the number of examples for each true class.
    totals = cm.sum(dim=1).clamp_min(1)

    return correct / totals


def parameter_count(model: nn.Module) -> int:
    # Count trainable parameters only.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 19.7 Metric Import Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch

from modern_cnn.metrics import accuracy, confusion_matrix

logits = torch.tensor([[1.0, 4.0], [3.0, 0.0], [0.0, 2.0]])
labels = torch.tensor([1, 1, 1])

print("accuracy:", accuracy(logits, labels))
print(confusion_matrix(logits, labels, num_classes=2))
```

Expected output:

```text
accuracy: 0.6666666865348816
tensor([[0, 0],
        [1, 2]])
```

## 20. Phase 4: CNN Blocks

### 20.1 Goal

Build reusable CNN blocks before building full models.

This phase teaches:

- Conv-BatchNorm-ReLU
- residual addition
- projection skip paths
- NCHW LayerNorm
- depthwise convolution
- ConvNeXt block grammar

### 20.2 Exact Run Order

Use this order:

```text
1. Run convolution output-shape drill.
2. Run ConvBNReLU drill.
3. Run residual addition drill.
4. Run projection skip drill.
5. Run LayerNorm permutation drill.
6. Run depthwise parameter drill.
7. Type blocks.py.
8. Run block import tests.
```

### 20.3 Syntax Preflight: Convolution Output Shape

Notebook: `experiments.ipynb`.

Formula:

```text
output = floor((input + 2*padding - kernel_size) / stride) + 1
```

Action:

Run:

```python
import torch
from torch import nn

x = torch.randn(2, 3, 32, 32)
conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
y = conv(x)

print(y.shape)
```

Expected output:

```text
torch.Size([2, 16, 16, 16])
```

Mechanical meaning:

- batch remains 2
- output channels become 16
- spatial dimensions become 16x16 because stride is 2

### 20.4 Syntax Preflight: Residual Addition

Notebook: `experiments.ipynb`.

Purpose:

See why shapes must match before addition.

Action:

Run:

```python
import torch

x = torch.randn(2, 16, 8, 8)
main = torch.randn(2, 16, 8, 8)

out = main + x

print(out.shape)
```

Expected output:

```text
torch.Size([2, 16, 8, 8])
```

Failure drill:

```python
bad_skip = torch.randn(2, 32, 8, 8)
main + bad_skip
```

Expected failure:

```text
RuntimeError
```

Mechanical meaning:

- residual addition is elementwise
- every dimension must be compatible
- changing channels requires a projection path

### 20.5 Syntax Preflight: LayerNorm2d Permutation

Notebook: `experiments.ipynb`.

Purpose:

PyTorch `nn.LayerNorm(channels)` expects channels to be the last dimension. Images usually use NCHW format.

Action:

Run:

```python
import torch
from torch import nn

x = torch.randn(2, 16, 8, 8)
norm = nn.LayerNorm(16)

x_nhwc = x.permute(0, 2, 3, 1)
y_nhwc = norm(x_nhwc)
y = y_nhwc.permute(0, 3, 1, 2)

print("x:", x.shape)
print("x_nhwc:", x_nhwc.shape)
print("y:", y.shape)
```

Expected output:

```text
x: torch.Size([2, 16, 8, 8])
x_nhwc: torch.Size([2, 8, 8, 16])
y: torch.Size([2, 16, 8, 8])
```

Mechanical meaning:

- temporary NHWC makes channels the last dimension
- final permutation restores NCHW for Conv2d

### 20.6 Syntax Preflight: Depthwise Convolution

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from torch import nn

channels = 32
kernel_size = 7

regular = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=3)
depthwise = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=3, groups=channels)

print("regular weights:", regular.weight.numel())
print("depthwise weights:", depthwise.weight.numel())
```

Expected output:

```text
regular weights: 50176
depthwise weights: 1568
```

Mechanical meaning:

- regular conv has `out_channels * in_channels * kernel * kernel`
- depthwise conv has `channels * 1 * kernel * kernel`
- `groups=channels` prevents cross-channel mixing in that layer

### 20.7 File Edit: `blocks.py`

Block grammar:

```text
ConvBNReLU:
    convolution changes channels/spatial size
    BatchNorm stabilizes channel activation scale
    ReLU adds nonlinearity

ResidualBlock:
    main path learns transformation
    skip path carries identity or projection
    addition combines them

LayerNorm2d:
    permute NCHW -> NHWC
    apply LayerNorm over channels
    permute NHWC -> NCHW

ConvNeXtBlock:
    depthwise conv handles spatial mixing
    LayerNorm normalizes channels
    pointwise conv expands channel dimension
    GELU adds smooth nonlinearity
    pointwise conv contracts back
    optional residual addition
```

Action:

Create `src/modern_cnn/blocks.py`.

```python
from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        # Same-padding for odd kernels when stride=1.
        padding = kernel_size // 2

        super().__init__(
            # bias=False because BatchNorm has learnable shift.
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),

            # Normalize each output channel over batch,height,width during training.
            nn.BatchNorm2d(out_channels),

            # ReLU keeps positive activations and zeroes negative activations.
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Main path: conv block, then another conv+BN without final ReLU.
        # The final ReLU is applied after adding the skip path.
        self.main = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # If spatial size or channel count changes, raw x cannot be added to main(x).
        needs_projection = stride != 1 or in_channels != out_channels

        if needs_projection:
            self.skip = nn.Sequential(
                # 1x1 conv changes channels and optionally downsamples spatial size.
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # Identity is valid only when shapes already match.
            self.skip = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # main(x) and skip(x) must have identical shape.
        out = self.main(x) + self.skip(x)
        return self.activation(out)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is NCHW: [batch, channels, height, width].
        x = x.permute(0, 2, 3, 1)

        # Now shape is NHWC, so the last dimension is channels.
        x = self.norm(x)

        # Restore NCHW for Conv2d layers.
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale: float = 1e-6,
        use_residual: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2
        hidden = channels * expansion
        self.use_residual = use_residual

        # Depthwise convolution learns spatial filters independently per channel.
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )

        # ConvNeXt uses LayerNorm-style normalization rather than BatchNorm inside the block.
        self.norm = LayerNorm2d(channels)

        # Pointwise channel mixer.
        # First 1x1 expands channels, GELU gates smoothly, second 1x1 contracts back.
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

        # Layer scale starts the residual branch small.
        # Shape [channels, 1, 1] broadcasts across batch,height,width.
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)
        x = self.gamma * x

        if self.use_residual:
            x = x + residual

        return x
```

### 20.8 Block Shape Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch

from modern_cnn.blocks import ConvBNReLU, ConvNeXtBlock, ResidualBlock

x = torch.randn(2, 16, 16, 16)

conv = ConvBNReLU(16, 32, stride=2)
res = ResidualBlock(16, 32, stride=2)
cnx = ConvNeXtBlock(16)

print("ConvBNReLU:", conv(x).shape)
print("ResidualBlock:", res(x).shape)
print("ConvNeXtBlock:", cnx(x).shape)
```

Expected output:

```text
ConvBNReLU: torch.Size([2, 32, 8, 8])
ResidualBlock: torch.Size([2, 32, 8, 8])
ConvNeXtBlock: torch.Size([2, 16, 16, 16])
```

Failure checks:

- residual add error: projection path is missing or wrong stride
- LayerNorm error: forgot NCHW/NHWC permutation
- depthwise conv error: `groups` must divide `in_channels` and `out_channels`

## 21. Phase 5: Models

### 21.1 Goal

Implement model families behind one registry.

The training loop should not care whether the model is an MLP, LeNet, AlexNet-small, ResNet-small, or ConvNeXt-tiny.

Universal contract:

```text
input: [batch, 3, 32, 32]
output: [batch, 10]
```

### 21.2 Exact Run Order

Use this order:

```text
1. Run flatten baseline drill.
2. Run LeNet shape drill.
3. Run global average pooling drill.
4. Run model registry grammar drill.
5. Type models.py.
6. Run model output shape drill.
7. Run parameter count drill.
8. Run trace_shapes drill.
```

### 21.3 Syntax Preflight: Flatten Baseline

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch
from torch import nn

x = torch.randn(4, 3, 32, 32)
flatten = nn.Flatten()
y = flatten(x)

print(y.shape)
```

Expected output:

```text
torch.Size([4, 3072])
```

Mechanical meaning:

- MLP sees the image as 3072 features
- spatial structure is no longer explicit

### 21.4 Syntax Preflight: LeNet-CIFAR Shape

Notebook: `experiments.ipynb`.

Purpose:

LeNet for MNIST used 1 channel and 28x28 inputs. CIFAR-10 uses 3 channels and 32x32 inputs.

Action:

Run:

```python
import torch
from torch import nn

features = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
)

x = torch.randn(1, 3, 32, 32)
y = features(x)

print(y.shape)
print("flattened:", y.flatten(1).shape)
```

Expected output:

```text
torch.Size([1, 16, 6, 6])
flattened: torch.Size([1, 576])
```

Mechanical meaning:

- first conv keeps 32x32 because padding is 2
- first pool gives 16x16
- second conv with kernel 5 gives 12x12
- second pool gives 6x6
- dense input is `16 * 6 * 6`, not `16 * 5 * 5`

### 21.5 Syntax Preflight: Global Average Pooling

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch
from torch import nn

x = torch.randn(4, 256, 4, 4)
pool = nn.AdaptiveAvgPool2d((1, 1))
y = pool(x)
z = y.flatten(1)

print("pooled:", y.shape)
print("flattened:", z.shape)
```

Expected output:

```text
pooled: torch.Size([4, 256, 1, 1])
flattened: torch.Size([4, 256])
```

Mechanical meaning:

- classifier input depends on channels, not spatial size
- this reduces dense head parameter count

### 21.6 Syntax Preflight: Registry Grammar

Notebook: `experiments.ipynb`.

Purpose:

The registry turns a string model name into a module.

Action:

Run:

```python
from torch import nn


def tiny_registry(name):
    if name == "linear":
        return nn.Linear(4, 2)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown model: {name}")


print(tiny_registry("linear"))
print(tiny_registry("identity"))
```

Mechanical meaning:

- configs can select models by name
- train/eval code can stay generic
- unknown names fail early

### 21.7 File Edit: `models.py`

Block grammar:

```text
MLPBaseline:
    flatten image
    dense layers
    class logits

LeNetCIFAR:
    historical CNN adapted to 3x32x32

AlexNetSmall:
    ReLU, max pooling, deeper conv stack
    adaptive pooling to keep classifier size controlled

ResNetSmall:
    stem
    residual stages
    global average pooling
    linear head

ConvNeXtTinyCIFAR:
    patch-style stem
    ConvNeXt stages
    downsample between stages
    global average pooling
    layer norm and head

build_model:
    config name -> model object

trace_shapes:
    forward hooks collect layer output shapes
```

Action:

Create `src/modern_cnn/models.py`.

```python
from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBNReLU, ConvNeXtBlock, LayerNorm2d, ResidualBlock
from .config import ModelConfig


class MLPBaseline(nn.Sequential):
    def __init__(self, num_classes: int = 10, width: int = 512, dropout: float = 0.1):
        super().__init__(
            # CIFAR-10 image [3, 32, 32] becomes a 3072-feature vector.
            nn.Flatten(),

            # Dense layer ignores spatial locality, which makes this a useful weak baseline.
            nn.Linear(3 * 32 * 32, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Second dense layer adds capacity but still has no convolutional image bias.
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Output logits for 10 classes.
            nn.Linear(width, num_classes),
        )


class LeNetCIFAR(nn.Sequential):
    def __init__(self, num_classes: int = 10):
        super().__init__(
            # Input is RGB CIFAR, so in_channels=3 instead of MNIST's 1.
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # Shape after first pool: [batch, 6, 16, 16].
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # Shape after second pool: [batch, 16, 6, 6].
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )


class AlexNetSmall(nn.Sequential):
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__(
            # CIFAR is small, so this scaled AlexNet uses 3x3 kernels instead of the original 11x11 stem.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Deeper conv stack increases feature variety.
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Adaptive pooling fixes the classifier input size.
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),

            # Dense classifier head, smaller than historical AlexNet.
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )


class ResNetSmall(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64, depth: int = 2):
        super().__init__()

        # Stem keeps CIFAR spatial size at 32x32.
        self.stem = ConvBNReLU(3, width, kernel_size=3, stride=1)

        # Stage 1 keeps spatial size.
        self.stage1 = self._stage(width, width, blocks=depth, stride=1)

        # Stage 2 halves spatial size and doubles channels.
        self.stage2 = self._stage(width, width * 2, blocks=depth, stride=2)

        # Stage 3 halves spatial size again and doubles channels again.
        self.stage3 = self._stage(width * 2, width * 4, blocks=depth, stride=2)

        # Global average pooling produces [batch, width*4].
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width * 4, num_classes)

    def _stage(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        # First block may change spatial size/channels.
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]

        # Later blocks in the same stage preserve shape.
        for _ in range(blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class ConvNeXtTinyCIFAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 7,
        use_residual: bool = True,
    ):
        super().__init__()

        # Three stages are enough for CIFAR-scale learning.
        dims = [width, width * 2, width * 4]

        # Patch-style stem: 32x32 -> 8x8 when kernel=stride=4.
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        self.stage1 = self._stage(dims[0], depth, kernel_size, use_residual)
        self.down1 = self._downsample(dims[0], dims[1])

        self.stage2 = self._stage(dims[1], depth, kernel_size, use_residual)
        self.down2 = self._downsample(dims[1], dims[2])

        # Final stage gets one extra block because it has the richest channels.
        self.stage3 = self._stage(dims[2], depth + 1, kernel_size, use_residual)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(dims[2])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dims[2], num_classes)

    def _stage(self, channels: int, depth: int, kernel_size: int, use_residual: bool) -> nn.Sequential:
        return nn.Sequential(
            *[
                ConvNeXtBlock(channels, kernel_size=kernel_size, use_residual=use_residual)
                for _ in range(depth)
            ]
        )

    def _downsample(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            # Normalize before downsampling convolution.
            LayerNorm2d(in_channels),

            # Halve spatial dimensions and increase channels.
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        # [batch, channels, height, width] -> [batch, channels]
        x = self.pool(x).flatten(1)

        # nn.LayerNorm here expects [batch, channels], so no LayerNorm2d wrapper is needed.
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x)


def build_model(
    name: str,
    num_classes: int = 10,
    width: int = 64,
    depth: int = 2,
    dropout: float = 0.1,
    kernel_size: int = 7,
    use_residual: bool = True,
) -> nn.Module:
    if name == "mlp":
        return MLPBaseline(num_classes=num_classes, width=width, dropout=dropout)

    if name == "lenet":
        return LeNetCIFAR(num_classes=num_classes)

    if name == "alexnet_small":
        return AlexNetSmall(num_classes=num_classes, dropout=dropout)

    if name == "resnet_small":
        return ResNetSmall(num_classes=num_classes, width=width, depth=depth)

    if name == "convnext_tiny":
        return ConvNeXtTinyCIFAR(
            num_classes=num_classes,
            width=width,
            depth=depth,
            dropout=dropout,
            kernel_size=kernel_size,
            use_residual=use_residual,
        )

    raise ValueError(f"Unknown model: {name}")


def build_model_from_config(cfg: ModelConfig) -> nn.Module:
    # Central bridge from config object to model object.
    return build_model(
        name=cfg.name,
        num_classes=cfg.num_classes,
        width=cfg.width,
        depth=cfg.depth,
        dropout=cfg.dropout,
        kernel_size=cfg.kernel_size,
        use_residual=cfg.use_residual,
    )


def trace_shapes(model: nn.Module, x: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    rows: list[tuple[str, tuple[int, ...]]] = []

    def hook(name: str):
        def capture(_module, _inputs, output):
            # Some modules could return tuples; this project tracks tensor outputs only.
            if isinstance(output, torch.Tensor):
                rows.append((name, tuple(output.shape)))
        return capture

    handles = []

    # Register hooks only on leaf modules so the trace is not overwhelmed by containers.
    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(hook(name)))

    model.eval()
    with torch.no_grad():
        model(x)

    # Always remove hooks so future forward passes are not polluted.
    for handle in handles:
        handle.remove()

    return rows
```

### 21.8 Model Output Shape Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch

from modern_cnn.models import build_model

for name in ["mlp", "lenet", "alexnet_small", "resnet_small", "convnext_tiny"]:
    model = build_model(name, num_classes=10, width=16, depth=1)
    logits = model(torch.randn(2, 3, 32, 32))
    print(name, logits.shape)
```

Expected output pattern:

```text
mlp torch.Size([2, 10])
lenet torch.Size([2, 10])
alexnet_small torch.Size([2, 10])
resnet_small torch.Size([2, 10])
convnext_tiny torch.Size([2, 10])
```

### 21.9 Parameter Count Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from modern_cnn.metrics import parameter_count
from modern_cnn.models import build_model

for name in ["mlp", "lenet", "alexnet_small", "resnet_small", "convnext_tiny"]:
    model = build_model(name, num_classes=10, width=32, depth=1)
    print(f"{name:16s} {parameter_count(model):8d}")
```

Interpretation:

- MLP parameters mostly live in dense layers
- LeNet is small but old-fashioned
- AlexNet-small has large convolutional capacity
- ResNet-small uses residual blocks
- ConvNeXt-tiny uses modern block structure

### 21.10 Shape Trace Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch

from modern_cnn.models import build_model, trace_shapes

model = build_model("convnext_tiny", num_classes=10, width=32, depth=1)
rows = trace_shapes(model, torch.randn(1, 3, 32, 32))

for name, out_shape in rows:
    print(f"{name:40s} -> {out_shape}")
```

Failure checks:

- final output is not `[batch, 10]`: head or flatten/pooling is wrong
- LeNet dense mismatch: CIFAR flattened size should be `16 * 6 * 6`
- ConvNeXt LayerNorm error: check LayerNorm2d permutation

## 22. Phase 6: Evaluation

### 22.1 Goal

Evaluation measures behavior without changing the model.

This requires:

```text
model.eval()
torch.no_grad()
```

Evaluation produces:

- average loss
- average accuracy
- confusion matrix

### 22.2 Exact Run Order

Use this order:

```text
1. Run no_grad drill.
2. Run eval-loop grammar drill.
3. Type evaluate.py.
4. Run smoke evaluation drill after data/model code works.
```

### 22.3 Syntax Preflight: `torch.no_grad`

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch
from torch import nn

model = nn.Linear(4, 2)
x = torch.randn(3, 4)

with torch.no_grad():
    y = model(x)

print(y.requires_grad)
```

Expected output:

```text
False
```

Mechanical meaning:

- evaluation does not build a backward graph
- memory use is lower
- parameters are not updated

### 22.4 Syntax Preflight: Eval Loop Grammar

Notebook: `experiments.ipynb`.

Read the loop as:

```text
for each batch:
    move tensors to device
    compute logits
    compute loss
    update weighted loss meter
    update weighted accuracy meter
    add batch confusion matrix
```

The evaluation loop is almost training without:

- `optimizer.zero_grad`
- `loss.backward`
- `optimizer.step`

### 22.5 File Edit: `evaluate.py`

Action:

Create `src/modern_cnn/evaluate.py`.

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import AverageMeter, accuracy, class_accuracy, confusion_matrix


@dataclass(frozen=True)
class EvalOutput:
    # Average loss over examples.
    loss: float

    # Average accuracy over examples.
    accuracy: float

    # Confusion matrix where rows=true class and columns=predicted class.
    confusion: torch.Tensor


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> EvalOutput:
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = images.size(0)
            loss_meter.update(float(loss.item()), batch_size)
            acc_meter.update(accuracy(logits, labels), batch_size)

            # Move to CPU so the accumulated matrix is device-independent.
            total_cm += confusion_matrix(logits.cpu(), labels.cpu(), num_classes)

    return EvalOutput(
        loss=loss_meter.average,
        accuracy=acc_meter.average,
        confusion=total_cm,
    )


def class_accuracy_dict(confusion: torch.Tensor, class_names: tuple[str, ...]) -> dict[str, float]:
    values = class_accuracy(confusion)
    return {
        class_name: float(values[i].item())
        for i, class_name in enumerate(class_names)
    }


def collect_misclassified(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit: int = 16,
) -> list[tuple[torch.Tensor, int, int]]:
    model.eval()
    examples: list[tuple[torch.Tensor, int, int]] = []

    with torch.no_grad():
        for images, labels in loader:
            images_device = images.to(device)
            logits = model(images_device)
            predicted = logits.argmax(dim=1).cpu()

            for image, label, pred in zip(images, labels, predicted):
                if int(label) != int(pred):
                    # Store CPU image tensor and plain integer labels.
                    examples.append((image.cpu(), int(label), int(pred)))

                if len(examples) >= limit:
                    return examples

    return examples
```

### 22.6 Smoke Evaluation Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch
from torch import nn

from modern_cnn.config import load_config
from modern_cnn.data import build_dataloaders
from modern_cnn.evaluate import evaluate
from modern_cnn.models import build_model_from_config

cfg = load_config("configs/smoke.toml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaders = build_dataloaders(cfg)
model = build_model_from_config(cfg.model).to(device)
output = evaluate(model, loaders.val, nn.CrossEntropyLoss(), device, cfg.data.num_classes)

print(output.loss)
print(output.accuracy)
print(output.confusion.shape)
```

Expected shape:

```text
torch.Size([10, 10])
```

## 23. Phase 7: Checkpoints

### 23.1 Goal

Checkpoints preserve trainable state.

For this project, a checkpoint stores:

- model weights
- optimizer state
- epoch
- best metric
- resolved config

### 23.2 Exact Run Order

Use this order:

```text
1. Run state_dict preflight.
2. Type checkpoint.py.
3. Run checkpoint round-trip drill.
4. Add checkpoint test later.
```

### 23.3 Syntax Preflight: `state_dict`

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from torch import nn

model = nn.Linear(4, 2)
state = model.state_dict()

print(state.keys())
print(state["weight"].shape)
print(state["bias"].shape)
```

Expected output:

```text
odict_keys(['weight', 'bias'])
torch.Size([2, 4])
torch.Size([2])
```

Mechanical meaning:

- `state_dict` contains tensor values, not the Python class definition
- when loading, the target model architecture must match

### 23.4 File Edit: `checkpoint.py`

Action:

Create `src/modern_cnn/checkpoint.py`.

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metric: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        # Model trainable values.
        "model_state": model.state_dict(),

        # Optimizer momentum/adaptive state. May be None for evaluation-only saves.
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,

        # Epoch number that produced this checkpoint.
        "epoch": epoch,

        # Usually best validation accuracy.
        "metric": metric,

        # Resolved config dictionary for reproducibility.
        "config": config,
    }

    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
```

### 23.5 Checkpoint Round-Trip Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from modern_cnn.checkpoint import load_checkpoint, save_checkpoint
from modern_cnn.models import build_model

model = build_model("lenet", num_classes=10)
path = "artifacts/drills/checkpoint.pt"

save_checkpoint(path, model=model, optimizer=None, epoch=3, metric=0.25, config={"run_name": "drill"})
payload = load_checkpoint(path)

print(payload.keys())
print(payload["epoch"])
print(payload["metric"])
```

Expected output pattern:

```text
dict_keys([...])
3
0.25
```

Failure checks:

- parent directory missing: `save_checkpoint` should create it
- load mismatch later: checkpoint model architecture must match config model architecture

## 24. Phase 8: Artifacts

### 24.1 Goal

Artifacts are the evidence trail for a run.

This project writes:

- copied config
- resolved config JSON
- metrics JSONL
- metrics plot
- confusion matrix image
- misclassification grid
- best checkpoint

### 24.2 Exact Run Order

Use this order:

```text
1. Run JSON drill.
2. Run JSONL drill.
3. Run run-directory drill.
4. Type artifacts.py.
5. Run artifact write/read drill.
6. Use artifact functions inside training and CLI.
```

### 24.3 Syntax Preflight: JSON

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import json

payload = {"run_name": "drill", "accuracy": 0.5}
text = json.dumps(payload, indent=2, sort_keys=True)

print(text)
```

Mechanical meaning:

- JSON stores one structured object
- good for configs and summaries

### 24.4 Syntax Preflight: JSONL

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import json

rows = [
    {"epoch": 1, "loss": 2.3},
    {"epoch": 2, "loss": 2.1},
]

text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
print(text)
```

Mechanical meaning:

- JSONL stores one JSON object per line
- good for metrics that grow epoch by epoch
- easy to append during training

### 24.5 Syntax Preflight: Run Directory Names

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from datetime import datetime

run_name = "smoke_convnext_tiny"
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"{stamp}_{run_name}")
```

Expected pattern:

```text
20260824_153000_smoke_convnext_tiny
```

Mechanical meaning:

- each run writes to a unique directory
- run names remain human-readable

### 24.6 File Edit: `artifacts.py`

Action:

Create `src/modern_cnn/artifacts.py`.

```python
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def make_run_dir(root: str | Path, run_name: str) -> Path:
    # Timestamp prevents one run from overwriting another run.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{stamp}_{run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Append one JSON row. This keeps metrics writable one epoch at a time.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def copy_config(src: str | Path, run_dir: str | Path) -> None:
    # Preserve the exact config file used to launch the run.
    shutil.copy2(src, Path(run_dir) / "config.toml")


def plot_history(jsonl_path: str | Path, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    rows = read_jsonl(jsonl_path)
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in rows], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_image(
    confusion: torch.Tensor,
    class_names: tuple[str, ...],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    cm = confusion.cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _unnormalize_cifar(image: torch.Tensor) -> torch.Tensor:
    # Images were normalized by channel; undo that for display.
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def save_misclassified_grid(
    examples: list[tuple[torch.Tensor, int, int]],
    class_names: tuple[str, ...],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    if not examples:
        return

    cols = min(4, len(examples))
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    for ax, (image, true_label, predicted_label) in zip(axes_list, examples):
        display = _unnormalize_cifar(image).permute(1, 2, 0).numpy()
        ax.imshow(display)
        ax.set_title(f"T: {class_names[true_label]}\nP: {class_names[predicted_label]}")
        ax.axis("off")

    for ax in axes_list[len(examples):]:
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

### 24.7 Artifact Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from modern_cnn.artifacts import append_jsonl, read_jsonl, save_json

save_json("artifacts/drills/config.json", {"run_name": "drill"})
append_jsonl("artifacts/drills/metrics.jsonl", {"epoch": 1, "loss": 2.3})
append_jsonl("artifacts/drills/metrics.jsonl", {"epoch": 2, "loss": 2.1})

rows = read_jsonl("artifacts/drills/metrics.jsonl")

print(rows)
```

Expected output pattern:

```text
[{'epoch': 1, 'loss': 2.3}, {'epoch': 2, 'loss': 2.1}]
```

Failure checks:

- JSON not readable: check file path and write permissions
- metric plot missing: check that `metrics.jsonl` has expected keys

## 25. Phase 9: Training

### 25.1 Goal

Training connects:

```text
data batch
-> model logits
-> classification loss
-> gradients
-> optimizer update
-> metrics
-> checkpoint/artifacts
```

This phase must be mechanically explicit.

### 25.2 Exact Run Order

Use this order:

```text
1. Run device-choice drill.
2. Run one-batch update drill.
3. Run train/eval mode drill.
4. Type train.py.
5. Run train_one_epoch drill.
6. Run smoke training through Python.
7. Run smoke training through CLI after cli.py exists.
```

### 25.3 Syntax Preflight: Device Choice

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.randn(2, 3).to(device)
print(x.device)
```

Mechanical meaning:

- model and tensors must be on the same device
- `"auto"` should resolve to CUDA when available and CPU otherwise

### 25.4 Syntax Preflight: One-Batch Update

Notebook: `experiments.ipynb`.

Purpose:

Prove the optimizer changes parameters.

Action:

Run:

```python
import torch
from torch import nn

model = nn.Linear(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

X = torch.randn(5, 4)
y = torch.tensor([0, 1, 0, 1, 0])

before = model.weight.detach().clone()

logits = model(X)
loss = loss_fn(logits, y)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

after = model.weight.detach().clone()

print("loss:", float(loss.detach()))
print("changed:", not torch.allclose(before, after))
```

Expected output:

```text
changed: True
```

Mechanical meaning:

- forward pass creates logits
- loss compares logits to integer labels
- backward populates gradients
- optimizer step mutates parameters

### 25.5 Syntax Preflight: Train vs Eval Mode

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from torch import nn

model = nn.Sequential(
    nn.BatchNorm2d(3),
    nn.Dropout(0.5),
)

model.train()
print("training mode:", model.training)

model.eval()
print("eval mode:", model.training)
```

Expected output:

```text
training mode: True
eval mode: False
```

Mechanical meaning:

- BatchNorm and Dropout behave differently in train and eval modes
- training loop must call `model.train()`
- evaluation loop must call `model.eval()`

### 25.6 File Edit: `train.py`

Block grammar:

```text
set_seed:
    seed Python and torch RNGs

choose_device:
    resolve "auto" to cuda/cpu

make_optimizer:
    construct optimizer from config

train_one_epoch:
    one pass over train loader
    update weights
    return average train metrics

run_training:
    build system from config
    create run directory
    train for N epochs
    evaluate on validation set
    write metrics
    save best checkpoint
    write final plots
```

Action:

Create `src/modern_cnn/train.py`.

```python
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import (
    append_jsonl,
    copy_config,
    make_run_dir,
    plot_history,
    save_confusion_matrix_image,
    save_json,
)
from .checkpoint import save_checkpoint
from .config import Config
from .data import build_dataloaders
from .evaluate import evaluate
from .metrics import AverageMeter, accuracy, parameter_count
from .models import build_model_from_config


def set_seed(seed: int) -> None:
    # Python random controls any standard-library random choices.
    random.seed(seed)

    # torch manual seed controls CPU torch randomness.
    torch.manual_seed(seed)

    # If CUDA exists, seed all visible CUDA devices.
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.optim.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.optim.lr,
            momentum=0.9,
            weight_decay=cfg.optim.weight_decay,
        )

    if cfg.optim.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {cfg.optim.name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    log_interval: int = 50,
) -> dict[str, float]:
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        # Move the batch to the same device as the model.
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: logits shape [batch, classes].
        logits = model(images)

        # CrossEntropyLoss expects raw logits and integer class labels.
        loss = loss_fn(logits, labels)

        # Clear old gradients, compute new gradients, then update parameters.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        acc_meter.update(accuracy(logits.detach(), labels), batch_size)

        if log_interval > 0 and batch_idx % log_interval == 0:
            print(
                {
                    "batch": batch_idx,
                    "train_loss": loss_meter.average,
                    "train_accuracy": acc_meter.average,
                }
            )

    return {
        "loss": loss_meter.average,
        "accuracy": acc_meter.average,
    }


def run_training(cfg: Config, config_path: str | Path | None = None) -> dict[str, Any]:
    set_seed(cfg.experiment.seed)

    device = choose_device(cfg.experiment.device)
    loaders = build_dataloaders(cfg)
    model = build_model_from_config(cfg.model).to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = make_run_dir(cfg.experiment.output_dir, cfg.run_name)
    metrics_path = run_dir / "metrics.jsonl"

    # Save both resolved config and original config file if available.
    save_json(run_dir / "resolved_config.json", cfg.to_dict())
    if config_path is not None:
        copy_config(config_path, run_dir)

    summary = {
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "device": str(device),
        "parameter_count": parameter_count(model),
        "epochs": cfg.experiment.epochs,
    }
    save_json(run_dir / "summary_start.json", summary)
    print(summary)

    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.experiment.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders.train,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            log_interval=cfg.experiment.log_interval,
        )

        val_output = evaluate(
            model=model,
            loader=loaders.val,
            loss_fn=loss_fn,
            device=device,
            num_classes=cfg.data.num_classes,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_output.loss,
            "val_accuracy": val_output.accuracy,
        }
        append_jsonl(metrics_path, row)
        print(row)

        if val_output.accuracy > best_val_acc:
            best_val_acc = val_output.accuracy
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric=best_val_acc,
                config=cfg.to_dict(),
            )
            save_confusion_matrix_image(
                val_output.confusion,
                loaders.class_names,
                run_dir / "best_val_confusion.png",
            )

    plot_history(metrics_path, run_dir / "metrics.png")

    final_summary = {
        "run_dir": str(run_dir),
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count(model),
    }
    save_json(run_dir / "summary_final.json", final_summary)
    return final_summary
```

### 25.7 `train_one_epoch` Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch
from torch import nn

from modern_cnn.config import load_config
from modern_cnn.data import build_dataloaders
from modern_cnn.models import build_model_from_config
from modern_cnn.train import choose_device, make_optimizer, train_one_epoch

cfg = load_config("configs/smoke.toml")
device = choose_device(cfg.experiment.device)

loaders = build_dataloaders(cfg)
model = build_model_from_config(cfg.model).to(device)
optimizer = make_optimizer(model, cfg)

metrics = train_one_epoch(
    model=model,
    loader=loaders.train,
    optimizer=optimizer,
    loss_fn=nn.CrossEntropyLoss(),
    device=device,
    log_interval=0,
)

print(metrics)
```

Expected pattern:

```text
{'loss': ..., 'accuracy': ...}
```

Failure checks:

- device mismatch: make sure both images and labels are moved to device
- loss error: labels must be integer class indices, not one-hot vectors
- no parameter change: check `loss.backward()` and `optimizer.step()`

## 26. Phase 10: CLI

### 26.1 Goal

The CLI makes the system runnable outside a notebook.

This is production-adjacent because cloud training should look like local training:

```bash
modern-cnn train --config configs/smoke.toml
```

### 26.2 Exact Run Order

Use this order:

```text
1. Run argparse preflight.
2. Type cli.py.
3. Reinstall editable package if needed.
4. Run inspect-model.
5. Run smoke training.
6. Run evaluate after a checkpoint exists.
7. Run analyze-errors after a checkpoint exists.
```

### 26.3 Syntax Preflight: `argparse`

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args(["--config", "configs/smoke.toml"])

print(args.config)
```

Expected output:

```text
configs/smoke.toml
```

Mechanical meaning:

- CLI flags become attributes on `args`
- subcommands will attach a function to `args.func`

### 26.4 File Edit: `cli.py`

Block grammar:

```text
show-config:
    load TOML and print resolved config

inspect-model:
    build model
    print parameter count
    run shape trace

train:
    load config
    call run_training

evaluate:
    load checkpoint
    run test evaluation
    save test artifacts

analyze-errors:
    load checkpoint
    collect misclassified examples
    save image grid

compare:
    read multiple metrics files
    print best validation accuracy per run
```

Action:

Create `src/modern_cnn/cli.py`.

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .artifacts import read_jsonl, save_confusion_matrix_image, save_json, save_misclassified_grid
from .checkpoint import load_checkpoint
from .config import load_config
from .data import build_dataloaders
from .evaluate import class_accuracy_dict, collect_misclassified, evaluate
from .metrics import parameter_count
from .models import build_model_from_config, trace_shapes
from .train import choose_device, run_training


def _cmd_show_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(cfg.to_dict())


def _cmd_inspect_model(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model = build_model_from_config(cfg.model)

    x = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size)
    y = model(x)

    print("model:", cfg.model.name)
    print("parameters:", parameter_count(model))
    print("output_shape:", tuple(y.shape))

    for name, out_shape in trace_shapes(model, x[:1]):
        print(f"{name:45s} {out_shape}")


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    result = run_training(cfg, config_path=args.config)
    print(result)


def _load_model_for_eval(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    device = choose_device(cfg.experiment.device)
    model = build_model_from_config(cfg.model).to(device)

    payload = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])

    return cfg, device, model


def _cmd_evaluate(args: argparse.Namespace) -> None:
    cfg, device, model = _load_model_for_eval(args.config, args.checkpoint)
    loaders = build_dataloaders(cfg)
    output = evaluate(model, loaders.test, nn.CrossEntropyLoss(), device, cfg.data.num_classes)

    result = {
        "test_loss": output.loss,
        "test_accuracy": output.accuracy,
        "class_accuracy": class_accuracy_dict(output.confusion, loaders.class_names),
    }
    print(result)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        save_json(output_dir / "test_metrics.json", result)
        save_confusion_matrix_image(output.confusion, loaders.class_names, output_dir / "test_confusion.png")


def _cmd_analyze_errors(args: argparse.Namespace) -> None:
    cfg, device, model = _load_model_for_eval(args.config, args.checkpoint)
    loaders = build_dataloaders(cfg)

    examples = collect_misclassified(model, loaders.test, device, limit=args.limit)
    save_misclassified_grid(examples, loaders.class_names, args.output)
    print({"saved": args.output, "count": len(examples)})


def _cmd_compare(args: argparse.Namespace) -> None:
    for metrics_path in args.metrics:
        rows = read_jsonl(metrics_path)
        best = max(rows, key=lambda row: row["val_accuracy"])
        print(
            {
                "metrics": metrics_path,
                "best_epoch": best["epoch"],
                "best_val_accuracy": best["val_accuracy"],
                "best_val_loss": best["val_loss"],
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modern-cnn")
    sub = parser.add_subparsers(dest="command", required=True)

    show_config = sub.add_parser("show-config")
    show_config.add_argument("--config", required=True)
    show_config.set_defaults(func=_cmd_show_config)

    inspect_model = sub.add_parser("inspect-model")
    inspect_model.add_argument("--config", required=True)
    inspect_model.set_defaults(func=_cmd_inspect_model)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.set_defaults(func=_cmd_train)

    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument("--config", required=True)
    evaluate_cmd.add_argument("--checkpoint", required=True)
    evaluate_cmd.add_argument("--output-dir", default="")
    evaluate_cmd.set_defaults(func=_cmd_evaluate)

    analyze = sub.add_parser("analyze-errors")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--limit", type=int, default=16)
    analyze.set_defaults(func=_cmd_analyze_errors)

    compare = sub.add_parser("compare")
    compare.add_argument("metrics", nargs="+")
    compare.set_defaults(func=_cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
```

### 26.5 CLI Smoke Drills

Terminal: project root shell, not `experiments.ipynb`.

Action:

Run:

```bash
modern-cnn show-config --config configs/smoke.toml
modern-cnn inspect-model --config configs/smoke.toml
modern-cnn train --config configs/smoke.toml
```

Expected artifacts:

```text
artifacts/<timestamp>_smoke_convnext_tiny/
  config.toml
  resolved_config.json
  summary_start.json
  metrics.jsonl
  metrics.png
  best.pt
  best_val_confusion.png
  summary_final.json
```

Failure checks:

- `modern-cnn` missing: reinstall with `python -m pip install -e ".[dev]"`
- slow smoke run: confirm dataset is `fake_cifar10`
- no checkpoint: validation loop may have failed before best save

## 27. Phase 11: Reference Notes

### 27.1 Goal

Keep an explicit record of what ideas came from the reference architectures and what was intentionally scaled down.

This prevents a common learning mistake:

```text
"I implemented ConvNeXt"
```

when the accurate claim is:

```text
"I implemented a CIFAR-sized ConvNeXt-style CNN using selected ConvNeXt ideas"
```

### 27.2 File Edit: `reference_notes.py`

Action:

Create `src/modern_cnn/reference_notes.py`.

```python
from __future__ import annotations

from pathlib import Path


REFERENCE_NOTES = """# Reference Notes

## Primary References

- LeNet: early convolution, sigmoid activations, average pooling, dense classifier.
- AlexNet: ReLU, max pooling, deeper convolution stack, large dense head.
- VGG: repeated small convolution blocks and simple staged design.
- NiN: replacing large dense heads with more convolutional processing and global average pooling.
- GoogLeNet: multi-branch feature extraction and aggressive parameter efficiency.
- ResNet: residual connections for easier optimization of deeper networks.
- DenseNet: feature reuse through concatenated connections.
- ConvNeXt: modernized CNN blocks using depthwise convolution, inverted bottleneck-style channel mixing, GELU, LayerNorm, residual paths, and modern training recipes.

## What This Project Implements

- CIFAR-10 data pipeline.
- MLP baseline.
- LeNet adapted to RGB 32x32 images.
- AlexNet-small scaled down for CIFAR-10.
- ResNet-small with residual blocks.
- ConvNeXt-tiny style model for CIFAR-10.

## What This Project Does Not Claim

- It does not reproduce original ImageNet results.
- It does not reproduce full ConvNeXt training scale.
- It does not implement distributed training.
- It does not tune for state-of-the-art CIFAR-10 accuracy.

## Interpretation Rule

Compare architectures under the same local pipeline.
Do not compare these small runs directly against published large-scale results.
"""


def write_reference_notes(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(REFERENCE_NOTES, encoding="utf-8")
```

### 27.3 Reference Notes Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from modern_cnn.reference_notes import write_reference_notes

write_reference_notes("artifacts/reference_notes.md")
```

Expected:

```text
artifacts/reference_notes.md exists
```

## 28. Phase 12: Tests

### 28.1 Goal

Tests check system contracts.

They are not accuracy tests.

They prove:

- configs load
- data shape is correct
- models return logits of the right shape
- blocks preserve or change shape as intended
- one training step updates parameters
- checkpoints round trip

### 28.2 Exact Run Order

Use this order:

```text
1. Type test_config.py.
2. Type test_data.py.
3. Type test_shapes.py.
4. Type test_blocks.py.
5. Type test_train_step.py.
6. Type test_checkpoint.py.
7. Run pytest.
8. Fix contract failures before training.
```

### 28.3 File Edit: `tests/test_config.py`

```python
from modern_cnn.config import load_config


def test_smoke_config_loads():
    cfg = load_config("configs/smoke.toml")
    assert cfg.run_name == "smoke_convnext_tiny"
    assert cfg.data.batch_size == 16
    assert cfg.model.name == "convnext_tiny"
```

### 28.4 File Edit: `tests/test_data.py`

```python
from modern_cnn.config import load_config
from modern_cnn.data import build_dataloaders, split_indices


def test_split_indices_do_not_overlap():
    train_idx, val_idx = split_indices(n=20, val_fraction=0.25, seed=0)
    assert len(train_idx) == 15
    assert len(val_idx) == 5
    assert set(train_idx).isdisjoint(set(val_idx))


def test_fake_loader_shapes():
    cfg = load_config("configs/smoke.toml")
    loaders = build_dataloaders(cfg)
    images, labels = next(iter(loaders.train))
    assert tuple(images.shape) == (16, 3, 32, 32)
    assert tuple(labels.shape) == (16,)
```

### 28.5 File Edit: `tests/test_shapes.py`

```python
import torch

from modern_cnn.models import build_model


def test_all_models_return_class_logits():
    for name in ["mlp", "lenet", "alexnet_small", "resnet_small", "convnext_tiny"]:
        model = build_model(name, num_classes=10, width=16, depth=1)
        logits = model(torch.randn(2, 3, 32, 32))
        assert tuple(logits.shape) == (2, 10)
```

### 28.6 File Edit: `tests/test_blocks.py`

```python
import torch

from modern_cnn.blocks import ConvNeXtBlock, LayerNorm2d, ResidualBlock


def test_residual_projection_changes_shape():
    block = ResidualBlock(8, 16, stride=2)
    out = block(torch.randn(2, 8, 16, 16))
    assert tuple(out.shape) == (2, 16, 8, 8)


def test_convnext_block_preserves_shape():
    block = ConvNeXtBlock(16, kernel_size=7)
    out = block(torch.randn(2, 16, 8, 8))
    assert tuple(out.shape) == (2, 16, 8, 8)


def test_layernorm2d_preserves_nchw_shape():
    norm = LayerNorm2d(16)
    out = norm(torch.randn(2, 16, 8, 8))
    assert tuple(out.shape) == (2, 16, 8, 8)
```

### 28.7 File Edit: `tests/test_train_step.py`

```python
import torch
from torch import nn

from modern_cnn.models import build_model


def test_one_step_changes_parameters():
    model = build_model("lenet", num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])

    first_param = next(model.parameters())
    before = first_param.detach().clone()

    loss = loss_fn(model(images), labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    after = first_param.detach().clone()
    assert not torch.allclose(before, after)
```

### 28.8 File Edit: `tests/test_checkpoint.py`

```python
from modern_cnn.checkpoint import load_checkpoint, save_checkpoint
from modern_cnn.models import build_model


def test_checkpoint_round_trip(tmp_path):
    model = build_model("lenet", num_classes=10)
    path = tmp_path / "model.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=3,
        metric=0.5,
        config={"run_name": "test"},
    )

    payload = load_checkpoint(path)

    assert payload["epoch"] == 3
    assert payload["metric"] == 0.5
    assert payload["config"] == {"run_name": "test"}
    assert "model_state" in payload
```

### 28.9 Test Run

Terminal: project root shell, not `experiments.ipynb`.

Action:

Run:

```bash
pytest
```

Expected:

```text
all tests pass
```

Failure checks:

- shape test fails for LeNet: check `16 * 6 * 6`
- data test tries to download CIFAR-10: smoke config should use `fake_cifar10`
- train step does not change parameter: check loss/backward/step order

## 29. Phase 13: Notebook

### 29.1 Goal

The notebook should teach and inspect.

It should not be the only place where the system exists.

Notebook responsibilities:

- explain architectural theory
- run small drills
- inspect tensors and artifacts
- compare results
- write final reflections

Package responsibilities:

- data loading
- model definitions
- training
- evaluation
- checkpointing
- artifact writing

### 29.2 Exact Notebook Sections

Use this structure:

```text
1. Setup imports.
2. Theory checkpoints.
3. Data contract drills.
4. Convolution/pooling/block drills.
5. Model shape and parameter inspection.
6. Smoke training from Python or CLI.
7. Artifact inspection.
8. Experiment comparison.
9. Error analysis.
10. Final report.
```

### 29.3 Setup Cell

Notebook: `experiments.ipynb`.

```python
from pathlib import Path

import torch

from modern_cnn.artifacts import read_jsonl
from modern_cnn.config import load_config
from modern_cnn.data import build_dataloaders
from modern_cnn.metrics import parameter_count
from modern_cnn.models import build_model_from_config, trace_shapes

torch.manual_seed(0)

PROJECT_ROOT = Path.cwd()
print(PROJECT_ROOT)
```

### 29.4 Theory Checkpoint Markdown Cell

Notebook: `experiments.ipynb`.

Answer before running more code.

```text
1. Why should a CNN usually beat an MLP on image data?

2. What does max pooling preserve, and what does it discard?

3. Why can dense classifier heads become parameter-heavy?

4. Why does BatchNorm require separate train/eval behavior?

5. What problem does a residual path solve?

6. In a ConvNeXt-style block, why separate depthwise spatial mixing from pointwise channel mixing?
```

### 29.5 Data Contract Cell

Notebook: `experiments.ipynb`.

```python
cfg = load_config("configs/smoke.toml")
loaders = build_dataloaders(cfg)

images, labels = next(iter(loaders.train))

print("images:", tuple(images.shape))
print("labels:", tuple(labels.shape))
print("image min/max:", float(images.min()), float(images.max()))
print("label sample:", labels[:10])

assert tuple(images.shape) == (cfg.data.batch_size, 3, 32, 32)
assert tuple(labels.shape) == (cfg.data.batch_size,)
```

### 29.6 Conv And Pool Drill Cell

Notebook: `experiments.ipynb`.

```python
from torch import nn

x = torch.randn(2, 3, 32, 32)

conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

after_conv = conv(x)
after_pool = pool(after_conv)

print("input:", tuple(x.shape))
print("after conv:", tuple(after_conv.shape))
print("after pool:", tuple(after_pool.shape))
```

Expected:

```text
input: (2, 3, 32, 32)
after conv: (2, 16, 32, 32)
after pool: (2, 16, 16, 16)
```

### 29.7 Model Inspection Cell

Notebook: `experiments.ipynb`.

```python
for config_path in [
    "configs/cifar10_mlp.toml",
    "configs/cifar10_lenet.toml",
    "configs/cifar10_alexnet_small.toml",
    "configs/cifar10_resnet_small.toml",
    "configs/cifar10_convnext_tiny.toml",
]:
    cfg = load_config(config_path)
    model = build_model_from_config(cfg.model)
    logits = model(torch.randn(2, 3, 32, 32))

    print(config_path)
    print("  output:", tuple(logits.shape))
    print("  params:", parameter_count(model))
```

### 29.8 Shape Trace Cell

Notebook: `experiments.ipynb`.

```python
cfg = load_config("configs/smoke.toml")
model = build_model_from_config(cfg.model)

for name, out_shape in trace_shapes(model, torch.randn(1, 3, 32, 32)):
    print(f"{name:45s} -> {out_shape}")
```

Interpretation prompt:

```text
Where does spatial size shrink?
Where does channel count increase?
Where does the model become a vector classifier?
```

### 29.9 Smoke Training Cell

Notebook: `experiments.ipynb`.

Option A:

```python
from modern_cnn.train import run_training

cfg = load_config("configs/smoke.toml")
result = run_training(cfg, config_path="configs/smoke.toml")
result
```

Option B:

```python
!modern-cnn train --config configs/smoke.toml
```

Interpretation:

- smoke metrics do not prove learning
- smoke run proves the system wiring works
- fake labels/images are not meaningful accuracy data

### 29.10 Artifact Inspection Cell

Notebook: `experiments.ipynb`.

```python
run_dir = Path("artifacts/YOUR_RUN_DIR")
metrics = read_jsonl(run_dir / "metrics.jsonl")

metrics
```

Then:

```python
import matplotlib.pyplot as plt

epochs = [row["epoch"] for row in metrics]
val_acc = [row["val_accuracy"] for row in metrics]

plt.plot(epochs, val_acc)
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
plt.show()
```

### 29.11 Experiment Comparison Cell

Notebook: `experiments.ipynb`.

After real runs:

```python
from pathlib import Path

run_dirs = sorted(Path("artifacts").glob("*cifar10*"))

rows = []
for run_dir in run_dirs:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        continue

    metrics = read_jsonl(metrics_path)
    best = max(metrics, key=lambda row: row["val_accuracy"])
    rows.append((run_dir.name, best["epoch"], best["val_accuracy"], best["val_loss"]))

for row in rows:
    print(row)
```

### 29.12 Error Analysis Prompt

Notebook: `experiments.ipynb`.

After `modern-cnn analyze-errors` creates a grid:

```text
1. Which true classes were confused most often?
2. Are the mistakes visually ambiguous?
3. Does the model confuse semantically similar classes?
4. Do errors suggest underfitting, overfitting, or data augmentation issues?
```

## 30. Phase 14: Local Run Order

### 30.1 Smoke Run

Terminal: project root shell, not `experiments.ipynb`.

Action:

Run:

```bash
python -m pip install -e ".[dev]"
pytest
modern-cnn show-config --config configs/smoke.toml
modern-cnn inspect-model --config configs/smoke.toml
modern-cnn train --config configs/smoke.toml
```

Expected:

```text
tests pass
model output shape is (2, 10)
smoke run writes artifacts
```

Do not move to CIFAR-10 until this works.

### 30.2 Real CIFAR-10 Runs

Terminal: project root shell, not `experiments.ipynb`.

Action:

Run:

```bash
modern-cnn train --config configs/cifar10_mlp.toml
modern-cnn train --config configs/cifar10_lenet.toml
modern-cnn train --config configs/cifar10_alexnet_small.toml
modern-cnn train --config configs/cifar10_resnet_small.toml
modern-cnn train --config configs/cifar10_convnext_tiny.toml
```

Interpretation order:

```text
MLP:
    what happens without image bias?

LeNet:
    how far does an old small CNN get?

AlexNet-small:
    what does ReLU + deeper conv + max pooling buy?

ResNet-small:
    do residual paths improve training?

ConvNeXt-tiny:
    does the modern block improve the tradeoff?
```

### 30.3 Ablation Runs

Terminal: project root shell, not `experiments.ipynb`.

Action:

Run:

```bash
modern-cnn train --config configs/ablation_convnext_no_residual.toml
modern-cnn train --config configs/ablation_convnext_small_kernel.toml
modern-cnn train --config configs/ablation_convnext_no_augmentation.toml
```

Each ablation should change one idea.

If two ideas change at the same time, interpretation becomes weaker.

## 31. Phase 15: Cloud Training Plan

### 31.1 Goal

Use cloud only after the local system is proven.

Cloud should accelerate training, not hide broken code.

### 31.2 Compute Requirements

Local CPU is enough for:

- package import checks
- tests
- shape tracing
- fake-data smoke training

GPU is useful for:

- full CIFAR-10 runs
- ablation matrix
- repeated experiments

Expected resources:

```text
VRAM: 8 GB is enough
RAM: 8-16 GB is comfortable
disk: less than 5 GB
batch size: start with 128
```

Reasonable GPUs:

```text
NVIDIA T4
NVIDIA L4
RTX 3060
RTX 4060
RTX 4070
```

Do not rent A100/H100 class GPUs for this project unless you intentionally scale the dataset/model later.

### 31.3 Cloud Run Checklist

Terminal: cloud project root shell, not `experiments.ipynb`.

Before cloud:

```text
pytest passes locally
smoke train passes locally
configs are committed or copied exactly
requirements install cleanly
```

On cloud:

```bash
python -m pip install -e ".[dev]"
pytest
modern-cnn train --config configs/cifar10_convnext_tiny.toml
```

After cloud:

```text
copy artifacts back
inspect metrics locally
write final report locally
```

Mechanical principle:

```text
same CLI command shape locally and in cloud
```

## 32. Phase 16: Final Report

### 32.1 Goal

The final report should explain results, not just paste scores.

A rigorous report ties observations to mechanisms:

```text
accuracy changed
because an architectural/training choice changed
and the artifacts support that interpretation
```

### 32.2 Report Template

Create `REPORT.md` or write this in the notebook.

```markdown
# Project 1 Report

## Question

Can a ConvNeXt-style CNN outperform older CNN baselines on the same CIFAR-10 pipeline?

## Setup

- dataset:
- train/validation/test split:
- hardware:
- epochs:
- optimizer:
- batch size:

## Model Contracts

| model | input shape | output shape | parameters |
|---|---|---|---:|
| MLP | [batch, 3, 32, 32] | [batch, 10] | |
| LeNet-CIFAR | [batch, 3, 32, 32] | [batch, 10] | |
| AlexNet-small | [batch, 3, 32, 32] | [batch, 10] | |
| ResNet-small | [batch, 3, 32, 32] | [batch, 10] | |
| ConvNeXt-tiny | [batch, 3, 32, 32] | [batch, 10] | |

## Results

| model | best epoch | best validation accuracy | test accuracy |
|---|---:|---:|---:|

## Architecture Interpretation

### MLP

What happened without image-specific locality?

### LeNet-CIFAR

What did the early CNN pipeline capture?

### AlexNet-small

What changed with ReLU, max pooling, and a deeper conv stack?

### ResNet-small

Did residual paths improve optimization?

### ConvNeXt-tiny

Did depthwise spatial mixing and modern block design help?

## Ablations

| ablation | changed idea | observed result | interpretation |
|---|---|---|---|
| no residual | remove identity path | | |
| small kernel | 7x7 -> 3x3 depthwise kernel | | |
| no augmentation | remove crop/flip | | |

## Error Analysis

- most confused classes:
- examples inspected:
- likely causes:

## What I Understand Now

- convolution:
- pooling/downsampling:
- dense heads:
- BatchNorm:
- residual connections:
- depthwise convolution:
- global average pooling:
- production-shaped training:
```

## 33. README

### 33.1 Goal

The README is the operational entry point.

It should tell a future reader how to install, test, train, and inspect.

### 33.2 File Edit: `README.md`

Action:

Create `README.md`.

````markdown
# Project 1: Modern CNN Training System

This project trains and compares CIFAR-10 CNN models using a production-shaped PyTorch package.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Smoke Run

```bash
modern-cnn inspect-model --config configs/smoke.toml
modern-cnn train --config configs/smoke.toml
```

## CIFAR-10 Runs

```bash
modern-cnn train --config configs/cifar10_mlp.toml
modern-cnn train --config configs/cifar10_lenet.toml
modern-cnn train --config configs/cifar10_alexnet_small.toml
modern-cnn train --config configs/cifar10_resnet_small.toml
modern-cnn train --config configs/cifar10_convnext_tiny.toml
```

## Evaluate

```bash
modern-cnn evaluate \
  --config configs/cifar10_convnext_tiny.toml \
  --checkpoint artifacts/YOUR_RUN_DIR/best.pt \
  --output-dir artifacts/YOUR_RUN_DIR/test_eval
```

## Analyze Errors

```bash
modern-cnn analyze-errors \
  --config configs/cifar10_convnext_tiny.toml \
  --checkpoint artifacts/YOUR_RUN_DIR/best.pt \
  --output artifacts/YOUR_RUN_DIR/misclassified.png
```

## Compare Runs

```bash
modern-cnn compare artifacts/*/metrics.jsonl
```
````

## 34. Done Criteria

The project is done when:

- `pytest` passes
- smoke training runs from the CLI
- every model produces `[batch, 10]` logits
- metrics are saved to JSONL
- best checkpoint is saved
- metric plot is saved
- confusion matrix is saved
- at least one misclassification grid is saved
- notebook imports package code instead of redefining it
- MLP, LeNet, AlexNet-small, ResNet-small, and ConvNeXt-tiny are compared
- at least three ablations are run
- final report explains results using the architectural theory

## 35. Final Principle

For every component, ask:

```text
what problem does it solve?
what tensor contract does it impose?
what does it cost?
what failure does it prevent?
what test or artifact proves it works?
```

That is the standard for this project.
