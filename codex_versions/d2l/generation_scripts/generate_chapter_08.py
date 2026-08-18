"""Generate Chapter 8 notebooks for the D2L rewrite project.

Chapter 8 keeps the Chapter 6-7 notebook-native style: architecture mechanics,
shape tracing, parameter contracts, deliberate breakage demos, and offline
synthetic tensors. Full dataset training is intentionally deferred.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Chapter 8 - Modern Convolutional Neural Networks"
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

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

def shape(x):
    return tuple(x.shape)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

def trace_module_shapes(module, X):
    rows = []
    current = X
    for name, layer in module.named_children():
        current = layer(current)
        rows.append((name, layer.__class__.__name__, shape(current)))
    return rows, current
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

Run the notebook from top to bottom in a clean kernel. The code uses small synthetic tensors so that architecture mechanics can be inspected without downloads, `torchvision`, ImageNet-scale images, or long training runs. Before important cells, predict the shape, parameter count, or failure mode, then read the assertions as executable contracts.

## You are done when you can

{outcome_lines}
"""
    )


def checkpoint(title: str, questions: list[str]) -> dict:
    lines = "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, start=1))
    return md(
        f"""## {title} Checkpoint

Answer these before moving on. Short markdown answers in the notebook are enough; the checkpoint is meant to test whether you can explain the mechanics without rereading the code.

{lines}
"""
    )


def build_81() -> None:
    cells = [
        title_cell(
            "Chapter 8.1 - Deep Convolutional Neural Networks (AlexNet)",
            "Chapter 7 ended with LeNet: a compact CNN that turns image grids into class logits. Chapter 8 starts the modern CNN tour. AlexNet matters because it showed that a deeper CNN trained at large scale could learn useful visual representations directly from data instead of relying on hand-engineered features.",
            [
                "explain representation learning in the AlexNet story",
                "trace how an AlexNet-style stack reduces spatial resolution and expands channels",
                "explain why ReLU helped deeper networks compared with saturating activations",
                "explain why dropout appears in the dense classifier head",
                "debug a fixed-flatten-size failure when the input resolution changes",
            ],
        ),
        setup_cell(
            """def conv2d_hw(height, width, kernel_size, stride=1, padding=0):
    h = math.floor((height + 2 * padding - kernel_size) / stride) + 1
    w = math.floor((width + 2 * padding - kernel_size) / stride) + 1
    return h, w

def pool2d_hw(height, width, kernel_size, stride):
    return conv2d_hw(height, width, kernel_size, stride, padding=0)"""
        ),
        md(
            """## 8.1.0 The Problem This Notebook Solves

LeNet already used convolution, nonlinear activation, pooling, flattening, and dense classification. AlexNet scales that idea in a historically important way:

- larger early receptive fields
- more channels
- more convolutional layers
- ReLU activations instead of saturating sigmoids
- dropout in the dense classifier head
- large-scale supervised training

Representation learning means the model learns intermediate features from data. Earlier computer vision systems often depended heavily on features written by humans, such as edge, texture, or shape descriptors. AlexNet-style CNNs learn many of those useful intermediate detectors as trainable parameters.

This notebook does not reproduce AlexNet's ImageNet training. That would require data, hardware, and time outside this chapter's purpose. The goal here is to make the architecture mechanically readable.
"""
        ),
        md(
            """## 8.1.1 Spatial Size Shrinks While Channels Grow

Modern CNNs often follow a repeated pattern:

```text
spatial resolution goes down
channel count goes up
semantic richness goes up
```

Spatial resolution means height and width. Channel count means the number of feature maps at each spatial location. Early layers preserve more local detail; later layers store more abstract feature evidence across fewer locations.

Before running the cell, predict:

- The first large-stride convolution should reduce 96 by 96 sharply.
- Pooling should reduce spatial size again.
- Later convolutions with padding should preserve spatial size inside the stack.
"""
        ),
        code(
            """height, width = 96, 96
steps = [
    ("conv11 stride4 pad2", lambda h, w: conv2d_hw(h, w, 11, stride=4, padding=2)),
    ("pool3 stride2", lambda h, w: pool2d_hw(h, w, 3, stride=2)),
    ("conv5 pad2", lambda h, w: conv2d_hw(h, w, 5, padding=2)),
    ("pool3 stride2", lambda h, w: pool2d_hw(h, w, 3, stride=2)),
    ("conv3 pad1", lambda h, w: conv2d_hw(h, w, 3, padding=1)),
    ("conv3 pad1", lambda h, w: conv2d_hw(h, w, 3, padding=1)),
    ("conv3 pad1", lambda h, w: conv2d_hw(h, w, 3, padding=1)),
    ("pool3 stride2", lambda h, w: pool2d_hw(h, w, 3, stride=2)),
]

trace = []
for name, update in steps:
    height, width = update(height, width)
    trace.append((name, height, width))

print(trace)
assert trace[0][1:] == (23, 23)
assert trace[-1][1:] == (2, 2)"""
        ),
        md(
            """## 8.1.2 A Small AlexNet-Style Network

The real AlexNet used much larger channel counts and dense layers. This version keeps the same architectural pattern while reducing width so that the forward pass is quick:

```text
large early convolution -> pooling -> deeper conv stack -> pooling -> dense head
```

The dense head has a fixed input feature contract. For 96 by 96 inputs, the convolutional feature extractor below produces `(batch, 16, 2, 2)`, so flattening gives 64 features per example.

Before running the cell, predict:

- The output logits should have shape `(2, 10)`.
- The flattened size before the first linear layer should be 64.
- The model has trainable parameters in both the convolutional feature extractor and dense head.
"""
        ),
        code(
            """alexnet_small = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=11, stride=4, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(8, 16, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(16, 24, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(24, 16, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 2 * 2, 32), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(32, 32), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(32, 10),
)

X = torch.randn(2, 1, 96, 96)
rows, logits = trace_module_shapes(alexnet_small, X)
for row in rows:
    print(row)

assert shape(logits) == (2, 10)
assert count_parameters(alexnet_small) > 0"""
        ),
        md(
            """## 8.1.3 ReLU Keeps a Stronger Gradient in the Positive Region

A saturating activation is an activation whose derivative becomes tiny over a wide input range. Sigmoid is useful in some places, but deep stacks of sigmoids can make gradient flow weak when activations saturate near 0 or 1.

ReLU is simple:

```text
relu(x) = max(0, x)
```

For positive inputs, its derivative is 1. That does not solve every optimization problem, but it makes deep networks easier to train than if every layer repeatedly squeezed values into a saturated range.

The cell compares gradients through a tiny activation-only computation. This is not a full training proof. It isolates one mechanical difference.
"""
        ),
        code(
            """values = torch.tensor([-6.0, -1.0, 0.0, 1.0, 6.0], requires_grad=True)
sigmoid_loss = torch.sigmoid(values).sum()
sigmoid_loss.backward()
sigmoid_grads = values.grad.clone()

values.grad.zero_()
relu_loss = F.relu(values).sum()
relu_loss.backward()
relu_grads = values.grad.clone()

print("sigmoid grads:", sigmoid_grads)
print("relu grads:", relu_grads)

assert relu_grads[-1].item() == 1.0
assert sigmoid_grads[-1].item() < 0.01"""
        ),
        md(
            """## 8.1.4 Dropout Changes Training Behavior, Not Evaluation Behavior

Dropout randomly zeros some activations during training. The purpose is regularization: it makes the dense classifier head less able to rely on one brittle co-adaptation of hidden units.

Two practical rules matter:

- `model.train()` enables dropout randomness.
- `model.eval()` disables dropout randomness and uses the full representation.

This is a software-state issue, not just a mathematical layer. A model that accidentally stays in training mode during inference can produce unstable predictions.
"""
        ),
        code(
            """drop = nn.Dropout(p=0.5)
X = torch.ones(12)

torch.manual_seed(1)
drop.train()
Y_train = drop(X)

drop.eval()
Y_eval = drop(X)

print("training output:", Y_train)
print("eval output:", Y_eval)

assert (Y_train == 0).any()
assert torch.equal(Y_eval, X)"""
        ),
        md(
            """## 8.1.5 Break It Deliberately: Fixed Flatten Size

The dense classifier sees a vector, not an image. If the convolutional stack produces a different spatial size, the flattened vector length changes. A fixed `Linear(in_features, out_features)` layer will then reject the input.

This is one of the most common CNN architecture mistakes:

```text
changed input resolution
changed convolution/pooling output size
forgot to update first dense layer
```

The cell intentionally feeds the small AlexNet a different input resolution and catches the failure so the notebook can continue.
"""
        ),
        code(
            """bad_input = torch.randn(2, 1, 80, 80)

try:
    alexnet_small(bad_input)
except RuntimeError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected the dense layer to reject the changed flatten size")"""
        ),
        checkpoint(
            "8.1",
            [
                "What does representation learning mean in the AlexNet story?",
                "Why do modern CNNs often reduce spatial size while increasing channel count?",
                "Why did ReLU help deeper CNNs compared with saturating activations?",
                "What does dropout do differently in training and evaluation modes?",
                "Why can changing image resolution break a fixed dense classifier head?",
            ],
        ),
    ]
    write_nb("Chapter 8.1 - Deep Convolutional Neural Networks (AlexNet).ipynb", cells)


def build_82() -> None:
    cells = [
        title_cell(
            "Chapter 8.2 - Networks Using Blocks (VGG)",
            "VGG turns CNN architecture into a block design discipline. Instead of treating every layer as a one-off decision, it repeatedly stacks small 3 by 3 convolutions followed by pooling. That makes the architecture easier to read, modify, and scale.",
            [
                "define a VGG block and explain why it is reusable",
                "compare stacked 3 by 3 convolutions with a larger single convolution",
                "build a small VGG-style classifier from an architecture configuration",
                "trace spatial downsampling through repeated blocks",
                "debug the effect of forgetting padding in a VGG block",
            ],
        ),
        setup_cell(),
        md(
            """## 8.2.0 The Problem This Notebook Solves

VGG's central lesson is not only a particular network. It is a way of designing networks:

```text
choose a simple block
repeat the block
increase channels across stages
reduce spatial resolution between stages
```

A block is a reusable module pattern. Chapter 6 introduced modules as software objects. VGG shows why this matters architecturally: repeated blocks let you express a deep model with a small, inspectable design vocabulary.
"""
        ),
        md(
            """## 8.2.1 A VGG Block Is Repeated Local Processing Plus Downsampling

A typical VGG block contains:

- one or more 3 by 3 convolution layers with padding 1
- ReLU after each convolution
- 2 by 2 max pooling with stride 2

Padding 1 is important. A 3 by 3 convolution with padding 1 preserves height and width before pooling. Then pooling halves the spatial size. This makes the block's shape contract predictable.

Before running the cell, predict:

- Input shape: `(2, 3, 32, 32)`.
- Two padded convolutions should keep `32 by 32`.
- Pooling should produce `16 by 16`.
- Output channel count should be 8.
"""
        ),
        code(
            """def vgg_block(in_channels, out_channels, num_convs):
    layers = []
    current_channels = in_channels
    for _ in range(num_convs):
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        current_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


block = vgg_block(3, 8, num_convs=2)
X = torch.randn(2, 3, 32, 32)
Y = block(X)

print("output shape:", shape(Y))
assert shape(Y) == (2, 8, 16, 16)"""
        ),
        md(
            """## 8.2.2 Two 3 by 3 Convolutions See a 5 by 5 Neighborhood

Stacking small kernels increases the effective receptive field. Receptive field means the region of the original input that can affect one output value.

Two stride-1 3 by 3 convolutions let a later output depend on a 5 by 5 region, but with an extra nonlinearity between the two convolutions. For equal input and output channel width, two 3 by 3 convolutions often use fewer parameters than one 5 by 5 convolution:

```text
two 3 by 3 layers: 2 * 3 * 3 * C * C
one 5 by 5 layer: 5 * 5 * C * C
```

This is a design tradeoff: VGG prefers a regular stack of small local operations.
"""
        ),
        code(
            """channels = 16
two_3x3 = 2 * 3 * 3 * channels * channels
one_5x5 = 5 * 5 * channels * channels

print("two 3x3 weights:", two_3x3)
print("one 5x5 weights:", one_5x5)
print("saving:", one_5x5 - two_3x3)

assert two_3x3 < one_5x5"""
        ),
        md(
            """## 8.2.3 Build VGG From an Architecture Configuration

An architecture configuration is a compact description of repeated stages. In this notebook, each tuple means:

```text
(number of convolutions in the block, output channels)
```

The builder below converts that list into an executable `nn.Sequential`. This is the bridge from design vocabulary to framework code.

Before running the cell, predict:

- Three blocks with pooling should reduce 64 to 32 to 16 to 8.
- The final adaptive average pool should make the dense head independent of the exact final spatial size.
- The logits should have shape `(2, 10)`.
"""
        ),
        code(
            """def make_vgg(in_channels, arch, num_classes=10):
    layers = []
    current_channels = in_channels
    for num_convs, out_channels in arch:
        layers.append(vgg_block(current_channels, out_channels, num_convs))
        current_channels = out_channels
    layers += [
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(current_channels, num_classes),
    ]
    return nn.Sequential(*layers)


tiny_vgg = make_vgg(1, [(1, 8), (1, 16), (2, 32)])
X = torch.randn(2, 1, 64, 64)
rows, logits = trace_module_shapes(tiny_vgg, X)
for row in rows:
    print(row)

assert shape(logits) == (2, 10)"""
        ),
        md(
            """## 8.2.4 Blocks Make Parameter Accounting Local

When a network is built from blocks, you can inspect each block separately. This matters because modern CNNs are not just long lists of layers; they are systems of repeated components.

The cell counts trainable scalars per top-level stage. This is not the same as measuring runtime speed, but it reveals where model capacity lives.
"""
        ),
        code(
            """for name, layer in tiny_vgg.named_children():
    print(name, layer.__class__.__name__, count_parameters(layer))

total = count_parameters(tiny_vgg)
print("total trainable scalars:", total)

assert total == sum(count_parameters(layer) for layer in tiny_vgg.children())"""
        ),
        md(
            """## 8.2.5 Break It Deliberately: Forget Padding

The VGG block pattern depends on padded 3 by 3 convolutions preserving spatial size before pooling. If you forget padding, every convolution shrinks height and width before the pooling layer runs.

The forward pass may still run, which makes this bug subtle. The architecture no longer has the shape story you intended.
"""
        ),
        code(
            """bad_block = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3), nn.ReLU(),
    nn.Conv2d(8, 8, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
)

good_Y = block(torch.randn(2, 3, 32, 32))
bad_Y = bad_block(torch.randn(2, 3, 32, 32))

print("good block:", shape(good_Y))
print("bad block:", shape(bad_Y))

try:
    assert shape(bad_Y)[-2:] == (16, 16)
except AssertionError:
    print("The block ran, but it violated the intended VGG shape contract.")"""
        ),
        checkpoint(
            "8.2",
            [
                "What is a VGG block?",
                "Why can repeated 3 by 3 convolutions be preferable to one larger convolution?",
                "Why does padding matter inside a VGG block?",
                "What does an architecture configuration buy you as a programmer?",
                "Why is a running forward pass not enough to prove the architecture is correct?",
            ],
        ),
    ]
    write_nb("Chapter 8.2 - Networks Using Blocks (VGG).ipynb", cells)


def build_83() -> None:
    cells = [
        title_cell(
            "Chapter 8.3 - Network in Network (NiN)",
            "NiN changes how we think about convolutional blocks. Instead of applying one linear filter to each local patch, a NiN block applies a small network at each location, using 1 by 1 convolutions to mix channels and add nonlinear processing.",
            [
                "explain why a 1 by 1 convolution is a per-location linear layer",
                "build a NiN block from spatial convolution and 1 by 1 convolutions",
                "explain why global average pooling can replace a large dense head",
                "trace logits from channels through global average pooling",
                "debug a class-count mismatch at the classifier output",
            ],
        ),
        setup_cell(),
        md(
            """## 8.3.0 The Problem This Notebook Solves

Chapter 7.4 showed that a 1 by 1 convolution mixes channels at each spatial location. NiN turns that idea into an architecture pattern:

```text
spatial convolution finds local evidence
1 by 1 convolution mixes feature channels at each location
another 1 by 1 convolution adds more local nonlinear processing
```

The name "Network in Network" points to the idea that each local patch is processed by a small neural network rather than a single linear filter.
"""
        ),
        md(
            """## 8.3.1 A 1 by 1 Convolution Is a Linear Layer Shared Across Locations

At one pixel location, the input is a channel vector. A 1 by 1 convolution applies the same linear transformation to that vector at every row and column.

The cell proves this by comparing `nn.Conv2d(..., kernel_size=1)` with `F.linear` applied to every pixel's channel vector.
"""
        ),
        code(
            """conv1x1 = nn.Conv2d(3, 4, kernel_size=1, bias=True)
X = torch.randn(2, 3, 5, 5)

Y_conv = conv1x1(X)
X_pixels = X.permute(0, 2, 3, 1)
Y_linear_pixels = F.linear(X_pixels, conv1x1.weight.squeeze(-1).squeeze(-1), conv1x1.bias)
Y_linear = Y_linear_pixels.permute(0, 3, 1, 2)

print("conv output:", shape(Y_conv))
print("linear rebuilt output:", shape(Y_linear))

assert torch.allclose(Y_conv, Y_linear, atol=1e-6)"""
        ),
        md(
            """## 8.3.2 A NiN Block Adds Local Nonlinear Channel Mixing

A NiN block begins with a normal spatial convolution. Then it uses two 1 by 1 convolutions. ReLU after each convolution makes the local computation nonlinear.

The important contract:

```text
the spatial convolution can change height and width
the 1 by 1 convolutions keep height and width
the output channel count is the block's chosen width
```

Before running the cell, predict:

- Input shape: `(2, 3, 16, 16)`.
- A 5 by 5 convolution with padding 2 should keep 16 by 16.
- The block should output 8 channels.
"""
        ),
        code(
            """def nin_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )


block = nin_block(3, 8, kernel_size=5, padding=2)
Y = block(torch.randn(2, 3, 16, 16))

print("NiN block output:", shape(Y))
assert shape(Y) == (2, 8, 16, 16)"""
        ),
        md(
            """## 8.3.3 Global Average Pooling Turns Class Channels Into Logits

Classic CNNs often flatten a spatial feature map and feed it to dense layers. NiN uses a different classifier head:

```text
make the final channel count equal the class count
average each class channel over all spatial positions
use the resulting values as logits
```

Global average pooling reduces `(batch, classes, height, width)` to `(batch, classes, 1, 1)`. After flattening, the result is `(batch, classes)`.
"""
        ),
        code(
            """nin_small = nn.Sequential(
    nin_block(1, 8, kernel_size=5, padding=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nin_block(8, 16, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 10, kernel_size=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
)

X = torch.randn(4, 1, 32, 32)
rows, logits = trace_module_shapes(nin_small, X)
for row in rows:
    print(row)

assert shape(logits) == (4, 10)"""
        ),
        md(
            """## 8.3.4 Global Average Pooling Reduces Dense-Head Parameters

Flattening a large spatial map into a dense layer can create many parameters. Global average pooling uses a stronger architectural assumption:

```text
for classification, the final evidence for each class can be averaged over location
```

That discards exact final-map position, but it greatly reduces the classifier head.
"""
        ),
        code(
            """batch, channels, height, width = 4, 10, 8, 8
dense_head_weights = channels * height * width * 10
gap_head_weights = 0

features = torch.randn(batch, channels, height, width)
gap_logits = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

print("dense head weights for 10 logits:", dense_head_weights)
print("GAP head weights when channels already equal classes:", gap_head_weights)
print("GAP logits shape:", shape(gap_logits))

assert shape(gap_logits) == (batch, channels)
assert gap_head_weights < dense_head_weights"""
        ),
        md(
            """## 8.3.5 Break It Deliberately: Final Channel Count Is Class Count

With the NiN classifier pattern, the final channel count becomes the number of logits. If the task has 10 classes but the model outputs 7 channels, the loss cannot interpret a target label such as 9.

This failure is a classifier-head contract error, not a convolution error.
"""
        ),
        code(
            """bad_logits = torch.randn(3, 7)
targets = torch.tensor([0, 3, 9])
loss_fn = nn.CrossEntropyLoss()

try:
    loss_fn(bad_logits, targets)
except IndexError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected class target 9 to be invalid for 7 logits")"""
        ),
        checkpoint(
            "8.3",
            [
                "Why is a 1 by 1 convolution equivalent to a shared linear layer over channel vectors?",
                "What does a NiN block add beyond a single spatial convolution?",
                "How does global average pooling convert class feature maps into logits?",
                "What representation information does global average pooling discard?",
                "Why must the final channel count match the number of classes in a NiN-style head?",
            ],
        ),
    ]
    write_nb("Chapter 8.3 - Network in Network (NiN).ipynb", cells)


def build_84() -> None:
    cells = [
        title_cell(
            "Chapter 8.4 - Multi-Branch Networks (GoogLeNet)",
            "GoogLeNet uses Inception blocks: several branches process the same input at different receptive-field scales, then concatenate their output channels. The architecture lesson is that a block can be a small directed computation graph, not only a straight sequence.",
            [
                "explain why an Inception block has multiple branches",
                "trace branch shapes and channel concatenation",
                "explain how 1 by 1 bottlenecks reduce computation before larger kernels",
                "build a tiny GoogLeNet-style classifier",
                "debug branch concatenation when spatial sizes do not match",
            ],
        ),
        setup_cell(),
        md(
            """## 8.4.0 The Problem This Notebook Solves

VGG repeats one simple path. GoogLeNet asks a different question:

```text
what if useful features need different receptive-field sizes at the same network depth?
```

An Inception block answers with branches:

- a 1 by 1 branch
- a 1 by 1 then 3 by 3 branch
- a 1 by 1 then 5 by 5 branch
- a pooling then 1 by 1 branch

The branch outputs are concatenated along the channel dimension. Concatenation means the next layer receives all branch feature maps as one wider tensor.
"""
        ),
        md(
            """## 8.4.1 An Inception Block Concatenates Channel Evidence

Each branch must output the same height and width. The channel counts can differ because concatenation happens along the channel axis.

Before running the cell, predict:

- Branch spatial shapes should all be `16 by 16`.
- Output channels should be `4 + 6 + 6 + 4 = 20`.
- The output tensor should have shape `(2, 20, 16, 16)`.
"""
        ),
        code(
            """class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(),
                                nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(),
                                nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.Conv2d(in_channels, c4, kernel_size=1), nn.ReLU())

    def forward(self, X):
        branches = [self.b1(X), self.b2(X), self.b3(X), self.b4(X)]
        self.last_branch_shapes = [shape(branch) for branch in branches]
        return torch.cat(branches, dim=1)


block = Inception(3, c1=4, c2=(4, 6), c3=(4, 6), c4=4)
Y = block(torch.randn(2, 3, 16, 16))

print("branch shapes:", block.last_branch_shapes)
print("output:", shape(Y))
assert shape(Y) == (2, 20, 16, 16)"""
        ),
        md(
            """## 8.4.2 Bottleneck 1 by 1 Convolutions Reduce Large-Kernel Cost

A 5 by 5 convolution from many input channels to many output channels is expensive. Inception often inserts a 1 by 1 convolution first to reduce channel width:

```text
many channels -> fewer channels -> 5 by 5 convolution
```

This is called a bottleneck because information passes through a narrower channel dimension before the expensive operation.
"""
        ),
        code(
            """in_channels = 64
reduced_channels = 16
out_channels = 32

direct_5x5 = in_channels * out_channels * 5 * 5
bottleneck_then_5x5 = in_channels * reduced_channels * 1 * 1
bottleneck_then_5x5 += reduced_channels * out_channels * 5 * 5

print("direct 5x5 weights:", direct_5x5)
print("1x1 bottleneck then 5x5 weights:", bottleneck_then_5x5)

assert bottleneck_then_5x5 < direct_5x5"""
        ),
        md(
            """## 8.4.3 Build a Tiny GoogLeNet-Style Classifier

The full GoogLeNet has many Inception blocks. This notebook uses a tiny version that preserves the design contract:

```text
stem -> Inception block -> pooling -> Inception block -> global average pool -> logits
```

The first Inception block receives 8 channels and outputs 16. After pooling, the second receives 16 and outputs 32.
"""
        ),
        code(
            """googlenet_tiny = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
    Inception(8, c1=4, c2=(4, 4), c3=(4, 4), c4=4),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Inception(16, c1=8, c2=(8, 8), c3=(8, 8), c4=8),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(32, 10),
)

X = torch.randn(2, 1, 32, 32)
rows, logits = trace_module_shapes(googlenet_tiny, X)
for row in rows:
    print(row)

assert shape(logits) == (2, 10)"""
        ),
        md(
            """## 8.4.4 Concatenation Preserves Branch Identity as Channels

After concatenation, the next layer sees one tensor. It does not know which branch each channel came from unless you track the channel ranges yourself.

That is the engineering discipline:

```text
branch 1 channels occupy one slice
branch 2 channels occupy the next slice
branch 3 channels occupy the next slice
branch 4 channels occupy the final slice
```

The cell manually verifies the channel slices after concatenation.
"""
        ),
        code(
            """branches = [torch.full((1, 2, 3, 3), fill_value=v) for v in [1.0, 2.0, 3.0]]
merged = torch.cat(branches, dim=1)

print("merged shape:", shape(merged))
print("channel means:", merged.mean(dim=(0, 2, 3)))

assert shape(merged) == (1, 6, 3, 3)
assert torch.equal(merged[:, 0:2], branches[0])
assert torch.equal(merged[:, 2:4], branches[1])
assert torch.equal(merged[:, 4:6], branches[2])"""
        ),
        md(
            """## 8.4.5 Break It Deliberately: Branch Spatial Mismatch

Concatenation along channels only works when every other dimension matches. If one branch changes height or width, `torch.cat(..., dim=1)` rejects the result.

The theory-level mistake is asking the next layer to treat nonaligned spatial grids as if they were feature channels at the same locations.
"""
        ),
        code(
            """class BadInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.good = nn.Conv2d(3, 4, kernel_size=1)
        self.bad = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)

    def forward(self, X):
        return torch.cat([self.good(X), self.bad(X)], dim=1)


try:
    BadInception()(torch.randn(2, 3, 16, 16))
except RuntimeError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected branch spatial mismatch to fail")"""
        ),
        checkpoint(
            "8.4",
            [
                "Why does an Inception block use several branches instead of one path?",
                "Why must branch outputs agree on height and width before concatenation?",
                "How does a 1 by 1 bottleneck reduce the cost of a 5 by 5 branch?",
                "After concatenation, where is branch identity stored?",
                "Why is an Inception block not naturally represented as a simple `nn.Sequential` inside its own forward logic?",
            ],
        ),
    ]
    write_nb("Chapter 8.4 - Multi-Branch Networks (GoogLeNet).ipynb", cells)


def build_85() -> None:
    cells = [
        title_cell(
            "Chapter 8.5 - Batch Normalization",
            "Batch normalization adds a trainable normalization layer inside a network. It normalizes activations during training, learns a scale and shift, and stores running statistics for evaluation. The concept is simple; the framework behavior is stateful and worth inspecting carefully.",
            [
                "compute batch normalization manually for dense activations",
                "explain which axes are normalized in convolutional feature maps",
                "distinguish training behavior from evaluation behavior",
                "identify gamma, beta, running mean, and running variance in PyTorch",
                "debug batch normalization when a training batch has too little data per feature",
            ],
        ),
        setup_cell(),
        md(
            """## 8.5.0 The Problem This Notebook Solves

Deep networks can be hard to train partly because activation distributions can shift as earlier layers update. Batch normalization gives the network a normalization step inside the model:

```text
center activations using batch mean
scale by batch standard deviation
learn gamma and beta to restore useful scale and offset
store running statistics for evaluation
```

The key warning is that batch normalization is not just a formula. It has trainable parameters and persistent buffers. That connects directly to Chapter 6 parameter and buffer management.
"""
        ),
        md(
            """## 8.5.1 Manual Batch Normalization for Dense Activations

For a 2D tensor shaped `(batch, features)`, batch normalization computes one mean and variance per feature across the batch.

Then it applies:

```text
X_hat = (X - mean) / sqrt(variance + epsilon)
Y = gamma * X_hat + beta
```

`epsilon` is a small positive number that prevents division by zero.
"""
        ),
        code(
            """def manual_batch_norm_2d(X, gamma, beta, eps=1e-5):
    mean = X.mean(dim=0, keepdim=True)
    var = ((X - mean) ** 2).mean(dim=0, keepdim=True)
    X_hat = (X - mean) / torch.sqrt(var + eps)
    return gamma * X_hat + beta, mean, var


X = torch.tensor([[1.0, 10.0, 100.0],
                  [2.0, 20.0, 200.0],
                  [3.0, 30.0, 300.0]])
gamma = torch.ones(1, 3)
beta = torch.zeros(1, 3)
Y, mean, var = manual_batch_norm_2d(X, gamma, beta)

print("mean:", mean)
print("var:", var)
print("normalized feature means:", Y.mean(dim=0))

assert torch.allclose(Y.mean(dim=0), torch.zeros(3), atol=1e-5)"""
        ),
        md(
            """## 8.5.2 Convolutional BatchNorm Normalizes Per Channel

For image-like tensors shaped `(batch, channels, height, width)`, `BatchNorm2d` computes one mean and variance per channel. It averages over:

```text
batch dimension
height dimension
width dimension
```

It does not compute separate statistics for every pixel location. The same channel scale and shift are shared across spatial locations.
"""
        ),
        code(
            """X = torch.randn(4, 3, 5, 5) * torch.tensor([1.0, 5.0, 10.0]).view(1, 3, 1, 1)
mean = X.mean(dim=(0, 2, 3), keepdim=True)
var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
X_hat = (X - mean) / torch.sqrt(var + 1e-5)

channel_means = X_hat.mean(dim=(0, 2, 3))
channel_vars = X_hat.var(dim=(0, 2, 3), unbiased=False)

print("channel means:", channel_means)
print("channel vars:", channel_vars)

assert torch.allclose(channel_means, torch.zeros(3), atol=1e-5)
assert torch.allclose(channel_vars, torch.ones(3), atol=1e-4)"""
        ),
        md(
            """## 8.5.3 PyTorch BatchNorm Has Parameters and Buffers

`nn.BatchNorm1d(3)` owns:

- `weight`: learnable gamma, initialized to 1
- `bias`: learnable beta, initialized to 0
- `running_mean`: buffer, not usually learned by gradient descent
- `running_var`: buffer, not usually learned by gradient descent

Parameters are updated by the optimizer. Buffers are saved and moved with the module, but they are not optimizer parameters.
"""
        ),
        code(
            """bn = nn.BatchNorm1d(3)

print("parameters:", list(dict(bn.named_parameters()).keys()))
print("buffers:", list(dict(bn.named_buffers()).keys()))
print("state_dict:", list(bn.state_dict().keys()))

assert set(dict(bn.named_parameters())) == {"weight", "bias"}
assert {"running_mean", "running_var"}.issubset(dict(bn.named_buffers()))
assert "running_mean" in bn.state_dict()"""
        ),
        md(
            """## 8.5.4 Training Mode Updates Running Statistics; Eval Mode Uses Them

BatchNorm behaves differently in training and evaluation:

- In training mode, it normalizes using the current batch and updates running statistics.
- In evaluation mode, it normalizes using stored running statistics.

This is why `model.train()` and `model.eval()` are not optional ceremony. They change model behavior.
"""
        ),
        code(
            """bn = nn.BatchNorm1d(3, momentum=0.5)
before = bn.running_mean.clone()

bn.train()
train_batch = torch.tensor([[10.0, 20.0, 30.0],
                            [12.0, 22.0, 32.0],
                            [14.0, 24.0, 34.0]])
train_output = bn(train_batch)
after = bn.running_mean.clone()

bn.eval()
eval_batch = torch.tensor([[100.0, 200.0, 300.0],
                           [101.0, 201.0, 301.0]])
eval_output = bn(eval_batch)

print("running mean before:", before)
print("running mean after:", after)
print("eval output mean:", eval_output.mean(dim=0))

assert not torch.equal(before, after)
assert not torch.allclose(eval_output.mean(dim=0), torch.zeros(3), atol=1e-2)"""
        ),
        md(
            """## 8.5.5 BatchNorm Inside a CNN Block

In CNNs, batch normalization usually appears after convolution and before activation:

```text
Conv2d -> BatchNorm2d -> ReLU
```

The convolution creates feature maps. BatchNorm stabilizes channel scale. ReLU adds nonlinearity. The batch norm layer has trainable gamma/beta and running-stat buffers, so it participates in both optimization and checkpoint state.
"""
        ),
        code(
            """conv_bn_relu = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(4),
    nn.ReLU(),
)

X = torch.randn(2, 1, 8, 8)
Y = conv_bn_relu(X)
loss = Y.pow(2).mean()
loss.backward()

state_keys = list(conv_bn_relu.state_dict().keys())
print("output shape:", shape(Y))
print("state keys:", state_keys)

assert shape(Y) == (2, 4, 8, 8)
assert conv_bn_relu[1].weight.grad is not None
assert "1.running_mean" in state_keys"""
        ),
        md(
            """## 8.5.6 Break It Deliberately: Too Little Data Per Feature

During training, BatchNorm needs enough values to estimate a variance for each normalized feature. For `BatchNorm1d` with input shape `(batch, features)`, a batch size of 1 gives only one value per feature.

PyTorch rejects that in training mode because the batch statistics would be degenerate.
"""
        ),
        code(
            """bn = nn.BatchNorm1d(3)
bn.train()

try:
    bn(torch.randn(1, 3))
except ValueError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected BatchNorm1d to reject one value per feature in training")"""
        ),
        checkpoint(
            "8.5",
            [
                "What does batch normalization normalize in a 2D dense activation tensor?",
                "Which axes does `BatchNorm2d` reduce over for image-like feature maps?",
                "What are gamma and beta?",
                "Why are running mean and running variance buffers rather than optimizer parameters?",
                "Why can forgetting `model.eval()` change inference behavior?",
            ],
        ),
    ]
    write_nb("Chapter 8.5 - Batch Normalization.ipynb", cells)


def build_86() -> None:
    cells = [
        title_cell(
            "Chapter 8.6 - Residual Networks (ResNet) and ResNeXt",
            "ResNet changes the default function a block has to learn. Instead of asking stacked layers to learn a full transformation from scratch, a residual block learns an update added to the input. ResNeXt extends this family with grouped convolutions for a different capacity-compute tradeoff.",
            [
                "explain why residual blocks make identity mappings easier",
                "implement residual addition with and without projection",
                "trace shape changes through a tiny ResNet",
                "explain grouped convolution in ResNeXt terms",
                "debug residual addition when the two paths have incompatible shapes",
            ],
        ),
        setup_cell(),
        md(
            """## 8.6.0 The Problem This Notebook Solves

Deeper networks should be more expressive, but they can become harder to optimize. ResNet's core idea is to make a block learn a residual update:

```text
output = input + learned_update(input)
```

If the best behavior is close to "do nothing", the learned update can move toward zero. This gives the architecture an easier path to identity mappings than asking several layers to directly learn identity from scratch.
"""
        ),
        md(
            """## 8.6.1 Residual Addition Requires Matching Shapes

Addition is stricter than concatenation. To compute `Y + X`, both tensors must have the same shape or be broadcast-compatible. In ResNet blocks, the intended case is same shape:

```text
main path output shape == shortcut path output shape
```

The block below keeps channel count and spatial size unchanged.
"""
        ),
        code(
            """class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_projection=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if use_projection:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, X):
        return F.relu(self.main(X) + self.shortcut(X))


block = ResidualBlock(8, 8)
X = torch.randn(2, 8, 16, 16)
Y = block(X)

print("output:", shape(Y))
assert shape(Y) == shape(X)"""
        ),
        md(
            """## 8.6.2 Projection Shortcuts Change Shape Deliberately

When a residual stage changes channel count or spatial size, the shortcut path must change shape too. A 1 by 1 convolution can project the input into the required shape.

Here the main path uses stride 2 and changes channels from 8 to 16. The shortcut must do the same.
"""
        ),
        code(
            """projection_block = ResidualBlock(8, 16, stride=2, use_projection=True)
X = torch.randn(2, 8, 16, 16)
Y = projection_block(X)

shortcut_Y = projection_block.shortcut(X)
main_Y = projection_block.main(X)

print("main path:", shape(main_Y))
print("shortcut:", shape(shortcut_Y))
print("block output:", shape(Y))

assert shape(main_Y) == shape(shortcut_Y)
assert shape(Y) == (2, 16, 8, 8)"""
        ),
        md(
            """## 8.6.3 Identity Is Easy When the Residual Update Is Zero

This tiny block strips away convolution and batch normalization to isolate the function-class idea. If the learned update is zero, the residual block returns the input, up to the final activation.

For nonnegative inputs, `relu(X + 0)` equals `X`.
"""
        ),
        code(
            """class ZeroUpdateResidual(nn.Module):
    def forward(self, X):
        update = torch.zeros_like(X)
        return F.relu(X + update)


X = torch.rand(2, 3, 4, 4)
Y = ZeroUpdateResidual()(X)

print("max difference:", (Y - X).abs().max().item())
assert torch.equal(Y, X)"""
        ),
        md(
            """## 8.6.4 A Tiny ResNet Uses Residual Blocks as Stages

A ResNet is not just one residual block. It uses stages:

```text
stem -> residual stage -> residual stage -> global average pool -> classifier
```

The first block of a new stage often downsamples and changes channel count with a projection shortcut. Later blocks in the same stage keep shape.
"""
        ),
        code(
            """tiny_resnet = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
    ResidualBlock(8, 8),
    ResidualBlock(8, 16, stride=2, use_projection=True),
    ResidualBlock(16, 16),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(16, 10),
)

X = torch.randn(2, 1, 32, 32)
rows, logits = trace_module_shapes(tiny_resnet, X)
for row in rows:
    print(row)

assert shape(logits) == (2, 10)"""
        ),
        md(
            """## 8.6.5 ResNeXt Uses Grouped Convolutions

Grouped convolution splits input channels into groups. Each group is convolved separately, and the results are concatenated as output channels.

With `groups=4`, a convolution with 16 input channels behaves like four smaller convolutions over 4 channels each. This can reduce parameters and computation while preserving multiple parallel transformation paths. ResNeXt uses this idea inside residual-style blocks.
"""
        ),
        code(
            """dense_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, groups=1, bias=False)
grouped_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, groups=4, bias=False)

X = torch.randn(2, 16, 8, 8)
Y = grouped_conv(X)

print("dense weight shape:", shape(dense_conv.weight))
print("grouped weight shape:", shape(grouped_conv.weight))
print("dense params:", count_parameters(dense_conv))
print("grouped params:", count_parameters(grouped_conv))

assert shape(Y) == (2, 32, 8, 8)
assert count_parameters(grouped_conv) < count_parameters(dense_conv)"""
        ),
        md(
            """## 8.6.6 Break It Deliberately: Add Tensors With Different Channels

If the main path changes channel count and the shortcut remains identity, residual addition fails. The right fix is not to silence the error; the right fix is to add a projection shortcut or keep the shapes unchanged.
"""
        ),
        code(
            """bad_block = ResidualBlock(8, 16, use_projection=False)

try:
    bad_block(torch.randn(2, 8, 16, 16))
except RuntimeError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected residual addition to fail with channel mismatch")"""
        ),
        checkpoint(
            "8.6",
            [
                "What function does a residual block learn?",
                "Why does residual addition require a strict shape contract?",
                "When do we need a projection shortcut?",
                "Why can residual connections make identity mappings easier to represent?",
                "What does grouped convolution change compared with a dense convolution?",
            ],
        ),
    ]
    write_nb("Chapter 8.6 - Residual Networks (ResNet) and ResNeXt.ipynb", cells)


def build_87() -> None:
    cells = [
        title_cell(
            "Chapter 8.7 - Densely Connected Networks (DenseNet)",
            "DenseNet takes feature reuse seriously. Instead of adding a block's new features to the old representation, it concatenates them. Every later layer in a dense block receives all earlier feature maps as input.",
            [
                "explain how DenseNet differs from ResNet",
                "implement a dense block with channel growth",
                "compute the output channels from growth rate and block depth",
                "use transition layers to compress channels and downsample space",
                "debug concatenating along the wrong dimension",
            ],
        ),
        setup_cell(),
        md(
            """## 8.7.0 The Problem This Notebook Solves

ResNet combines old and new information by addition:

```text
new representation = old representation + update
```

DenseNet combines old and new information by concatenation:

```text
new representation = concatenate(old features, newly computed features)
```

Addition keeps channel count the same. Concatenation grows channel count. DenseNet's growth rate controls how many new channels each layer adds.
"""
        ),
        md(
            """## 8.7.1 Residual Addition and Dense Concatenation Preserve Information Differently

Addition merges two tensors into the same channel slots. Concatenation keeps both tensors as separate channel slices.

This is not automatically better or worse. It is a different inductive bias:

- ResNet says: learn an update to the current representation.
- DenseNet says: keep earlier features directly available to later layers.
"""
        ),
        code(
            """old = torch.randn(2, 4, 8, 8)
update = torch.randn(2, 4, 8, 8)

residual_style = old + update
dense_style = torch.cat([old, update], dim=1)

print("residual shape:", shape(residual_style))
print("dense shape:", shape(dense_style))

assert shape(residual_style) == (2, 4, 8, 8)
assert shape(dense_style) == (2, 8, 8, 8)"""
        ),
        md(
            """## 8.7.2 A Dense Block Grows Channels by the Growth Rate

Each layer in a dense block receives all previous features and produces `growth_rate` new channels. Those new channels are concatenated onto the running representation.

If the input has `C` channels, the block has `L` layers, and the growth rate is `G`, then:

```text
output channels = C + L * G
```
"""
        ),
        code(
            """def conv_block(in_channels, growth_rate):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
    )


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_convs, growth_rate):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = in_channels
        for _ in range(num_convs):
            self.blocks.append(conv_block(channels, growth_rate))
            channels += growth_rate
        self.out_channels = channels

    def forward(self, X):
        features = X
        self.channel_trace = [features.shape[1]]
        for block in self.blocks:
            new_features = block(features)
            features = torch.cat([features, new_features], dim=1)
            self.channel_trace.append(features.shape[1])
        return features


dense_block = DenseBlock(6, num_convs=3, growth_rate=4)
Y = dense_block(torch.randn(2, 6, 8, 8))

print("channel trace:", dense_block.channel_trace)
assert shape(Y) == (2, 18, 8, 8)
assert dense_block.out_channels == 18"""
        ),
        md(
            """## 8.7.3 Transition Layers Compress Channels and Downsample

Dense blocks grow channel count. Without a control mechanism, the network becomes increasingly wide. A transition layer usually performs:

```text
BatchNorm -> ReLU -> 1 by 1 convolution -> average pooling
```

The 1 by 1 convolution compresses channels. Average pooling reduces spatial size.
"""
        ),
        code(
            """def transition_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


transition = transition_block(18, 10)
Y2 = transition(Y)

print("after transition:", shape(Y2))
assert shape(Y2) == (2, 10, 4, 4)"""
        ),
        md(
            """## 8.7.4 Build a Tiny DenseNet

The DenseNet pattern is:

```text
stem -> dense block -> transition -> dense block -> global average pool -> classifier
```

The builder must track channel count carefully because dense blocks change it by concatenation.
"""
        ),
        code(
            """class TinyDenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 6, kernel_size=3, padding=1))
        self.block1 = DenseBlock(6, num_convs=2, growth_rate=4)
        self.trans1 = transition_block(self.block1.out_channels, 10)
        self.block2 = DenseBlock(10, num_convs=2, growth_rate=4)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                  nn.Linear(self.block2.out_channels, num_classes))

    def forward(self, X):
        X = self.stem(X)
        X = self.block1(X)
        X = self.trans1(X)
        X = self.block2(X)
        return self.head(X)


model = TinyDenseNet()
logits = model(torch.randn(2, 1, 16, 16))

print("block1 out channels:", model.block1.out_channels)
print("block2 out channels:", model.block2.out_channels)
print("logits:", shape(logits))

assert shape(logits) == (2, 10)"""
        ),
        md(
            """## 8.7.5 Break It Deliberately: Concatenate Along the Wrong Dimension

DenseNet growth is channel growth. If you concatenate along height or width, the code might produce a tensor, but it is not a dense feature stack.

The theory-level mistake is confusing "put tensors together" with "add feature channels for later layers".
"""
        ),
        code(
            """X = torch.randn(2, 6, 8, 8)
new = torch.randn(2, 4, 8, 8)
new_same_channels = torch.randn(2, 6, 8, 8)

correct = torch.cat([X, new], dim=1)
wrong = torch.cat([X, new_same_channels], dim=2)

print("correct:", shape(correct))
print("wrong:", shape(wrong))

try:
    assert wrong.shape[1] == X.shape[1] + new.shape[1]
except AssertionError:
    print("The tensor was concatenated, but channels did not grow.")"""
        ),
        checkpoint(
            "8.7",
            [
                "How does DenseNet combine old and new features differently from ResNet?",
                "What is the growth rate?",
                "Why does a dense block need careful channel bookkeeping?",
                "What does a transition layer do?",
                "Why is concatenating along height not the same as DenseNet feature reuse?",
            ],
        ),
    ]
    write_nb("Chapter 8.7 - Densely Connected Networks (DenseNet).ipynb", cells)


def build_88() -> None:
    cells = [
        title_cell(
            "Chapter 8.8 - Designing Convolution Network Architectures",
            "The final Chapter 8 notebook shifts from named architectures to architecture design spaces. AnyNet and RegNet-style thinking ask us to describe families of CNNs with stage rules, widths, depths, groups, and bottleneck ratios rather than hand-writing every layer one at a time.",
            [
                "describe a CNN as stem, stages, and head",
                "build a network from a compact architecture specification",
                "explain why design spaces are useful for architecture search",
                "generate RegNet-style stage widths from a simple rule",
                "debug grouped convolution divisibility constraints",
            ],
        ),
        setup_cell(),
        md(
            """## 8.8.0 The Problem This Notebook Solves

After AlexNet, VGG, NiN, GoogLeNet, ResNet, ResNeXt, and DenseNet, a pattern should be visible:

```text
modern CNNs are built from repeatable blocks arranged into stages
```

Architecture design then becomes a search over structured choices:

- how many stages?
- how deep is each stage?
- how wide is each stage?
- where does downsampling happen?
- are convolutions dense or grouped?
- does the block use residual connections?

AnyNet describes networks from broad stage choices. RegNet narrows the design space with simple rules for stage widths and depths.
"""
        ),
        md(
            """## 8.8.1 Stem, Stages, and Head

A practical CNN can often be described as:

```text
stem: early image-to-feature conversion
stages: repeated blocks at progressively lower spatial resolutions
head: global pooling and classifier
```

This vocabulary makes architectures easier to compare. Instead of memorizing a long list of layers, you can ask what each stage is doing.
"""
        ),
        code(
            """def conv_bn_relu(in_channels, out_channels, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                  padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


stem = conv_bn_relu(1, 8, stride=2)
X = torch.randn(2, 1, 64, 64)
Y = stem(X)

print("stem output:", shape(Y))
assert shape(Y) == (2, 8, 32, 32)"""
        ),
        md(
            """## 8.8.2 Build AnyNet From a Stage Specification

The architecture specification below is a list of stages. Each stage says:

```text
(depth, output_channels, first_block_stride)
```

Depth means how many blocks the stage repeats. The first block may downsample; later blocks usually keep spatial size.
"""
        ),
        code(
            """def make_stage(in_channels, depth, out_channels, first_stride):
    layers = []
    current = in_channels
    for i in range(depth):
        stride = first_stride if i == 0 else 1
        layers.append(conv_bn_relu(current, out_channels, stride=stride))
        current = out_channels
    return nn.Sequential(*layers), current


class TinyAnyNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        self.stem = conv_bn_relu(1, 8, stride=2)
        stages = []
        current = 8
        for depth, out_channels, stride in arch:
            stage, current = make_stage(current, depth, out_channels, stride)
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                  nn.Linear(current, num_classes))

    def forward(self, X):
        return self.head(self.stages(self.stem(X)))


anynet = TinyAnyNet(arch=[(1, 16, 2), (2, 32, 2), (1, 64, 2)])
logits = anynet(torch.randn(2, 1, 64, 64))

print("logits:", shape(logits))
assert shape(logits) == (2, 10)"""
        ),
        md(
            """## 8.8.3 Compare Design Choices With Parameter Counts

A design space is useful because it lets you compare many related models systematically. The cell compares two specifications:

- shallow and wider
- deeper and narrower

Parameter count is not the only metric. Runtime, memory access, accuracy, and hardware fit also matter. But parameter counting is a concrete first inspection tool.
"""
        ),
        code(
            """candidate_arches = {
    "shallow_wide": [(1, 24, 2), (1, 48, 2), (1, 96, 2)],
    "deeper_narrow": [(2, 16, 2), (2, 32, 2), (2, 64, 2)],
}

for name, arch in candidate_arches.items():
    model = TinyAnyNet(arch)
    logits = model(torch.randn(2, 1, 64, 64))
    print(name, "params", count_parameters(model), "logits", shape(logits))
    assert shape(logits) == (2, 10)"""
        ),
        md(
            """## 8.8.4 A RegNet-Style Width Rule

RegNet-style design narrows the search space by generating widths with a simple rule, then grouping repeated equal widths into stages.

This simplified version uses:

```text
raw width at block i = w0 + wa * i
quantize raw widths to multiples of q
merge consecutive blocks with the same quantized width into stages
```

The real design space includes more details, but this drill captures the key abstraction: generate architecture structure from a few parameters.
"""
        ),
        code(
            """def quantize(width, q):
    return int(round(width / q) * q)

def regnet_like_stages(depth, w0, wa, q):
    widths = [max(q, quantize(w0 + wa * i, q)) for i in range(depth)]
    stages = []
    for width in widths:
        if stages and stages[-1][1] == width:
            stages[-1] = (stages[-1][0] + 1, width)
        else:
            stages.append((1, width))
    return widths, stages


widths, stages = regnet_like_stages(depth=8, w0=16, wa=7, q=8)
print("block widths:", widths)
print("merged stages:", stages)

assert len(widths) == 8
assert sum(depth for depth, width in stages) == 8
assert all(width % 8 == 0 for width in widths)"""
        ),
        md(
            """## 8.8.5 Convert Generated Stages Into a Network

The generated stage list lacks stride information, so this cell adds a simple policy:

```text
downsample at the first block of every generated stage
```

This policy is not universal. The point is to separate design-space generation from model construction.
"""
        ),
        code(
            """generated_arch = [(depth, width, 2) for depth, width in stages[:3]]
model = TinyAnyNet(generated_arch)
X = torch.randn(2, 1, 64, 64)
rows = []
current = model.stem(X)
rows.append(("stem", shape(current)))
for i, stage in enumerate(model.stages):
    current = stage(current)
    rows.append((f"stage_{i}", shape(current)))
logits = model.head(current)

print(rows)
print("logits:", shape(logits))

assert shape(logits) == (2, 10)
assert rows[-1][1][1] == generated_arch[-1][1]"""
        ),
        md(
            """## 8.8.6 Break It Deliberately: Grouped Convolution Divisibility

Grouped convolution imposes a hard channel contract:

```text
input channels must be divisible by groups
output channels must be divisible by groups
```

Design-space search must respect these arithmetic constraints. Otherwise the model cannot even be constructed.
"""
        ),
        code(
            """try:
    nn.Conv2d(10, 16, kernel_size=3, padding=1, groups=4)
except ValueError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected grouped convolution construction to fail")"""
        ),
        md(
            """## 8.8.7 What This Chapter Does Not Do

This chapter does not run architecture search, train RegNet variants, benchmark GPUs, compare ImageNet accuracy, tune regularization, or report production metrics.

Those tasks require real datasets, controlled training budgets, and hardware-aware measurement. The chapter goal is narrower and more foundational:

- read modern CNNs as combinations of reusable blocks
- trace shape and channel contracts
- understand why bottlenecks, branches, residuals, dense concatenation, normalization, and design spaces exist
- recognize common architecture failures before large training runs hide them
"""
        ),
        checkpoint(
            "8.8",
            [
                "What are stem, stages, and head in a CNN architecture?",
                "Why is a design space different from one fixed model?",
                "What does stage depth control?",
                "Why do RegNet-style rules generate widths before constructing layers?",
                "Why must grouped convolution constraints be checked during architecture design?",
            ],
        ),
    ]
    write_nb("Chapter 8.8 - Designing Convolution Network Architectures.ipynb", cells)


def main() -> None:
    build_81()
    build_82()
    build_83()
    build_84()
    build_85()
    build_86()
    build_87()
    build_88()
    print(f"Wrote Chapter 8 notebooks to {OUT_DIR}")


if __name__ == "__main__":
    main()
