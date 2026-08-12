"""Generate Chapter 7 notebooks for the D2L rewrite project.

These notebooks stay notebook-native: all mechanics, checks, breakage demos, and
checkpoint prompts live directly in the .ipynb files. Full integration training
work remains outside the chapter notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Chapter 7 - Convolutional Neural Networks"
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
    body = """import torch
from torch import nn

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

def shape(x):
    return tuple(x.shape)
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

Run the notebook from top to bottom in a clean kernel. Every required tensor, helper, and model is defined inside this notebook. The cells use small tensors so that the mechanics are visible without turning the chapter into a larger experiment.

## You are done when you can

{outcome_lines}
"""
    )


def checkpoint(title: str, questions: list[str]) -> dict:
    lines = "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, start=1))
    return md(
        f"""## {title} Checkpoint

Answer these before moving on. Short markdown answers in the notebook are enough; the chapter does not need a separate notes file.

{lines}
"""
    )


def build_71() -> None:
    cells = [
        title_cell(
            "Chapter 7.1 - From Fully Connected Layers to Convolutions",
            "Chapter 7 begins with a modeling question: what changes when the input is not just a feature vector, but a spatial object? CNNs are built from assumptions about images and other grid-like data: nearby values are related, useful local patterns repeat across positions, and channels store different measurements at the same location.",
            [
                "explain why CNNs are an inductive bias rather than just another layer type",
                "explain why flattening images too early throws away spatial structure",
                "distinguish translation equivariance from invariance in plain English",
                "compare MLP and convolution parameter counts",
                "connect locality and weight sharing to the convolution layer",
            ],
        ),
        setup_cell(),
        md(
            """## 7.1.0 The Problem This Notebook Solves

An MLP treats an input as a vector. That is appropriate when the feature order has no special geometry or when the model has enough data and capacity to learn all relationships from scratch. Images are different. Pixel location matters, and nearby pixels usually form meaningful local patterns such as edges, corners, strokes, textures, and object parts.

If we flatten an image immediately, the model can still learn, but we have hidden useful structure:

```text
image grid: row, column, channel relationships are explicit
flattened vector: location relationships must be rediscovered from index positions
```

CNNs add an inductive bias. An inductive bias is a built-in modeling assumption that makes some solutions easier to learn than others. The CNN bias says:

- local neighborhoods are important
- the same detector can be useful at many positions
- channels are different measurements attached to each location

This is not a guarantee that CNNs are always best. It is a bet about the data. For images, that bet is often strong.

The handoff from Chapter 6 is direct: Chapter 6 taught modules as reusable computation blocks. Chapter 7 now asks what kind of block is appropriate when the input has spatial structure.
"""
        ),
        md(
            """## 7.1.1 Flattening Hides Where Pixels Live

A flattened image is still data, but adjacency becomes implicit. Pixel 6 and pixel 7 might be neighbors in the original image, or they might be separated by a row boundary depending on the flattening convention. A fully connected layer can learn from flattened values, but it has to learn spatial relationships indirectly.

The cell creates a simple vertical bright region and then shifts it one column to the right. As images, those two patterns are obviously related. As flat vectors, they are just different positions in a long list.

The theoretical issue is not that flattening destroys the values. It destroys the easy access to geometry. A dense layer can assign separate weights to every position, but it does not automatically know that the same edge detector might be useful one column over.

Before running the cell, predict:

- Both tensors should have shape `(1, 1, 5, 5)`.
- The flattened nonzero indices should change after shifting.
- The shifted image is related spatially, but not identical as a flat vector.
"""
        ),
        code(
            """image = torch.zeros(1, 1, 5, 5)
image[:, :, :, 0:2] = 1.0

shifted = torch.zeros(1, 1, 5, 5)
shifted[:, :, :, 1:3] = 1.0

image_nonzero = image.flatten().nonzero().flatten()[:10]
shifted_nonzero = shifted.flatten().nonzero().flatten()[:10]

print("first nonzero flattened positions:", image_nonzero.tolist())
print("first nonzero flattened positions after shift:", shifted_nonzero.tolist())

assert shape(image) == (1, 1, 5, 5)
assert not torch.equal(image.flatten(), shifted.flatten())"""
        ),
        md(
            """## 7.1.2 A Shared Local Detector Moves With the Pattern

This is the central CNN idea in miniature.

A local detector is a small set of weights that looks at a local window. A shared detector means the same weights are reused at many positions. If the input pattern moves, the detector's response moves with it.

That behavior is called translation equivariance:

```text
shift the input -> the feature map shifts in the corresponding way
```

Equivariance is not invariance. Invariance would mean:

```text
shift the input -> the final answer stays the same
```

Convolutions give equivariant feature maps. Pooling, aggregation, and later classifier layers can help produce more invariant final decisions. Keeping that distinction clear prevents a common conceptual mistake: a convolution does not ignore location; it preserves location while reusing a detector across locations.

The cell uses a hand-set detector so the idea is visible before learning enters the picture.
"""
        ),
        code(
            """edge_detector = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
with torch.no_grad():
    edge_detector.weight[:] = torch.tensor([[[[1.0, -1.0]]]])

response = edge_detector(image)
shifted_response = edge_detector(shifted)

edge_col = int(response[0, 0].mean(dim=0).argmax())
shifted_edge_col = int(shifted_response[0, 0].mean(dim=0).argmax())

print("response shape:", shape(response))
print("edge column:", edge_col)
print("edge column after shift:", shifted_edge_col)

assert shape(response) == (1, 1, 5, 4)
assert shifted_edge_col == edge_col + 1"""
        ),
        md(
            """## 7.1.3 Locality and Weight Sharing Reduce Parameter Count

CNNs are not only about fewer parameters, but parameter efficiency is one concrete consequence of the theory.

A dense hidden unit connected to a 28 by 28 image has a separate weight for every pixel. If we want 64 hidden units, the first layer already needs many weights. A convolutional layer uses a different idea:

```text
learn a small local detector
reuse it at many spatial positions
learn multiple detectors as output channels
```

This encodes two assumptions:

- locality: small neighborhoods contain useful patterns
- sharing: the same kind of pattern can matter in different locations

The parameter count drops because the model no longer learns a separate detector for every absolute position. It learns detectors that scan.

The tradeoff is that we have constrained the model. If absolute position matters in a way that should not be shared, this bias can hurt. For ordinary image features, it usually helps.
"""
        ),
        code(
            """height, width = 28, 28
hidden_units = 64

mlp_weights = height * width * hidden_units
conv_weights = 6 * 1 * 5 * 5  # six 5 by 5 filters over one input channel
conv_bias = 6

print("MLP first-layer weights:", mlp_weights)
print("Conv weights plus bias:", conv_weights + conv_bias)
print("ratio:", round(mlp_weights / (conv_weights + conv_bias), 1))

assert conv_weights + conv_bias < mlp_weights"""
        ),
        md(
            """## 7.1.4 Channels Store What Is Present at Each Location

A channel is a feature dimension attached to every spatial location.

For an RGB image:

```text
at row r, column c:
red value
green value
blue value
```

For a hidden CNN layer:

```text
at row r, column c:
edge-like evidence
texture-like evidence
color-like evidence
part-like evidence
```

The exact hidden channel meanings are learned, not manually named. But the shape idea is the same: spatial position says where evidence is; channel index says what kind of evidence is stored there.

This prepares the handoff to Chapter 7.4. A real convolution kernel does not just look across height and width. It also combines input channels.
"""
        ),
        code(
            """X = torch.zeros(3, 2, 2)
X[0] += 1.0
X[1] += 2.0
X[2] += 3.0

print("channel-first shape:", shape(X))
print("values at row 0, col 0 across channels:", X[:, 0, 0])

assert shape(X) == (3, 2, 2)
assert torch.equal(X[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))"""
        ),
        md(
            """## 7.1.5 Break It Deliberately: Shuffle Pixels

The CNN assumption is only useful when the grid has meaning. If pixels are randomly shuffled, nearby values in the tensor may no longer be nearby values in the original image.

This is a theory failure, not just a preprocessing error. The model's inductive bias says local windows matter, but the data pipeline has destroyed locality. A convolution can still compute something, but its local windows no longer correspond to meaningful image neighborhoods.

This is why data representation matters. The model's assumptions and the tensor layout must agree.
"""
        ),
        code(
            """torch.manual_seed(0)
flat = image.flatten()
permutation = torch.randperm(flat.numel())
shuffled = flat[permutation].reshape_as(image)

original_response = edge_detector(image)
shuffled_response = edge_detector(shuffled)

print("original response sum:", float(original_response.abs().sum()))
print("shuffled response sum:", float(shuffled_response.abs().sum()))
print("same response:", torch.allclose(original_response, shuffled_response))

assert not torch.allclose(original_response, shuffled_response)"""
        ),
        checkpoint(
            "7.1",
            [
                "What is an inductive bias, and what bias does a CNN add for image-like data?",
                "Why does flattening an image make spatial structure harder for an MLP to exploit?",
                "What is the difference between equivariance and invariance?",
                "Which two constraints turn a large dense image layer into a convolution-like layer?",
                "Why is parameter sharing useful for images but not automatically correct for every dataset?",
                "Why are channels best understood as measurements at the same location?",
            ],
        ),
    ]
    write_nb("Chapter 7.1 - From Fully Connected Layers to Convolutions.ipynb", cells)


def build_72() -> None:
    cells = [
        title_cell(
            "Chapter 7.2 - Convolutions for Images",
            "After Chapter 7.1 motivates locality and weight sharing, Chapter 7.2 opens the actual operation. A convolutional layer turns a small local pattern detector into a feature map by applying the same learned weights across many local windows.",
            [
                "explain a kernel as a learned local pattern detector",
                "implement two-dimensional cross-correlation with tiny tensors",
                "verify the manual operation against `nn.Conv2d`",
                "learn a tiny edge detector through gradient descent",
                "debug channel-shape errors in `Conv2d`",
            ],
        ),
        setup_cell(
            """def corr2d(X, K):
    h, w = K.shape
    out_h = X.shape[0] - h + 1
    out_w = X.shape[1] - w + 1
    Y = torch.zeros(out_h, out_w, dtype=X.dtype)
    for i in range(out_h):
        for j in range(out_w):
            window = X[i : i + h, j : j + w]
            Y[i, j] = (window * K).sum()
    return Y"""
        ),
        md(
            """## 7.2.0 The Problem This Notebook Solves

Chapter 7.1 said a CNN reuses local detectors. This notebook answers:

```text
what does one local detector actually compute?
```

A kernel is a small grid of weights. At each valid location, it aligns with a window of the input. Matching entries are multiplied, and all products are added into one output value. Sliding the kernel over all valid locations creates a feature map.

Plain-English meaning:

- The kernel asks a local question.
- The feature map records where the answer is strong.
- The same question is asked at many positions.

In deep learning libraries, this operation is usually cross-correlation, even though the layer is called convolution. The difference is whether the kernel is flipped before sliding. For learned kernels, this naming mismatch usually does not matter because training can learn whichever orientation is useful. For hand-written kernels, it matters enough to be precise.

The handoff from this notebook to later CNN sections is direct:

- padding changes which windows are valid near the border
- stride changes which window positions are visited
- channels let a kernel combine several measurements at each location
- pooling summarizes feature maps after convolution
"""
        ),
        md(
            """## 7.2.1 Manual Cross-Correlation

The manual implementation is intentionally tiny. It is not for speed. It is for ownership of the idea.

For each output position:

```text
choose a window from X
multiply the window by K entry by entry
sum the products
write that scalar into Y
```

The output is smaller than the input because the kernel only visits positions where it fully fits inside the input. No padding is used yet.

Before running the cell, inspect the top-left window:

```text
[[0, 1],
 [3, 4]]
```

With kernel:

```text
[[1, 0],
 [0, -1]]
```

The response is `0 * 1 + 1 * 0 + 3 * 0 + 4 * -1 = -4`.
"""
        ),
        code(
            """X = torch.tensor([
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0],
])
K = torch.tensor([
    [1.0, 0.0],
    [0.0, -1.0],
])

Y = corr2d(X, K)

print(Y)

expected = torch.tensor([[-4.0, -4.0], [-4.0, -4.0]])
assert torch.equal(Y, expected)
assert shape(Y) == (2, 2)"""
        ),
        md(
            """## 7.2.2 Verify Against `nn.Conv2d`

Manual code proves the operation. `nn.Conv2d` is the production-shaped PyTorch module.

PyTorch image batches are shaped:

```text
batch, channels, height, width
```

That extra batch dimension matters because models usually process many examples at once. The channel dimension matters because real images and hidden feature maps carry multiple measurements per location.

In this cell, there is one example, one input channel, one output channel, and a 2 by 2 kernel. We manually copy the kernel values into the layer's weight so the PyTorch result should match `corr2d` exactly.
"""
        ),
        code(
            """conv = nn.Conv2d(1, 1, kernel_size=(2, 2), bias=False)
with torch.no_grad():
    conv.weight[:] = K.reshape(1, 1, 2, 2)

X4 = X.reshape(1, 1, 3, 3)
Y4 = conv(X4)

print("manual:")
print(Y)
print("conv2d:")
print(Y4[0, 0])

assert torch.allclose(Y4[0, 0], Y)
assert shape(Y4) == (1, 1, 2, 2)"""
        ),
        md(
            """## 7.2.3 A Hand-Written Edge Detector

An edge is a sharp local change. The kernel `[1, -1]` asks a simple local question:

```text
is the left value larger than the right value?
```

If the left pixel is bright and the right pixel is dark, the response is positive. If both are similar, the response is near zero. If the direction is reversed, the response is negative.

This is a handcrafted detector. It is useful pedagogically because the kernel has an interpretable meaning. Real CNNs usually learn kernels from data, and early learned kernels often become edge-like because edges are useful low-level image features.
"""
        ),
        code(
            """X = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
])
K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X, K)
print(Y)

assert shape(Y) == (3, 3)
assert torch.equal(Y[:, 1], torch.ones(3))"""
        ),
        md(
            """## 7.2.4 Learn the Kernel Instead of Hand-Picking It

The hand-written edge detector shows what a kernel can mean. But manually designing every useful visual feature does not scale.

The deep learning move is:

```text
make the kernel a parameter
define a loss
use gradients to change the kernel
```

This cell gives the model an artificial target feature map: respond strongly at the vertical edge and weakly elsewhere. The target is not a real classification label; it is a tiny supervised signal designed to expose the learning mechanism.

The important theoretical point is that convolutional layers do not need humans to write the filters. The architecture constrains the form of the computation - local and shared - but the actual detector values are learned.

Before running the cell, predict:

- The final loss should become very small.
- The learned kernel should become edge-like.
- The weight shape should remain `(1, 1, 1, 2)`: output channel, input channel, kernel height, kernel width.
"""
        ),
        code(
            """torch.manual_seed(0)
X = torch.tensor([[[[1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0]]]])
target = torch.tensor([[[[0.0, 1.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 1.0, 0.0]]]])

conv = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
optimizer = torch.optim.SGD(conv.parameters(), lr=0.3)

for step in range(120):
    pred = conv(X)
    loss = ((pred - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("final loss:", float(loss.detach()))
print("learned kernel:", conv.weight.detach().reshape(1, 2))

assert float(loss.detach()) < 1e-4
assert shape(conv.weight) == (1, 1, 1, 2)"""
        ),
        md(
            """## 7.2.5 Break It Deliberately: Wrong Input Channel Count

`Conv2d(in_channels=1, ...)` means the layer expects each input example to have exactly one channel. If the input tensor has three channels, the layer's kernel does not have enough channel slices to combine them.

The theory-level mistake is a mismatch between the model's expected measurement structure and the data's measurement structure.

```text
model says: each location has 1 measurement
input says: each location has 3 measurements
```

Chapter 7.4 will show how kernels handle multiple input channels correctly. Here, the goal is to make the failure recognizable.
"""
        ),
        code(
            """conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
bad_X = torch.zeros(1, 3, 4, 4)

try:
    conv(bad_X)
except RuntimeError as err:
    print(type(err).__name__)
    print(str(err).splitlines()[0])
else:
    raise AssertionError("The channel mismatch should have failed.")"""
        ),
        checkpoint(
            "7.2",
            [
                "Why is a kernel best understood as a local pattern detector?",
                "In the manual `corr2d`, what does each output value summarize?",
                "Why does no-padding convolution shrink the spatial output?",
                "Why is `Conv2d` input shaped as batch, channels, height, width?",
                "How did the learned-kernel cell prove that kernels are trainable parameters?",
                "Why did the wrong-channel input fail?",
            ],
        ),
    ]
    write_nb("Chapter 7.2 - Convolutions for Images.ipynb", cells)


def build_73() -> None:
    cells = [
        title_cell(
            "Chapter 7.3 - Padding and Stride",
            "A convolution is not only defined by its kernel. Padding decides how the border is treated, and stride decides how densely the kernel samples locations. These choices control spatial resolution, information loss, computation, and the shape contracts between layers.",
            [
                "explain padding and stride as architectural choices, not only formula inputs",
                "compute convolution output sizes as ordinary Python code",
                "verify padding and stride against `nn.Conv2d`",
                "explain why padding preserves border information",
                "debug impossible kernel/input geometry",
            ],
        ),
        setup_cell(
            """def conv_out_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size + 2 * padding - kernel_size) // stride + 1"""
        ),
        md(
            """## 7.3.0 The Problem This Notebook Solves

Chapter 7.2 used valid convolution: the kernel only visited positions where it fit fully inside the input. That made the output smaller. If we stack many such layers, spatial maps can shrink quickly.

Padding and stride are the two main controls introduced here:

- Padding adds border values before the kernel slides.
- Stride changes how far the kernel moves between windows.

These are not cosmetic settings. They change the representation pipeline:

```text
padding controls border treatment and spatial preservation
stride controls downsampling and compute
kernel size controls local receptive field
```

The key habit is to predict output shape before connecting layers. Shape formulas are not abstract math here; they are engineering contracts. If one layer produces an unexpected height or width, the next layer may fail or silently receive a representation different from what you intended.
"""
        ),
        md(
            """## 7.3.1 Output Size Is a Mechanical Contract

For one spatial dimension, use this code pattern:

```text
add padding on both sides
subtract the kernel size
count stride steps that fit
include the first position
```

The formula is written as ordinary Python because the purpose is shape reasoning, not symbolic manipulation:

```python
out = (input_size + 2 * padding - kernel_size) // stride + 1
```

The integer division matters. With stride greater than 1, not every possible window start is visited. The output counts only starts where the kernel still fits.
"""
        ),
        code(
            """configs = [
    {"input_size": 8, "kernel_size": 3, "padding": 0, "stride": 1},
    {"input_size": 8, "kernel_size": 3, "padding": 1, "stride": 1},
    {"input_size": 8, "kernel_size": 3, "padding": 1, "stride": 2},
]

for cfg in configs:
    out = conv_out_size(**cfg)
    conv = nn.Conv2d(1, 1, cfg["kernel_size"], padding=cfg["padding"], stride=cfg["stride"])
    Y = conv(torch.zeros(1, 1, cfg["input_size"], cfg["input_size"]))
    print(cfg, "formula:", out, "pytorch:", shape(Y))
    assert shape(Y)[2:] == (out, out)"""
        ),
        md(
            """## 7.3.2 Padding Adds Artificial Border Values

Without padding, border pixels participate in fewer windows than center pixels. A corner pixel might appear in only one 3 by 3 valid window, while a center pixel appears in many. Padding changes that treatment.

Padding can preserve spatial size, which is useful when stacking layers because it prevents feature maps from shrinking after every convolution.

But padding is not new evidence. Zero padding adds artificial border values. The model can learn to handle them, but they are not real pixels. That is the tradeoff:

```text
padding preserves spatial dimensions and border participation
padding also introduces artificial boundary assumptions
```
"""
        ),
        code(
            """X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
padded = torch.zeros(4, 4)
padded[1:3, 1:3] = X

print(padded)

conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
Y = conv(torch.zeros(1, 1, 5, 5))

assert shape(padded) == (4, 4)
assert shape(Y) == (1, 1, 5, 5)"""
        ),
        md(
            """## 7.3.3 Stride Skips Window Positions

Stride controls sampling density.

Stride 1 asks the local detector at every possible position. Stride 2 asks it at every other position. Larger stride reduces the spatial size of the feature map and reduces computation, but it also discards some location detail.

The theoretical meaning is downsampling:

```text
smaller spatial map
larger effective jump between neighboring output values
less memory and compute
less precise spatial information
```

Stride is therefore an architectural decision. It affects what information later layers can access.
"""
        ),
        code(
            """positions_stride_1 = list(range(0, 6 - 3 + 1, 1))
positions_stride_2 = list(range(0, 6 - 3 + 1, 2))

print("stride 1 starts:", positions_stride_1)
print("stride 2 starts:", positions_stride_2)

conv = nn.Conv2d(1, 1, kernel_size=3, stride=2)
Y = conv(torch.zeros(1, 1, 6, 6))

assert positions_stride_1 == [0, 1, 2, 3]
assert positions_stride_2 == [0, 2]
assert shape(Y) == (1, 1, 2, 2)"""
        ),
        md(
            """## 7.3.4 Padding and Stride Do Not Change Parameter Count

This is a key separation:

```text
parameter count: what weights are learned
output shape: where and how often those weights are applied
```

Padding and stride change the spatial geometry of the computation, but they do not create new kernel weights. A 3 by 3 kernel over 3 input channels with 8 output channels has the same learned weight count regardless of whether it scans densely, skips positions, or uses padding.

This helps separate capacity from resolution. More channels or larger kernels increase parameter count. Stride and padding mainly change feature-map size and border behavior.
"""
        ),
        code(
            """a = nn.Conv2d(3, 8, kernel_size=3, padding=0, stride=1)
b = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2)

params_a = sum(p.numel() for p in a.parameters())
params_b = sum(p.numel() for p in b.parameters())

print("params without padding/stride:", params_a)
print("params with padding/stride:", params_b)

assert params_a == params_b == 8 * 3 * 3 * 3 + 8"""
        ),
        md(
            """## 7.3.5 Break It Deliberately: Kernel Too Large

The kernel must fit somewhere. If the input is 3 by 3 and the kernel is 5 by 5 with no padding, there is no valid local window.

The theory-level mistake is asking a local detector to inspect a neighborhood larger than the available representation. Padding could create artificial space around the input, but without padding, the operation has no valid output location.

This failure is useful because it connects the shape formula to a physical sliding-window interpretation.
"""
        ),
        code(
            """conv = nn.Conv2d(1, 1, kernel_size=5)
X = torch.zeros(1, 1, 3, 3)

try:
    conv(X)
except RuntimeError as err:
    print(type(err).__name__)
    print(str(err).splitlines()[0])
else:
    raise AssertionError("The kernel should not fit inside the input.")"""
        ),
        checkpoint(
            "7.3",
            [
                "Why are padding and stride architectural choices rather than only API arguments?",
                "What do padding and stride each control?",
                "Why does `padding=1` preserve size for a 3 by 3 kernel with stride 1?",
                "What information tradeoff does larger stride create?",
                "Why does stride affect output shape but not parameter count?",
                "What does it mean mechanically when the kernel is too large for the input?",
            ],
        ),
    ]
    write_nb("Chapter 7.3 - Padding and Stride.ipynb", cells)


def build_74() -> None:
    cells = [
        title_cell(
            "Chapter 7.4 - Multiple Input and Multiple Output Channels",
            "A real CNN layer does two things at once: it looks over a local spatial window, and it mixes the channel values available in that window. Output channels are learned feature maps, each produced by its own kernel across all input channels.",
            [
                "explain channels as feature dimensions attached to spatial locations",
                "manually combine multiple input channels",
                "explain the `Conv2d` weight shape",
                "count parameters with and without bias",
                "show that a 1 by 1 convolution is a per-location channel mixer",
            ],
        ),
        setup_cell(
            """def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    pieces = [corr2d(x, k) for x, k in zip(X, K)]
    return torch.stack(pieces).sum(dim=0)

def conv2d_param_count(in_channels, out_channels, kernel_h, kernel_w, bias=True):
    weights = out_channels * in_channels * kernel_h * kernel_w
    return weights + (out_channels if bias else 0)"""
        ),
        md(
            """## 7.4.0 The Problem This Notebook Solves

Earlier examples used one input channel so the sliding-window idea stayed simple. Real images and hidden CNN layers usually have multiple channels.

The channel dimension answers:

```text
what kinds of information exist at each spatial location?
```

For RGB images, the channels are color measurements. For hidden layers, the channels are learned feature maps. A useful detector often needs to combine those measurements. For example, a detector might care about a red-green contrast, or a hidden-layer detector might combine edge evidence and texture evidence.

A convolutional kernel for one output channel therefore has:

```text
one spatial kernel per input channel
```

Each channel-specific kernel creates a response map. Those response maps are added to form one output feature map. Multiple output channels repeat this process with different learned kernels.
"""
        ),
        md(
            """## 7.4.1 Multiple Input Channels Are Summed Into One Response Map

For one output channel, the layer has one small spatial kernel per input channel. Each channel produces a response map, and those maps are added.

The theory-level meaning is that one feature detector may depend on several kinds of evidence. It is not forced to inspect red, green, blue, or hidden feature channels independently. It can learn how to combine them.

The manual function below makes that explicit:

```text
for each input channel:
    run 2D correlation
sum the channel responses
```
"""
        ),
        code(
            """X = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
K = torch.ones(2, 2, 2)

Y = corr2d_multi_in(X, K)

print("input shape:", shape(X))
print("kernel shape:", shape(K))
print(Y)

assert shape(Y) == (2, 2)"""
        ),
        md(
            """## 7.4.2 Verify Multi-Input-Channel Behavior With `Conv2d`

PyTorch stores convolution weights in this order:

```text
output channels, input channels, kernel height, kernel width
```

That order says exactly how to read the layer:

- choose an output feature map
- for that output map, look across all input channels
- for each input channel, use a small spatial kernel

The cell copies the manual kernel into `Conv2d` and verifies the result. This is the bridge from scratch mechanics to the library API.
"""
        ),
        code(
            """conv = nn.Conv2d(2, 1, kernel_size=2, bias=False)
with torch.no_grad():
    conv.weight[:] = K.reshape(1, 2, 2, 2)

Y_torch = conv(X.reshape(1, 2, 3, 3))

print("conv weight shape:", shape(conv.weight))
print(Y_torch[0, 0])

assert shape(conv.weight) == (1, 2, 2, 2)
assert torch.allclose(Y_torch[0, 0], Y)"""
        ),
        md(
            """## 7.4.3 Multiple Output Channels Mean Multiple Learned Kernels

One output channel gives one learned feature map. Real layers need many feature maps because images contain many useful patterns.

The output channel count is therefore the layer's width. More output channels mean more learned detectors and more capacity. They also mean more computation and memory.

The parameter count comes from:

```text
output channels * input channels * kernel height * kernel width
plus one bias per output channel if bias=True
```

The output shape separates batch, output channels, and spatial dimensions:

```text
batch, output_channels, output_height, output_width
```
"""
        ),
        code(
            """conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1, bias=True)
X = torch.zeros(4, 3, 8, 8)
Y = conv(X)

param_count = sum(p.numel() for p in conv.parameters())
expected_params = conv2d_param_count(3, 5, 3, 3, bias=True)

print("output shape:", shape(Y))
print("weight shape:", shape(conv.weight))
print("parameter count:", param_count)

assert shape(Y) == (4, 5, 8, 8)
assert param_count == expected_params"""
        ),
        md(
            """## 7.4.4 A 1 by 1 Convolution Mixes Channels at Each Location

A 1 by 1 convolution sounds spatially tiny, but it can be powerful because it mixes channels.

At each row and column, the layer sees the channel vector at that exact location. It applies the same linear transformation to that vector everywhere.

Plain-English meaning:

```text
do not look at neighboring locations
recombine the feature types at each location
optionally change the channel count
```

This is why 1 by 1 convolutions appear in many modern CNN blocks. They can compress channels, expand channels, or mix feature evidence before or after spatial convolutions.

The cell proves the equivalence by rebuilding a 1 by 1 convolution as a regular linear layer applied to every pixel's channel vector.
"""
        ),
        code(
            """torch.manual_seed(0)
conv = nn.Conv2d(3, 2, kernel_size=1, bias=False)
linear = nn.Linear(3, 2, bias=False)

with torch.no_grad():
    linear.weight[:] = conv.weight[:, :, 0, 0]

X = torch.randn(1, 3, 4, 5)
Y_conv = conv(X)

pixels = X.permute(0, 2, 3, 1).reshape(-1, 3)
Y_linear = linear(pixels).reshape(1, 4, 5, 2).permute(0, 3, 1, 2)

print("conv output shape:", shape(Y_conv))
print("linear-rebuilt output shape:", shape(Y_linear))
print("max difference:", float((Y_conv - Y_linear).abs().max()))

assert torch.allclose(Y_conv, Y_linear, atol=1e-6)"""
        ),
        md(
            """## 7.4.5 Break It Deliberately: Use NHWC Instead of NCHW

PyTorch `Conv2d` expects:

```text
NCHW: batch, channels, height, width
```

Some other libraries or image utilities use:

```text
NHWC: batch, height, width, channels
```

These shapes can contain the same raw numbers but mean different things. If you pass NHWC to PyTorch without rearranging dimensions, PyTorch interprets the height as the channel count.

The theory-level mistake is not respecting the semantic meaning of tensor axes. Shape numbers alone are not enough; axis meaning matters.
"""
        ),
        code(
            """conv = nn.Conv2d(3, 4, kernel_size=3)
bad_X = torch.zeros(1, 8, 8, 3)

try:
    conv(bad_X)
except RuntimeError as err:
    print(type(err).__name__)
    print(str(err).splitlines()[0])
else:
    raise AssertionError("NHWC input should fail for this Conv2d layer.")"""
        ),
        checkpoint(
            "7.4",
            [
                "Why does a useful detector often need to combine several input channels?",
                "Why does a convolution kernel span all input channels?",
                "What does each output channel represent?",
                "How do output channels relate to model capacity and computation?",
                "Why does a 1 by 1 convolution still have many parameters when channel counts are large?",
                "What is wrong with passing NHWC tensors to PyTorch `Conv2d`?",
            ],
        ),
    ]
    write_nb("Chapter 7.4 - Multiple Input and Multiple Output Channels.ipynb", cells)


def build_75() -> None:
    cells = [
        title_cell(
            "Chapter 7.5 - Pooling",
            "Pooling is a fixed spatial summarization operation. It has no learned weights, but it strongly shapes what information survives, how large feature maps remain, and how sensitive later layers are to small shifts.",
            [
                "explain pooling as an information tradeoff rather than just downsampling",
                "compute max and average pooling on tiny tensors",
                "explain why pooling preserves channel count",
                "show how pooling can discard precise location information",
                "use global average pooling to collapse spatial dimensions",
            ],
        ),
        setup_cell(),
        md(
            """## 7.5.0 The Problem This Notebook Solves

Convolution produces feature maps. Those maps can be large, and nearby positions often contain related evidence. Pooling summarizes small spatial neighborhoods to reduce resolution and make later representations less sensitive to tiny movements.

The important word is summarize. Pooling does not learn a detector. It applies a fixed rule:

- max pooling keeps the strongest response
- average pooling keeps the average response

This creates a tradeoff:

```text
less spatial detail
less memory and compute
more tolerance to small shifts
possible loss of exact location information
```

Pooling is therefore not automatically good or bad. It is an architectural choice about which information should survive.

The handoff from earlier sections:

- convolution creates local feature responses
- padding and stride control response-map geometry
- channels store different feature types
- pooling summarizes each feature map spatially
"""
        ),
        md(
            """## 7.5.1 Max Pooling and Average Pooling

Max pooling and average pooling answer different questions.

Max pooling asks:

```text
was this feature strongly present anywhere in the window?
```

Average pooling asks:

```text
what was the typical strength of this feature in the window?
```

For sparse detector-like activations, max pooling can preserve a strong local signal. For smoother feature maps, average pooling can preserve broader context. Neither is universally correct; each encodes a different summary.
"""
        ),
        code(
            """window = torch.tensor([[1.0, 3.0], [2.0, 0.0]])

print("max:", window.max())
print("average:", window.mean())

assert window.max().item() == 3.0
assert window.mean().item() == 1.5"""
        ),
        md(
            """The next cell applies both pooling rules over non-overlapping 2 by 2 windows. Read the input as a single feature map. Each output value summarizes one local block of four values.
"""
        ),
        code(
            """X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
max_pool = nn.MaxPool2d(kernel_size=2)
avg_pool = nn.AvgPool2d(kernel_size=2)

Y_max = max_pool(X)
Y_avg = avg_pool(X)

print("input:")
print(X[0, 0])
print("max pooled:")
print(Y_max[0, 0])
print("average pooled:")
print(Y_avg[0, 0])

assert shape(Y_max) == (1, 1, 2, 2)
assert shape(Y_avg) == (1, 1, 2, 2)"""
        ),
        md(
            """## 7.5.2 Pooling Uses Window, Padding, and Stride

Pooling has geometry like convolution, but no learned kernel weights. It still has a window size and a stride, so it still changes output shape.

By default, many pooling layers use stride equal to the pooling window size. That creates non-overlapping windows. Setting stride to 1 creates overlapping windows and preserves more spatial resolution.

The theory-level distinction is:

```text
convolution: learned local weighted summary
pooling: fixed local summary
```
"""
        ),
        code(
            """X = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
pool_stride_1 = nn.MaxPool2d(kernel_size=2, stride=1)
pool_stride_2 = nn.MaxPool2d(kernel_size=2, stride=2)

Y1 = pool_stride_1(X)
Y2 = pool_stride_2(X)

print("stride 1 shape:", shape(Y1))
print("stride 2 shape:", shape(Y2))
print("stride 1 output:")
print(Y1[0, 0])

assert shape(Y1) == (1, 1, 2, 2)
assert shape(Y2) == (1, 1, 1, 1)"""
        ),
        md(
            """## 7.5.3 Pooling Preserves Channel Count

Pooling summarizes within each channel separately. It does not mix channels and it does not choose new output channels.

This is a major difference from convolution:

```text
convolution can change channel count
pooling usually preserves channel count
```

The reason is conceptual. Each channel is a feature map. Pooling says "summarize where this same feature appears nearby." It does not say "combine this feature with other feature types." Channel mixing is the job of convolution, especially ordinary multi-channel convolution or 1 by 1 convolution.
"""
        ),
        code(
            """X = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
pool = nn.MaxPool2d(2)
Y = pool(X)

print("input shape:", shape(X))
print("output shape:", shape(Y))

assert shape(Y) == (1, 2, 2, 2)
assert shape(Y)[1] == shape(X)[1]"""
        ),
        md(
            """## 7.5.4 Pooling Can Discard Precise Location

Pooling can make a representation less sensitive to small shifts because several nearby input configurations collapse to the same output. That is useful when the exact pixel location of a feature should not matter much.

But this is also information loss. If two different local arrangements produce the same pooled value, later layers cannot recover which arrangement happened.

The example places the same strong activation at two different locations inside one 2 by 2 pooling window. Max pooling returns the same result for both.

This is the cleanest way to understand the tradeoff:

```text
pooling gives tolerance by discarding detail
```
"""
        ),
        code(
            """A = torch.zeros(1, 1, 4, 4)
B = torch.zeros(1, 1, 4, 4)
A[0, 0, 0, 0] = 5.0
B[0, 0, 1, 1] = 5.0

pool = nn.MaxPool2d(2)
YA = pool(A)
YB = pool(B)

print("pooled A:")
print(YA[0, 0])
print("pooled B:")
print(YB[0, 0])

assert torch.equal(YA, YB)"""
        ),
        md(
            """## 7.5.5 Global Average Pooling Collapses Spatial Dimensions

Global average pooling averages each entire feature map into one value per channel.

That means it intentionally discards exact spatial location at the end of a feature extractor. The resulting vector says, roughly:

```text
how strongly was each feature channel present overall?
```

Modern CNN classifiers often use global average pooling before the final linear layer because it reduces parameters and encourages the classifier to depend on feature presence rather than exact final-map position.

This also creates a graceful handoff to LeNet. Classic LeNet flattens a small spatial map and uses dense layers. Many modern CNNs instead use global average pooling before classification.
"""
        ),
        code(
            """X = torch.randn(3, 8, 6, 6)
pool = nn.AdaptiveAvgPool2d((1, 1))
Y = pool(X)
flattened = Y.flatten(start_dim=1)

print("pooled shape:", shape(Y))
print("flattened shape:", shape(flattened))

assert shape(Y) == (3, 8, 1, 1)
assert shape(flattened) == (3, 8)"""
        ),
        md(
            """## 7.5.6 Break It Deliberately: Pooling Window Too Large

Like convolution, ordinary pooling needs a valid window location. If the pooling window is larger than the feature map and there is no adaptive rule or padding to handle it, there is no output to compute.

This failure reinforces the same geometry discipline from padding and stride:

```text
every spatial operation has a window geometry
the window must fit or be deliberately adapted
```
"""
        ),
        code(
            """pool = nn.MaxPool2d(5)
X = torch.zeros(1, 1, 4, 4)

try:
    pool(X)
except RuntimeError as err:
    print(type(err).__name__)
    print(str(err).splitlines()[0])
else:
    raise AssertionError("The pooling window should not fit.")"""
        ),
        checkpoint(
            "7.5",
            [
                "Why is pooling best understood as summarization with information loss?",
                "How do max pooling and average pooling summarize a window differently?",
                "Why does pooling preserve channel count?",
                "Why does pooling sometimes help with small shifts?",
                "How can pooling help with small shifts while losing exact location?",
                "What shape does global average pooling produce before flattening?",
            ],
        ),
    ]
    write_nb("Chapter 7.5 - Pooling.ipynb", cells)


def build_76() -> None:
    cells = [
        title_cell(
            "Chapter 7.6 - Convolutional Neural Networks (LeNet)",
            "LeNet is the first full CNN architecture in this path. It combines the pieces from Chapter 7 into one representation pipeline: convolution for local feature extraction, activation for nonlinearity, pooling for spatial summarization, flattening for vector classification, and dense layers for class logits.",
            [
                "explain LeNet as a staged representation pipeline",
                "trace LeNet tensor shapes layer by layer",
                "count parameters by layer",
                "run one synthetic classification training step",
                "debug the common flatten-to-linear size mismatch",
            ],
        ),
        setup_cell(
            """def make_lenet():
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )

def trace_shapes(net, X):
    rows = []
    current = X
    for idx, layer in enumerate(net):
        current = layer(current)
        rows.append((idx, type(layer).__name__, shape(current)))
    return rows"""
        ),
        md(
            """## 7.6.0 The Problem This Notebook Solves

The previous notebooks studied CNN pieces separately. LeNet is where those pieces become an end-to-end classifier.

The representation story is:

```text
image pixels
-> local low-level feature maps
-> smaller pooled feature maps
-> richer feature maps
-> smaller pooled feature maps
-> flattened feature vector
-> class logits
```

This is not just a list of layers. It is a sequence of changing representation types:

- early tensors preserve spatial structure
- channels increase as learned feature variety increases
- spatial size shrinks as the model summarizes location
- flattening converts spatial feature maps into a vector
- dense layers turn that vector into class scores

This notebook does not try to get real accuracy. That belongs later. Here, the rigorous goal is to make the architecture auditable: every shape, parameter count, loss, gradient, and update should make sense.
"""
        ),
        md(
            """## 7.6.1 Build LeNet

The original LeNet used sigmoid-style activations and average pooling. Modern CNNs often use ReLU, batch normalization, residual connections, and different downsampling blocks. LeNet is still useful because it is small enough to audit completely.

Read the model as a contract:

```text
input: batch, 1 channel, 28 height, 28 width
output: batch, 10 class scores
```

The final output values are logits. They are raw class scores, not probabilities.
"""
        ),
        code(
            """net = make_lenet()
X = torch.zeros(2, 1, 28, 28)
Y = net(X)

print("output shape:", shape(Y))
print(net)

assert shape(Y) == (2, 10)"""
        ),
        md(
            """## 7.6.2 Trace Shapes Layer by Layer

Shape tracing is the most important LeNet skill. Without it, the first dense layer is a guess.

The expected spatial story is:

```text
28 by 28 input
Conv5 padding2 -> 28 by 28
AvgPool2 stride2 -> 14 by 14
Conv5 no padding -> 10 by 10
AvgPool2 stride2 -> 5 by 5
Flatten with 16 channels -> 16 * 5 * 5 = 400 features
```

The first dense layer only works because its input feature count is 400. If any earlier padding, kernel, or stride changes, that number changes too.

The trace is a form of architectural proof. It shows that the tensor produced by each stage is the tensor the next stage expects.
"""
        ),
        code(
            """net = make_lenet()
rows = trace_shapes(net, torch.zeros(1, 1, 28, 28))

for idx, name, out_shape in rows:
    print(f"{idx:2d} {name:10s} -> {out_shape}")

assert rows[0][2] == (1, 6, 28, 28)
assert rows[2][2] == (1, 6, 14, 14)
assert rows[3][2] == (1, 16, 10, 10)
assert rows[5][2] == (1, 16, 5, 5)
assert rows[6][2] == (1, 400)
assert rows[-1][2] == (1, 10)"""
        ),
        md(
            """## 7.6.3 Count Parameters by Layer

Parameter counting tells you where model capacity lives.

For convolution:

```text
output channels * input channels * kernel height * kernel width
plus bias values
```

For dense layers:

```text
output features * input features
plus bias values
```

LeNet's dense layers contain many parameters because flattening creates a 400-feature vector and then connects it densely. This is one reason modern CNNs often use global average pooling or other designs to reduce dense classifier size.

Counting parameters is not performance worship. It is a way to reason about capacity, memory, overfitting risk, and where updates will occur.
"""
        ),
        code(
            """net = make_lenet()
total = 0

for idx, layer in enumerate(net):
    params = sum(p.numel() for p in layer.parameters())
    if params:
        print(f"{idx:2d} {type(layer).__name__:10s} params={params}")
    total += params

print("total parameters:", total)

expected = (
    6 * 1 * 5 * 5 + 6
    + 16 * 6 * 5 * 5 + 16
    + 120 * (16 * 5 * 5) + 120
    + 84 * 120 + 84
    + 10 * 84 + 10
)

assert total == expected"""
        ),
        md(
            """## 7.6.4 One Synthetic Training Step

This cell is a wiring proof, not an accuracy experiment.

It checks the end-to-end training contract:

```text
image batch -> logits
logits plus integer labels -> scalar classification loss
loss.backward -> gradients on parameters
optimizer.step -> parameter values change
```

The labels are synthetic and the images are random, so the loss value has no real-world meaning. That restraint is intentional. A full dataset experiment would introduce data loading, metrics, train/test splits, regularization, and comparison baselines. Here we isolate the CNN mechanics.

This is the chapter-level version of rigor: prove the model is trainable before asking whether it is useful.
"""
        ),
        code(
            """torch.manual_seed(0)
net = make_lenet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

X = torch.randn(4, 1, 28, 28)
y = torch.tensor([0, 1, 2, 3])

first_weight = net[0].weight
before = first_weight.detach().clone()

logits = net(X)
loss = loss_fn(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

after = first_weight.detach().clone()

print("logits shape:", shape(logits))
print("loss:", float(loss.detach()))
print("first conv grad shape:", shape(first_weight.grad))
print("first conv changed:", not torch.allclose(before, after))

assert shape(logits) == (4, 10)
assert loss.ndim == 0
assert shape(first_weight.grad) == shape(first_weight)
assert not torch.allclose(before, after)"""
        ),
        md(
            """## 7.6.5 Prediction Mechanics

The network returns logits: raw class scores. `CrossEntropyLoss` expects logits because it internally handles the softmax-like normalization in a numerically stable way.

For prediction, `argmax(dim=1)` selects the highest-scoring class for each example. This does not require converting to probabilities first because softmax preserves score order.

The axis matters:

```text
dim=0 would compare across batch examples
dim=1 compares across classes for each example
```

That is why the predicted label tensor has shape `(batch,)`.
"""
        ),
        code(
            """with torch.no_grad():
    logits = net(torch.randn(3, 1, 28, 28))
    predicted = logits.argmax(dim=1)

print("logits shape:", shape(logits))
print("predicted labels:", predicted)

assert shape(logits) == (3, 10)
assert shape(predicted) == (3,)"""
        ),
        md(
            """## 7.6.6 Break It Deliberately: Wrong Flattened Size

This is one of the most common CNN mistakes.

The convolution and pooling stack produces `(batch, 16, 5, 5)`. `Flatten` converts that to `(batch, 400)`. If the next dense layer expects 256 features instead of 400, the model fails at matrix multiplication.

The conceptual mistake is losing track of the representation handoff:

```text
spatial feature map -> flattened vector -> dense layer
```

The dense layer does not know the spatial history. It only receives a vector of a certain length. Your architecture must make that length correct.
"""
        ),
        code(
            """bad_net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120),  # wrong: actual flattened size is 16 * 5 * 5
)

try:
    bad_net(torch.zeros(1, 1, 28, 28))
except RuntimeError as err:
    print(type(err).__name__)
    print(str(err).splitlines()[0])
else:
    raise AssertionError("The wrong flattened size should have failed.")"""
        ),
        md(
            """## 7.6.7 What This Chapter Does Not Do

This chapter does not run a full dataset experiment, compare MLP against CNN accuracy, tune regularization, save curves, or write a custom optimizer. Those belong after Chapters 5-7 are mechanically solid.

The chapter goal is conceptual and mechanical fluency:

- why convolution is a good bias for image-like data
- how channels and spatial dimensions flow
- how pooling trades detail for summary
- why LeNet's dense layer expects 400 features
- how logits, loss, gradients, and updates connect in one CNN step

That is the handoff: after this chapter, a larger CNN training exercise can focus on data, evaluation, baselines, and training discipline because the layer mechanics are no longer mysterious.
"""
        ),
        checkpoint(
            "7.6",
            [
                "How does LeNet change the representation from image grid to class scores?",
                "Why is the flattened LeNet feature size `16 * 5 * 5` for 28 by 28 inputs?",
                "Why do dense layers contain many of LeNet's parameters?",
                "What is the difference between logits and probabilities?",
                "What did the synthetic training step prove, and what did it not prove?",
                "Why does the wrong dense-layer input size fail only after `Flatten`?",
            ],
        ),
    ]
    write_nb("Chapter 7.6 - Convolutional Neural Networks (LeNet).ipynb", cells)


def main() -> None:
    build_71()
    build_72()
    build_73()
    build_74()
    build_75()
    build_76()


if __name__ == "__main__":
    main()
