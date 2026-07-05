"""Generate Chapter 7 notebooks for the D2L rewrite project."""

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
    base = "import torch\nfrom torch import nn"
    return code(base + ("\n" + extra.strip() if extra.strip() else ""))


def build_71() -> None:
    cells = [
        title_cell(
            "Chapter 7.1 - From Fully Connected Layers to Convolutions",
            "Convolutional networks are neural networks that use the spatial structure of images instead of treating every pixel as an unrelated feature.",
        ),
        imports_cell(),
    ]

    cells += section(
        "7.1.1 Invariance",
        """Invariance means a model's answer should stay the same when an irrelevant change happens.

For images, the exact location of a useful pattern often changes. A vertical edge may appear on the left, center, or right, but it is still a vertical edge.""",
        """Fully connected layers learn separate weights for every input position. That makes them inefficient when the same pattern can appear at many positions.""",
        [
            ("md", "A tiny detector can be reused at several positions."),
            ("code", """image = [0, 1, 1, 0, 1]
pattern = [1, 1]
hits = []
for i in range(len(image) - 1):
    hits.append(image[i:i + 2] == pattern)
hits"""),
        ],
        """`image` is a one-dimensional toy image.

`pattern` is the local feature we want to detect.

The loop moves the same pattern detector across positions.

The result records where the pattern appears.""",
        """CNNs use this idea in two dimensions: the same small filter is applied across many image locations.""",
        """- Invariance does not mean ignoring all location information.
- It means small shifts should not force learning a brand-new detector.
- A detector can be local even when the final prediction is global.
- Pooling and later layers help turn local detections into stable predictions.""",
    )

    cells += section(
        "7.1.2 Constraining the MLP",
        """Constraining an MLP means deliberately limiting which weights it can use.

Instead of connecting every pixel to every hidden unit, we can force each hidden unit to look only at a small neighborhood and reuse the same weights everywhere.""",
        """Images have local structure. A nearby pixel usually matters more than a far-away pixel for detecting edges, corners, and textures.""",
        [
            ("md", "Compare parameter counts for a tiny image."),
            ("code", """height, width = 4, 4
hidden_units = 3
mlp_params = height * width * hidden_units
kernel_params = 2 * 2
mlp_params, kernel_params"""),
            ("md", "The local kernel uses far fewer weights."),
            ("code", """saving = mlp_params - kernel_params
saving"""),
        ],
        """A 4 by 4 image has 16 pixel positions.

A fully connected hidden layer with 3 units needs 48 weights.

A 2 by 2 reusable kernel needs only 4 weights.

The same kernel can scan many local neighborhoods.""",
        """Convolutional layers are constrained linear layers with local connectivity and weight sharing.""",
        """- Constraints are not weakness by default.
- Good constraints encode useful assumptions.
- CNNs assume local patterns matter and can repeat across space.
- Bad constraints can hurt when the assumption is false.""",
    )

    cells += section(
        "7.1.3 Convolutions",
        """A convolutional layer applies a small matrix of weights, called a kernel or filter, over local windows of an image.

In deep learning libraries, the operation is usually cross-correlation: the kernel is not flipped before sliding.""",
        """Convolutions let the model learn local feature detectors while sharing the same detector across the whole image.""",
        [
            ("md", "Compute one local response by hand."),
            ("code", """window = [[1, 2], [3, 4]]
kernel = [[1, 0], [0, -1]]
score = 0
for r in range(2):
    for c in range(2):
        score += window[r][c] * kernel[r][c]
score"""),
        ],
        """`window` is the local image patch.

`kernel` contains learned weights.

Each aligned pair is multiplied.

The products are added into one output value.""",
        """PyTorch's `nn.Conv2d` performs this sliding local weighted sum across batches, channels, height, and width.""",
        """- Deep learning often says convolution while computing cross-correlation.
- The kernel values are learned during training.
- One output number summarizes one local window.
- The same kernel is reused at many positions.""",
    )

    cells += section(
        "7.1.4 Channels",
        """A channel is a separate measurement at each spatial location.

A grayscale image has one channel. A color image commonly has red, green, and blue channels. Hidden CNN layers also have channels, but those channels represent learned feature maps.""",
        """Channels let a model store multiple kinds of evidence at the same spatial location.""",
        [
            ("md", "Create a tiny color-like image tensor."),
            ("code", """X = torch.zeros(3, 2, 2)
X[0] += 1
X[1] += 2
X[2] += 3
X.shape, X[:, 0, 0]"""),
        ],
        """The tensor shape is channels, height, width.

Channel 0 could represent red-like values.

Channel 1 and 2 store different measurements.

At location `(0, 0)`, three channel values are available.""",
        """CNNs treat channels as feature dimensions attached to each pixel or hidden spatial location.""",
        """- Channels are not extra image rows.
- A convolution kernel usually spans all input channels.
- Output channels are learned feature maps.
- Shape order differs across libraries, so always inspect it.""",
    )

    cells += section(
        "7.1.5 Summary and Discussion",
        """CNNs combine local connectivity, weight sharing, and channels.

These choices match image structure better than flattening every pixel immediately.""",
        """The design makes image models more parameter-efficient and better at reusing local feature detectors.""",
        [
            ("md", "Represent the three CNN assumptions as plain strings."),
            ("code", """assumptions = [
    "nearby pixels are related",
    "useful patterns repeat across positions",
    "channels store multiple measurements",
]
assumptions"""),
        ],
        """The first assumption motivates local windows.

The second motivates shared kernels.

The third motivates multiple feature maps.

Together they define the core CNN bias.""",
        """Modern vision models still rely heavily on these ideas, even when they use much deeper architectures.""",
        """- CNNs are not only for photographs.
- Any grid-like data may benefit from convolution.
- Flattening too early discards spatial organization.
- Later dense layers can still use high-level CNN features.""",
    )

    cells += section(
        "7.1.6 Exercises",
        """These exercises check the basic ideas behind convolutions.""",
        """Before writing larger CNNs, you should be able to reason about locality, sharing, and channels with tiny examples.""",
        [
            ("md", "Exercise 1: count values in a 3-channel 2 by 2 image."),
            ("code", """channels, height, width = 3, 2, 2
total_values = channels * height * width
total_values"""),
            ("md", "Exercise 2: count weights in two 3 by 3 kernels."),
            ("code", """input_channels = 3
output_channels = 2
weights = output_channels * input_channels * 3 * 3
weights"""),
        ],
        """Exercise 1 multiplies channels by spatial size.

Exercise 2 multiplies output channels, input channels, and kernel area.

This is the starting point for understanding CNN parameter counts.""",
        """Model summaries report convolutional parameter counts using this same shape logic.""",
        """- Output channels mean number of learned kernels.
- Input channels are consumed by each kernel.
- Kernel height and width define local window size.
- Bias terms add one extra value per output channel if enabled.""",
    )

    write_nb("Chapter 7.1 - From Fully Connected Layers to Convolutions.ipynb", cells)


def build_72() -> None:
    cells = [
        title_cell(
            "Chapter 7.2 - Convolutions for Images",
            "This notebook makes the image convolution operation concrete using tiny matrices before mapping it to PyTorch layers.",
        ),
        imports_cell(),
    ]

    cells += section(
        "7.2.1 The Cross-Correlation Operation",
        """Cross-correlation slides a kernel over an input and computes a weighted sum at each valid location.

Valid location means the whole kernel fits inside the input without going outside the border.""",
        """This operation turns local image patches into feature responses, which is the core computation of a convolutional layer.""",
        [
            ("md", "Implement two-dimensional cross-correlation from scratch."),
            ("code", """def corr2d(X, K):
    h, w = len(K), len(K[0])
    out = []
    for i in range(len(X) - h + 1):
        row = []
        for j in range(len(X[0]) - w + 1):
            row.append(sum(X[i+a][j+b] * K[a][b] for a in range(h) for b in range(w)))
        out.append(row)
    return out"""),
            ("md", "Apply it to a tiny image."),
            ("code", """X = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
K = [[1, 0], [0, -1]]
corr2d(X, K)"""),
        ],
        """`corr2d` reads the kernel height and width.

The outer loops choose top-left window positions.

The inner expression multiplies matching input and kernel entries.

Each window produces one output value.""",
        """PyTorch uses highly optimized kernels for this same mathematical pattern.""",
        """- Cross-correlation does not flip the kernel.
- Output size shrinks when no padding is used.
- Every output value sees only one local window.
- Tiny manual code is for learning, not speed.""",
    )

    cells += section(
        "7.2.2 Convolutional Layers",
        """A convolutional layer stores one or more learnable kernels.

During training, gradient descent changes the kernel values so useful patterns produce useful feature maps.""",
        """Manual kernels are good for explanation, but a model needs trainable kernels that update from data.""",
        [
            ("md", "Create a minimal PyTorch convolutional layer."),
            ("code", """conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
X = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
Y = conv(X)
Y.shape"""),
            ("md", "Inspect the stored kernel shape."),
            ("code", """conv.weight.shape"""),
        ],
        """`1, 1` means one input channel and one output channel.

`kernel_size=2` means a 2 by 2 local window.

The input shape is batch, channels, height, width.

The output contains one learned response map.""",
        """`nn.Conv2d` is the standard PyTorch module for two-dimensional image convolution.""",
        """- The batch dimension is separate from channels.
- The layer stores parameters in `weight`.
- Kernel values start from initialization, not from hand-picked edge filters.
- Bias can be disabled for simpler examples.""",
    )

    cells += section(
        "7.2.3 Object Edge Detection in Images",
        """An edge is a sharp change in pixel values.

A simple vertical-edge kernel can respond strongly when the left side of a window differs from the right side.""",
        """Edges are useful low-level visual features. Early CNN layers often learn edge-like detectors automatically.""",
        [
            ("md", "Detect a vertical edge in a tiny image."),
            ("code", """X = [[1, 1, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 0, 0]]
K = [[1, -1]]
corr2d(X, K)"""),
        ],
        """The image has bright values on the left and dark values on the right.

The kernel subtracts the right pixel from the left pixel.

Large positive responses appear where brightness drops.

This is a handcrafted feature detector.""",
        """CNNs learn many such detectors from data instead of requiring humans to design every kernel.""",
        """- Edge detection is only an example, not all a CNN does.
- The sign of the response depends on kernel direction.
- Horizontal and vertical edges need different kernels.
- Learned filters can become more complex than simple edges.""",
    )

    cells += section(
        "7.2.4 Learning a Kernel",
        """Learning a kernel means changing its values so the produced feature map matches a target behavior.

The kernel is just a small parameter tensor, so ordinary gradient descent can update it.""",
        """Hand-designed kernels do not scale to real tasks. Learning lets the model discover useful local patterns from examples.""",
        [
            ("md", "Set up a tiny learnable kernel and loss."),
            ("code", """X = torch.tensor([[[[1., 1., 0., 0.]]]])
target = torch.tensor([[[[0., 1., 0.]]]])
conv = nn.Conv2d(1, 1, (1, 2), bias=False)
pred = conv(X)
loss = ((pred - target) ** 2).mean()
loss.backward()
conv.weight.grad"""),
        ],
        """`conv.weight` is the learnable kernel.

`pred` is the feature map produced by the current kernel.

The loss measures mismatch with the target response.

`backward()` computes gradients for changing the kernel.""",
        """Training CNNs uses the same autograd and optimizer machinery as MLPs; only the layer computation changes.""",
        """- A kernel is learned through gradients.
- The target here is artificial to keep the example tiny.
- Real training uses many images and labels.
- Gradients say how to change weights, not the final values directly.""",
    )

    cells += section(
        "7.2.5 Cross-Correlation and Convolution",
        """Mathematical convolution flips the kernel before sliding it.

Deep learning libraries usually implement cross-correlation but call the layer convolution because the learnable kernel can adapt either way.""",
        """This naming mismatch appears in almost every CNN library and textbook, so it should be understood early.""",
        [
            ("md", "Show a one-dimensional flip."),
            ("code", """kernel = [1, 2, 3]
flipped = list(reversed(kernel))
kernel, flipped"""),
            ("md", "Learnable kernels make the naming issue harmless."),
            ("code", """message = "A learned kernel can store either orientation."
message"""),
        ],
        """A true convolution would use the flipped kernel.

Cross-correlation uses the kernel as written.

If the kernel is learned, the model can learn the needed orientation.

That is why `Conv2d` remains the standard name.""",
        """When reading papers, assume `convolution` in neural networks usually means the deep-learning convention unless stated otherwise.""",
        """- The distinction matters for manual math.
- It usually does not matter for learned CNN layers.
- PyTorch uses cross-correlation behavior.
- Be precise when implementing from scratch.""",
    )

    cells += section(
        "7.2.6 Feature Map and Receptive Field",
        """A feature map is the output grid produced by a kernel.

A receptive field is the region of the input that can affect one output value.""",
        """These terms describe what information each hidden value contains and how local evidence grows through layers.""",
        [
            ("md", "A 2 by 2 kernel sees four input positions."),
            ("code", """kernel_height, kernel_width = 2, 2
receptive_field = kernel_height * kernel_width
receptive_field"""),
            ("md", "A feature map stores one response per window."),
            ("code", """input_size = 4
kernel_size = 2
output_size = input_size - kernel_size + 1
output_size"""),
        ],
        """The receptive field area is the kernel area for one layer.

With no padding and stride 1, a 4-wide input becomes a 3-wide output.

Each output location summarizes a different local window.

Deeper layers can see larger input regions indirectly.""",
        """Architecture design often controls receptive field size through kernel size, depth, stride, and pooling.""",
        """- Feature map means output activations, not the original image.
- Receptive field grows across layers.
- A larger receptive field is not always better.
- Border handling affects which input positions are seen.""",
    )

    cells += section(
        "7.2.7 Summary",
        """Image convolution turns small local windows into feature responses.

Kernels can be handcrafted for explanation or learned as model parameters.""",
        """The operation gives CNNs their core ability to detect reusable local patterns.""",
        [
            ("md", "Summarize the core vocabulary."),
            ("code", """terms = {
    "kernel": "small weight grid",
    "feature map": "grid of responses",
    "receptive field": "input region seen",
}
terms"""),
        ],
        """The dictionary maps each term to a plain-English meaning.

These meanings are enough to read most beginner CNN code.

Later chapters add deeper architecture patterns.

The core operation remains local weighted summation.""",
        """CNN debugging often starts by checking kernel shapes, output shapes, and feature map sizes.""",
        """- Do not confuse kernel shape with output shape.
- Do not confuse channels with batch size.
- Manual correlation explains the math but is not efficient.
- Learned kernels remove the need to hand-design every feature.""",
    )

    cells += section(
        "7.2.8 Exercises",
        """These exercises practice cross-correlation and output-size reasoning.""",
        """Shape fluency prevents many CNN implementation errors.""",
        [
            ("md", "Exercise 1: compute a tiny output shape."),
            ("code", """input_height = 5
kernel_height = 3
output_height = input_height - kernel_height + 1
output_height"""),
            ("md", "Exercise 2: inspect Conv2d output shape."),
            ("code", """conv = nn.Conv2d(1, 2, kernel_size=3)
X = torch.zeros(4, 1, 5, 5)
conv(X).shape"""),
        ],
        """Exercise 1 uses the no-padding, stride-1 formula.

Exercise 2 adds batch size and output channels.

The spatial size shrinks from 5 to 3.

The channel count becomes 2.""",
        """These calculations appear constantly when stacking CNN layers.""",
        """- Batch size does not change because of convolution.
- Output channels come from the number of kernels.
- Spatial size depends on kernel, padding, and stride.
- Always test shapes with tiny tensors.""",
    )

    write_nb("Chapter 7.2 - Convolutions for Images.ipynb", cells)


def build_73() -> None:
    cells = [
        title_cell(
            "Chapter 7.3 - Padding and Stride",
            "Padding and stride control how a convolution moves across an input and how large the output feature map becomes.",
        ),
        imports_cell(),
    ]

    cells += section(
        "7.3.1 Padding",
        """Padding adds extra values around the border of an input before convolution.

Zero padding means the added border values are zeros.""",
        """Without padding, feature maps shrink after each convolution. Padding can preserve spatial size and let border pixels influence more outputs.""",
        [
            ("md", "Pad a tiny tensor with zeros."),
            ("code", """X = torch.tensor([[1., 2.], [3., 4.]])
padded = torch.zeros(4, 4)
padded[1:3, 1:3] = X
padded"""),
            ("md", "Use PyTorch padding in a convolution."),
            ("code", """conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.zeros(1, 1, 5, 5)
conv(X).shape"""),
        ],
        """The manual example places a 2 by 2 tensor inside a zero border.

`padding=1` adds one border cell on each side.

A 3 by 3 kernel with padding 1 preserves 5 by 5 spatial size.

The batch and channel dimensions remain separate.""",
        """Padding is standard in CNNs because it controls spatial resolution across layers.""",
        """- Padding adds artificial border values.
- Padding can preserve size but does not create new real information.
- Border behavior matters in small images.
- PyTorch padding is specified per spatial dimension or as one shared value.""",
    )

    cells += section(
        "7.3.2 Stride",
        """Stride is the step size used when sliding a kernel.

Stride 1 moves one pixel at a time. Stride 2 skips every other possible position.""",
        """Stride reduces spatial size and computation. It is one way to downsample feature maps.""",
        [
            ("md", "Collect positions visited by stride 2."),
            ("code", """positions = []
for i in range(0, 5 - 2 + 1, 2):
    positions.append(i)
positions"""),
            ("md", "Use stride in a PyTorch convolution."),
            ("code", """conv = nn.Conv2d(1, 1, kernel_size=2, stride=2)
X = torch.zeros(1, 1, 6, 6)
conv(X).shape"""),
        ],
        """The loop starts at 0 and advances by 2.

Some possible stride-1 positions are skipped.

The PyTorch example maps 6 by 6 to 3 by 3.

The output has fewer spatial positions.""",
        """CNNs use stride to reduce memory and gradually build higher-level representations over larger input regions.""",
        """- Stride is not the same as kernel size.
- Larger stride loses some spatial detail.
- Stride affects output shape, not parameter count.
- Downsampling should be deliberate.""",
    )

    cells += section(
        "7.3.3 Summary and Discussion",
        """Padding controls borders. Stride controls movement.

Together with kernel size, they determine output spatial shape.""",
        """CNN architecture design depends on controlling how quickly feature maps shrink.""",
        [
            ("md", "Compute one output-size formula."),
            ("code", """input_size = 7
kernel = 3
padding = 1
stride = 2
out = (input_size + 2 * padding - kernel) // stride + 1
out"""),
        ],
        """The formula starts with the padded input size.

It subtracts the kernel size.

Integer division counts stride steps that fit.

Adding 1 includes the first window position.""",
        """Frameworks use this shape logic internally when building convolution outputs.""",
        """- Off-by-one shape mistakes are common.
- Padding changes spatial size but not learned kernel count.
- Stride changes spatial size but not learned kernel count.
- Test with tiny tensors before building large models.""",
    )

    cells += section(
        "7.3.4 Exercises",
        """These exercises practice padding and stride shape calculations.""",
        """Being able to predict output shape makes CNN debugging much easier.""",
        [
            ("md", "Exercise 1: preserve size with a 3 by 3 kernel."),
            ("code", """input_size = 8
kernel = 3
padding = 1
out = input_size + 2 * padding - kernel + 1
out"""),
            ("md", "Exercise 2: downsample with stride 2."),
            ("code", """conv = nn.Conv2d(1, 1, 3, padding=1, stride=2)
X = torch.zeros(1, 1, 8, 8)
conv(X).shape"""),
        ],
        """Exercise 1 uses stride 1, so the simplified formula preserves size.

Exercise 2 uses PyTorch to confirm the formula with stride 2.

The output is smaller because the kernel moves in larger steps.

The channel count remains one.""",
        """These checks are useful before connecting convolutional layers to dense layers.""",
        """- Shape formulas use spatial dimensions only.
- Batch size is unchanged.
- Output channels are chosen by the layer.
- Integer division appears when stride is greater than 1.""",
    )

    write_nb("Chapter 7.3 - Padding and Stride.ipynb", cells)


def build_74() -> None:
    cells = [
        title_cell(
            "Chapter 7.4 - Multiple Input and Multiple Output Channels",
            "Real CNN layers usually consume multiple input channels and produce multiple output channels.",
        ),
        imports_cell(),
    ]

    cells += section(
        "7.4.1 Multiple Input Channels",
        """With multiple input channels, each output value combines evidence from every channel in the local window.

The kernel has one small spatial grid per input channel.""",
        """Color images and hidden feature maps contain multiple measurements at each location, so a useful detector often needs to combine channels.""",
        [
            ("md", "Inspect a kernel for three input channels."),
            ("code", """conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2)
conv.weight.shape"""),
            ("md", "Apply it to one tiny color-like image."),
            ("code", """X = torch.zeros(1, 3, 4, 4)
Y = conv(X)
Y.shape"""),
        ],
        """The weight shape is output channels, input channels, height, width.

There is one output channel here.

The kernel spans all three input channels.

The output has one feature map per image.""",
        """Each CNN layer learns how to mix feature channels from the previous layer.""",
        """- Input channels are consumed, not preserved automatically.
- A kernel for RGB images has depth 3.
- The spatial kernel size is separate from channel count.
- Channel order must match the framework convention.""",
    )

    cells += section(
        "7.4.2 Multiple Output Channels",
        """Multiple output channels mean the layer learns multiple kernels.

Each output channel is a different learned feature map.""",
        """One detector is not enough for vision. A layer may need edge detectors, color detectors, texture detectors, and many other patterns.""",
        [
            ("md", "Create four output channels."),
            ("code", """conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
X = torch.zeros(2, 3, 8, 8)
Y = conv(X)
Y.shape"""),
            ("md", "Count weight values without bias."),
            ("code", """weights = 4 * 3 * 3 * 3
weights"""),
        ],
        """The batch has 2 images.

The input has 3 channels.

The layer learns 4 output channels.

Padding preserves the 8 by 8 spatial size.""",
        """CNN width is often described by the number of channels in each layer.""",
        """- Output channels are chosen by the model designer.
- More channels usually mean more capacity and computation.
- Bias adds one value per output channel.
- Channel count is independent of batch size.""",
    )

    cells += section(
        "7.4.3 1 x 1 Convolutional Layer",
        """A 1 x 1 convolution looks at one spatial location at a time but mixes the channel values at that location.

It does not combine neighboring pixels directly.""",
        """1 x 1 convolutions are useful for changing channel count cheaply and mixing feature information across channels.""",
        [
            ("md", "Use a 1 x 1 convolution to change channel count."),
            ("code", """conv = nn.Conv2d(3, 2, kernel_size=1)
X = torch.zeros(1, 3, 5, 5)
Y = conv(X)
Y.shape"""),
            ("md", "Count parameters without bias."),
            ("code", """params = 2 * 3 * 1 * 1
params"""),
        ],
        """The layer reads 3 channel values at each location.

It writes 2 output channel values at the same location.

The 5 by 5 spatial size is unchanged.

Only channel mixing occurs.""",
        """1 x 1 convolutions are central to many modern CNN blocks, including NiN, GoogLeNet, ResNet bottlenecks, and MobileNet-style designs.""",
        """- 1 x 1 does not mean one parameter total.
- It has input-channel times output-channel weights.
- It mixes channels, not neighboring spatial positions.
- It is often used before or after larger spatial convolutions.""",
    )

    cells += section(
        "7.4.4 Discussion",
        """Channels are the feature dimension of a CNN.

Spatial dimensions say where information is. Channel dimensions say what kind of information is stored there.""",
        """Understanding channel flow is necessary before reading modern CNN architectures.""",
        [
            ("md", "Track channel counts through a small stack."),
            ("code", """net = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(4, 8, 3, padding=1),
)
net(torch.zeros(1, 1, 6, 6)).shape"""),
        ],
        """The first convolution maps 1 channel to 4.

ReLU keeps the same shape.

The second convolution maps 4 channels to 8.

Padding keeps the spatial size at 6 by 6.""",
        """Architecture diagrams often list only channel counts and spatial sizes because these define the tensor flow.""",
        """- ReLU changes values but not shape.
- Convolutions decide output channels.
- Padding and stride decide spatial size.
- Shape tracking is a core CNN skill.""",
    )

    cells += section(
        "7.4.5 Exercises",
        """These exercises practice channel counting.""",
        """Most CNN shape mistakes come from confusing batch, channel, height, and width.""",
        [
            ("md", "Exercise 1: count weights in a 5-output-channel layer."),
            ("code", """out_ch, in_ch, k = 5, 3, 3
weights = out_ch * in_ch * k * k
weights"""),
            ("md", "Exercise 2: inspect a 1 x 1 convolution weight shape."),
            ("code", """layer = nn.Conv2d(8, 4, kernel_size=1)
layer.weight.shape"""),
        ],
        """Exercise 1 multiplies output channels, input channels, and kernel area.

Exercise 2 shows the same rule for a 1 x 1 kernel.

The spatial kernel area is one.

The channel mixing still has many weights.""",
        """These calculations prepare for modern CNN block designs.""",
        """- Always identify input channels before creating a Conv2d layer.
- Output channels become input channels for the next layer.
- Kernel size does not include batch size.
- Bias count equals output channels if bias is enabled.""",
    )

    write_nb("Chapter 7.4 - Multiple Input and Multiple Output Channels.ipynb", cells)


def build_75() -> None:
    cells = [
        title_cell(
            "Chapter 7.5 - Pooling",
            "Pooling summarizes small spatial neighborhoods, usually to reduce size and make features less sensitive to tiny shifts.",
        ),
        imports_cell(),
    ]

    cells += section(
        "7.5.1 Maximum Pooling and Average Pooling",
        """Maximum pooling takes the largest value in each window.

Average pooling takes the mean value in each window.""",
        """Pooling can make feature maps smaller and can make detections less sensitive to exact pixel position.""",
        [
            ("md", "Compute max and average of one window."),
            ("code", """window = torch.tensor([[1., 3.], [2., 0.]])
max_value = window.max()
avg_value = window.mean()
max_value, avg_value"""),
            ("md", "Apply max pooling to a tiny feature map."),
            ("code", """pool = nn.MaxPool2d(kernel_size=2)
X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
pool(X)"""),
        ],
        """The first example summarizes one 2 by 2 window.

Max pooling keeps the strongest response.

Average pooling keeps the average response.

The PyTorch example pools non-overlapping 2 by 2 windows.""",
        """Pooling layers have no learned weights but strongly affect feature-map size and information flow.""",
        """- Pooling is not convolution because it has no learned kernel weights.
- Max pooling keeps strong activations.
- Average pooling smooths values.
- Pooling can discard precise location information.""",
    )

    cells += section(
        "7.5.2 Padding and Stride",
        """Pooling also uses window size, padding, and stride.

By default, PyTorch max pooling often uses stride equal to the pooling window size when stride is not specified.""",
        """These settings control how aggressively pooling reduces spatial size.""",
        [
            ("md", "Compare pooling with stride 1 and stride 2."),
            ("code", """X = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
pool1(X).shape, pool2(X).shape"""),
            ("md", "Use padding in pooling."),
            ("code", """pool = nn.MaxPool2d(2, stride=1, padding=1)
pool(X).shape"""),
        ],
        """Stride 1 moves the pooling window one cell at a time.

Stride 2 skips more positions and produces a smaller output.

Padding adds border values before pooling.

The spatial output size follows the same style of formula as convolution.""",
        """CNNs often use pooling to reduce resolution between convolutional blocks.""",
        """- Pooling stride controls downsampling.
- Padding in pooling affects border behavior.
- Pooling has no trainable parameters.
- Pooling output shape still needs careful checking.""",
    )

    cells += section(
        "7.5.3 Multiple Channels",
        """Pooling is applied separately to each channel.

It summarizes spatial neighborhoods inside a channel but does not mix different channels together.""",
        """This preserves the meaning of each feature map while reducing its spatial size.""",
        [
            ("md", "Pool a two-channel tensor."),
            ("code", """X = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
pool = nn.MaxPool2d(2)
Y = pool(X)
Y.shape"""),
            ("md", "The channel count is unchanged."),
            ("code", """X.shape[1], Y.shape[1]"""),
        ],
        """The input has 2 channels.

Pooling reduces 4 by 4 to 2 by 2.

The channel count remains 2.

Each channel is pooled independently.""",
        """Convolutions mix channels; pooling usually only reduces spatial dimensions.""",
        """- Pooling does not choose output channels.
- Pooling does not learn channel mixtures.
- Channel count stays the same.
- Spatial size can shrink dramatically.""",
    )

    cells += section(
        "7.5.4 Summary",
        """Pooling summarizes spatial neighborhoods with fixed operations such as max or mean.

It changes feature-map size and sensitivity to small shifts, but it does not learn weights.""",
        """Pooling is a simple tool for controlling spatial resolution in CNNs.""",
        [
            ("md", "A compact pooling checklist."),
            ("code", """pooling_facts = [
    "no learned weights",
    "spatial summary",
    "channels unchanged",
    "size depends on window and stride",
]
pooling_facts"""),
        ],
        """The checklist separates pooling from convolution.

Pooling operates spatially.

It keeps channels separate.

Its output size depends on geometry settings.""",
        """Some modern CNNs reduce pooling usage, but the concept remains important for understanding classic networks.""",
        """- Pooling is not required in every CNN.
- Strided convolutions can also downsample.
- Global average pooling averages an entire feature map.
- Pooling choices affect information loss.""",
    )

    cells += section(
        "7.5.5 Exercises",
        """These exercises practice pooling behavior.""",
        """Pooling is easiest to understand by inspecting tiny tensors.""",
        [
            ("md", "Exercise 1: apply average pooling."),
            ("code", """pool = nn.AvgPool2d(kernel_size=2)
X = torch.ones(1, 1, 4, 4)
pool(X)"""),
            ("md", "Exercise 2: verify channel preservation."),
            ("code", """X = torch.zeros(3, 5, 6, 6)
Y = nn.MaxPool2d(2)(X)
Y.shape"""),
        ],
        """Exercise 1 keeps ones because every average is one.

Exercise 2 uses batch size 3 and channel count 5.

Pooling changes 6 by 6 to 3 by 3.

The channel count stays 5.""",
        """These shape checks are useful before designing LeNet and later CNNs.""",
        """- Average pooling and max pooling summarize differently.
- Batch size is unchanged.
- Channel count is unchanged.
- Spatial size changes according to pooling settings.""",
    )

    write_nb("Chapter 7.5 - Pooling.ipynb", cells)


def build_76() -> None:
    cells = [
        title_cell(
            "Chapter 7.6 - Convolutional Neural Networks (LeNet)",
            "LeNet is a classic CNN that stacks convolution, activation, pooling, and dense layers for image classification.",
        ),
        imports_cell(),
    ]

    cells += section(
        "7.6.1 LeNet",
        """LeNet processes an image in stages.

Early layers detect local visual patterns. Pooling reduces spatial size. Later dense layers combine the extracted features into class scores.""",
        """LeNet is small enough to inspect while still showing the basic CNN recipe used by much larger image models.""",
        [
            ("md", "Build a small LeNet-style network."),
            ("code", """net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(), nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(), nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))"""),
            ("md", "Check the output shape for one small batch."),
            ("code", """X = torch.zeros(2, 1, 28, 28)
net(X).shape"""),
        ],
        """The first convolution maps one image channel to six feature maps.

Pooling reduces spatial size.

The second convolution increases channels to sixteen.

Dense layers map flattened features to ten class scores.""",
        """LeNet is often used to introduce end-to-end CNN classification before modern architectures.""",
        """- LeNet originally used sigmoid activations, while modern CNNs often use ReLU.
- `Flatten` is needed before dense layers.
- The first `Linear` input size depends on earlier spatial shapes.
- Class scores are logits, not probabilities by default.""",
    )

    cells += section(
        "7.6.2 Training",
        """Training LeNet uses the same loop as earlier neural networks.

The difference is that inputs are image tensors shaped as batch, channel, height, width.""",
        """A CNN layer stack still needs a loss function, optimizer, forward pass, backward pass, and parameter update.""",
        [
            ("md", "Run one tiny synthetic training step."),
            ("code", """X = torch.randn(4, 1, 28, 28)
y = torch.tensor([0, 1, 2, 3])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
loss = loss_fn(net(X), y)
loss.backward()
optimizer.step()"""),
            ("md", "Record the scalar loss value."),
            ("code", """float(loss.detach())"""),
        ],
        """`X` is a tiny fake image batch.

`y` contains integer class labels.

The network produces one score vector per image.

Backpropagation computes gradients and the optimizer updates parameters.""",
        """Real LeNet training would use an image dataset such as Fashion-MNIST, but this notebook keeps code offline and shape-focused.""",
        """- `CrossEntropyLoss` expects raw logits.
- Labels should be integer class indices.
- Synthetic data teaches mechanics, not real accuracy.
- The training loop is not special just because the model is convolutional.""",
    )

    cells += section(
        "7.6.3 Summary",
        """LeNet combines convolutional feature extraction with dense classification.

It demonstrates the standard flow from image tensor to class scores.""",
        """Understanding LeNet makes later CNN families easier because they reuse the same pieces in deeper patterns.""",
        [
            ("md", "List the main LeNet building blocks."),
            ("code", """blocks = [
    "convolution",
    "activation",
    "pooling",
    "flatten",
    "linear classifier",
]
blocks"""),
        ],
        """The list follows the model's rough data flow.

Convolutions keep spatial structure.

Pooling reduces spatial size.

Flatten and linear layers produce final class scores.""",
        """Modern CNNs alter the block design but still rely on tensor shapes, channels, activations, and training loops.""",
        """- LeNet is a teaching model, not a modern benchmark model.
- The dense layer size must match the flattened feature size.
- Activations and pooling choices can be changed.
- Understanding shape flow is more important than memorizing the architecture.""",
    )

    cells += section(
        "7.6.4 Exercises",
        """These exercises practice inspecting a LeNet-style model.""",
        """CNN architectures become manageable when you can trace shapes layer by layer.""",
        [
            ("md", "Exercise 1: print layer names."),
            ("code", """[(i, type(layer).__name__) for i, layer in enumerate(net)]"""),
            ("md", "Exercise 2: trace shapes through the network."),
            ("code", """X = torch.zeros(1, 1, 28, 28)
shapes = []
for layer in net:
    X = layer(X)
    shapes.append(tuple(X.shape))
shapes"""),
        ],
        """Exercise 1 identifies the module sequence.

Exercise 2 runs a dummy tensor through each layer.

Each recorded shape shows how the representation changes.

This is a practical debugging method.""",
        """Shape tracing is used constantly when adapting CNNs to new image sizes or class counts.""",
        """- Dummy inputs are safe for shape inspection.
- Shape tracing does not prove training quality.
- Flatten is where spatial dimensions become one feature vector.
- If a linear layer fails, check the preceding flattened size.""",
    )

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
