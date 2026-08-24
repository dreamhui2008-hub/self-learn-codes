# Cloud Computer Vision In Practice

This is a customized Kaggle-style computer vision tutorial.

It is not a clone of Kaggle Learn. It uses the same useful rhythm:

```text
short concept
-> runnable drill
-> expected output
-> mechanical meaning
-> failure checks
-> small integration step
```

The difference is that this tutorial does not skip the uncomfortable practical layer.

You will practice:

- cloud runtime checks
- mounted dataset discovery
- path inspection
- label mapping
- image loading
- dataset objects
- transforms
- dataloaders
- model training
- metrics
- checkpoints
- artifact writing
- batch inference
- deployment-style handoff files

The point is to make Project 1 feel less mysterious before you start it.

## 0. Cloud-Only Contract

This tutorial assumes you are working in a cloud notebook.

Primary target:

```text
Kaggle Notebook
```

Secondary target:

```text
Colab or another cloud GPU notebook
```

The notebook name is:

```text
experiments.ipynb
```

The intended Kaggle filesystem contract is:

```text
/kaggle/input
    read-only mounted datasets

/kaggle/working
    writable outputs, checkpoints, plots, predictions
```

Do not download datasets to your local machine.

Do not use local VS Code for this tutorial.

## 1. What Dataset Should You Attach?

Use any Kaggle image classification dataset with this shape:

```text
some_root/
  class_a/
    image_001.jpg
    image_002.jpg
  class_b/
    image_003.jpg
    image_004.jpg
```

Good choices:

- Intel Image Classification
- Flowers Recognition
- Animal image datasets with class folders
- Any small image classification dataset with at least two class folders

The tutorial includes discovery code because dataset folders are rarely as clean as course examples pretend.

If the dataset has:

```text
train/class_name/image.jpg
test/class_name/image.jpg
```

you can use the `train` root for training and the `test` root for final scoring.

If the dataset has only:

```text
class_name/image.jpg
```

you will create your own train/validation split.

## 2. Learning Contract

You are done when you can:

- find mounted data inside a cloud notebook
- inspect real image files before training
- turn folder names into class labels
- write a dataset class from image paths
- explain `__len__` and `__getitem__`
- use transforms without losing track of shape and scale
- build dataloaders and inspect batches
- train a small CNN from real images
- save metrics and checkpoints to cloud output storage
- reload a checkpoint and prove it works
- batch-score images into `predictions.csv`
- explain the handoff files a downstream user would need

## 3. Exact Run Order

Use this order:

```text
1. Create experiments.ipynb in a Kaggle notebook.
2. Attach an image classification dataset through Kaggle's Add Input panel.
3. Run cloud runtime checks.
4. Inspect /kaggle/input.
5. Discover candidate image roots.
6. Select DATA_ROOT.
7. Build image path table.
8. Build class_to_idx mapping.
9. Load and visualize one image.
10. Build PathImageDataset.
11. Build transforms.
12. Split train/validation paths.
13. Build DataLoaders.
14. Inspect one batch.
15. Build TinyCNN.
16. Run one-batch training proof.
17. Train for a few epochs.
18. Save metrics and checkpoint.
19. Reload checkpoint.
20. Run batch inference.
21. Save predictions.csv.
22. Write handoff notes.
```

## 4. Phase 0: Runtime And Paths

### 4.1 Goal

Before modeling, prove you understand where the notebook is running and where files live.

Cloud notebooks often fail for simple reasons:

- dataset was not attached
- code uses a local path that does not exist
- outputs were written to the wrong directory
- GPU was expected but not enabled

### 4.2 Runtime Check Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from pathlib import Path
import os
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("cwd:", Path.cwd())
print("kaggle input exists:", Path("/kaggle/input").exists())
print("kaggle working exists:", Path("/kaggle/working").exists())
```

Expected pattern on Kaggle:

```text
cuda available: True or False
kaggle input exists: True
kaggle working exists: True
```

Mechanical meaning:

- `/kaggle/input` is where attached datasets appear.
- `/kaggle/working` is where saved outputs should go.
- GPU availability depends on notebook accelerator settings.

Failure checks:

- `cuda available: False`: enable GPU in notebook settings if needed.
- `/kaggle/input` missing: you are not on Kaggle; use the Colab note below or adapt paths.
- `/kaggle/input` exists but is empty: attach a dataset.

### 4.3 Cloud Path Constants

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from pathlib import Path

if Path("/kaggle/input").exists():
    INPUT_ROOT = Path("/kaggle/input")
    OUTPUT_ROOT = Path("/kaggle/working")
else:
    # Colab or other cloud notebook fallback.
    # This still keeps data and outputs in the cloud runtime, not on your local machine.
    INPUT_ROOT = Path("/content/input")
    OUTPUT_ROOT = Path("/content/working")
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("INPUT_ROOT:", INPUT_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
```

Mechanical meaning:

- all later input discovery starts from `INPUT_ROOT`
- all artifacts are written under `OUTPUT_ROOT`
- local Windows paths should not appear anywhere in this notebook

## 5. Phase 1: Dataset Mount Discovery

### 5.1 Goal

Find the actual mounted dataset path.

Kaggle examples often hide this step. Real work does not.

You need to answer:

```text
What dataset directories exist?
Where are the image files?
Do folder names represent class labels?
Is there a train/test split already?
```

### 5.2 List Mounted Inputs Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
for path in sorted(INPUT_ROOT.iterdir()):
    print(path)
```

Expected pattern:

```text
/kaggle/input/some-dataset-name
```

Failure checks:

- no output: no dataset is attached
- many outputs: choose the dataset you intend to use

### 5.3 Shallow File Tree Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def print_tree(root, max_depth=2, max_items=80):
    root = Path(root)
    count = 0
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * depth
        print(f"{indent}{path.name}/" if path.is_dir() else f"{indent}{path.name}")
        count += 1
        if count >= max_items:
            print("...")
            break


for dataset_dir in sorted(INPUT_ROOT.iterdir()):
    print("\nDATASET:", dataset_dir)
    print_tree(dataset_dir, max_depth=2, max_items=60)
```

Mechanical meaning:

- `rglob("*")` recursively walks files and folders.
- depth limiting prevents printing thousands of filenames.
- the goal is to detect whether class labels are folder names.

### 5.4 Image Root Discovery Drill

Notebook: `experiments.ipynb`.

Purpose:

Automatically find directories that look like image classification roots.

A candidate root has at least two child folders, and those child folders contain images.

Action:

Run:

```python
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def contains_image(folder):
    folder = Path(folder)
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def find_classification_roots(input_root):
    candidates = []
    for root in Path(input_root).rglob("*"):
        if not root.is_dir():
            continue

        child_dirs = [p for p in root.iterdir() if p.is_dir()]
        class_like_dirs = [p for p in child_dirs if contains_image(p)]

        if len(class_like_dirs) >= 2:
            candidates.append((root, sorted(p.name for p in class_like_dirs)))

    return candidates


candidates = find_classification_roots(INPUT_ROOT)

for i, (root, class_names) in enumerate(candidates[:20]):
    print(f"[{i}] {root}")
    print("    classes:", class_names[:10])
```

Expected pattern:

```text
[0] /kaggle/input/.../train
    classes: ['cat', 'dog', ...]
```

Mechanical meaning:

- this is not magic dataset loading
- it inspects the mounted file tree
- it guesses which folder can be treated as an image classification root

Failure checks:

- no candidates: dataset may not use class folders
- too many candidates: choose the one with `train` or the most meaningful class folders
- slow scan: huge datasets can make recursive discovery slow; inspect the tree manually and set `DATA_ROOT` yourself

### 5.5 Select Data Root

Notebook: `experiments.ipynb`.

Action:

Choose one candidate.

```python
# Change this index after inspecting the printed candidates.
DATA_ROOT = candidates[0][0]

print("DATA_ROOT:", DATA_ROOT)
print("class folders:", candidates[0][1])
```

Mechanical meaning:

- `DATA_ROOT` should be the directory whose direct child folders are class names
- every later path table is built from this selected root

Failure checks:

- `IndexError`: no candidates were found
- wrong classes: choose another candidate index

## 6. Phase 2: Paths Become Labels

### 6.1 Goal

Convert real image paths into supervised learning examples.

The training loop needs:

```text
image tensor
integer label
```

The mounted dataset gives you:

```text
file path
folder name
```

So the mechanical bridge is:

```text
path -> parent folder -> class name -> integer label
```

### 6.2 Gather Path Table Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def gather_image_items(data_root):
    data_root = Path(data_root)
    items = []

    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append((image_path, class_name))

    return items


items = gather_image_items(DATA_ROOT)

print("num images:", len(items))
print("first item:", items[0])
print("last item:", items[-1])
```

Expected pattern:

```text
num images: a positive integer
first item: (Path(...), 'some_class')
```

Mechanical meaning:

- each `item` is one supervised example before tensor conversion
- path points to the image file
- class name comes from the class folder

Failure checks:

- `num images: 0`: `DATA_ROOT` is wrong or image extensions are missing
- class names look like `train` or `images`: you selected one folder too high or too low

### 6.3 Class Mapping Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
class_names = sorted({class_name for _, class_name in items})
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

print("num classes:", len(class_names))
print("class_to_idx:", class_to_idx)
```

Mechanical meaning:

- neural networks train on integer labels, not strings
- sorting class names makes the mapping stable
- this mapping must be saved with the checkpoint for inference

Failure checks:

- one class only: dataset root is wrong or dataset is not classification-style
- class names are unstable: do not build mapping from unsorted sets alone

### 6.4 Class Count Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from collections import Counter

counts = Counter(class_name for _, class_name in items)

for class_name, count in counts.most_common():
    print(f"{class_name:25s} {count}")
```

Mechanical meaning:

- class imbalance affects accuracy interpretation
- a model can look good by overpredicting common classes
- later confusion matrices should be read with class counts in mind

## 7. Phase 3: Image Loading

### 7.1 Goal

Before building a dataset class, load one image manually.

This catches:

- broken paths
- grayscale/RGBA modes
- unexpected image sizes
- unreadable files

### 7.2 Single Image Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from PIL import Image

image_path, class_name = items[0]
image = Image.open(image_path)

print("path:", image_path)
print("class:", class_name)
print("mode:", image.mode)
print("size:", image.size)
```

Mechanical meaning:

- `image.size` is `(width, height)`, not `(height, width)`
- `image.mode` should usually become RGB for CNN training
- different images may have different sizes before transforms

### 7.3 RGB Conversion Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
image_rgb = image.convert("RGB")

print("before:", image.mode)
print("after:", image_rgb.mode)
```

Mechanical meaning:

- CNN input expects a consistent channel count
- RGB gives 3 channels
- this avoids grayscale/RGBA surprises

### 7.4 Visual Inspection Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import matplotlib.pyplot as plt

plt.imshow(image_rgb)
plt.title(class_name)
plt.axis("off")
plt.show()
```

Mechanical meaning:

- always inspect at least a few images
- file labels can be wrong
- image orientation and quality can affect training

## 8. Phase 4: Dataset Class

### 8.1 Goal

Build the dataset object that PyTorch `DataLoader` will index.

The dataset contract is:

```text
len(dataset) -> number of examples
dataset[i] -> image_tensor, label_int, path_string
```

The path string is included for error analysis and batch inference.

### 8.2 Syntax Preflight: `__len__` And `__getitem__`

Notebook: `experiments.ipynb`.

Action:

Run:

```python
class TinyListDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]


demo = TinyListDataset(["a", "b", "c"])

print(len(demo))
print(demo[0])
print(demo[2])
```

Expected output:

```text
3
a
c
```

Mechanical meaning:

- `DataLoader` repeatedly calls `dataset[idx]`
- `__len__` tells the loader how many valid indices exist
- your real dataset will replace strings with image tensors and labels

### 8.3 File-Backed Dataset Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from torch.utils.data import Dataset


class PathImageDataset(Dataset):
    def __init__(self, items, class_to_idx, transform=None):
        self.items = list(items)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, class_name = self.items[idx]

        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[class_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(image_path)
```

Mechanical meaning:

- image bytes are loaded lazily when `__getitem__` is called
- labels become integers inside `__getitem__`
- transforms are applied after image loading
- path string survives for debugging

### 8.4 Dataset Without Transform Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
raw_ds = PathImageDataset(items[:5], class_to_idx, transform=None)

sample_image, sample_label, sample_path = raw_ds[0]

print(type(sample_image))
print(sample_label)
print(sample_path)
```

Expected pattern:

```text
<class 'PIL.Image.Image'>
integer label
path string
```

Mechanical meaning:

- without transform, the dataset returns PIL images
- training will need tensors, so transforms come next

## 9. Phase 5: Transforms And Normalization

### 9.1 Goal

Transforms turn messy images into consistent model inputs.

Training needs:

```text
PIL image
-> resized/cropped image
-> tensor [3, H, W]
-> normalized tensor
```

### 9.2 Tensor Conversion Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from torchvision import transforms

to_tensor = transforms.ToTensor()
tensor_image = to_tensor(image_rgb)

print(type(tensor_image))
print(tensor_image.shape)
print(float(tensor_image.min()), float(tensor_image.max()))
```

Expected pattern:

```text
torch.Tensor
torch.Size([3, H, W])
values between 0 and 1
```

Mechanical meaning:

- PIL image uses height/width/color concepts
- PyTorch image tensor uses `[channels, height, width]`
- `ToTensor` scales byte pixels into `[0, 1]`

### 9.3 Resize Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
IMAGE_SIZE = 96

resize_to_tensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

resized = resize_to_tensor(image_rgb)

print(resized.shape)
```

Expected output:

```text
torch.Size([3, 96, 96])
```

Mechanical meaning:

- CNN batches require images to have the same height and width
- `Resize((96, 96))` forces that consistency
- 96 is small enough for fast cloud drills

### 9.4 Normalization Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

normalize_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

normalized = normalize_transform(image_rgb)

print(normalized.shape)
print(float(normalized.mean()))
print(float(normalized.std()))
```

Mechanical meaning:

- normalization shifts/scales each channel
- values no longer need to be in `[0, 1]`
- ImageNet mean/std are common defaults for RGB images

### 9.5 Unnormalize Visualization Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import torch


def unnormalize(image_tensor, mean=MEAN, std=STD):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (image_tensor.cpu() * std_t + mean_t).clamp(0, 1)


display_image = unnormalize(normalized).permute(1, 2, 0)

plt.imshow(display_image)
plt.title("unnormalized display")
plt.axis("off")
plt.show()
```

Mechanical meaning:

- normalized tensors are good for models
- unnormalized tensors are better for human inspection
- `permute(1, 2, 0)` changes `[C, H, W]` into `[H, W, C]` for matplotlib

### 9.6 Train And Eval Transforms

Notebook: `experiments.ipynb`.

Action:

Run:

```python
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

Mechanical meaning:

- training transform can be random
- evaluation transform should be deterministic
- augmentation belongs in training, not validation/test metrics

Failure checks:

- `matplotlib` image looks strange: you may be displaying normalized values directly
- batch collation fails later: image sizes may be inconsistent

## 10. Phase 6: Split And DataLoaders

### 10.1 Goal

Build train and validation dataloaders from the discovered paths.

The dataloader contract is:

```text
for images, labels, paths in loader:
    images shape: [batch, 3, 96, 96]
    labels shape: [batch]
    paths length: batch
```

### 10.2 Stratified Split Drill

Notebook: `experiments.ipynb`.

Purpose:

Split within each class so validation contains examples from every class.

Action:

Run:

```python
from collections import defaultdict
import random


def stratified_split(items, val_fraction=0.2, seed=0, max_per_class=300):
    rng = random.Random(seed)
    grouped = defaultdict(list)

    for image_path, class_name in items:
        grouped[class_name].append((image_path, class_name))

    train_items = []
    val_items = []

    for class_name, class_items in grouped.items():
        class_items = list(class_items)
        rng.shuffle(class_items)

        if max_per_class is not None:
            class_items = class_items[:max_per_class]

        val_size = max(1, int(len(class_items) * val_fraction))
        val_items.extend(class_items[:val_size])
        train_items.extend(class_items[val_size:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


train_items, val_items = stratified_split(items, val_fraction=0.2, seed=0, max_per_class=300)

print("train:", len(train_items))
print("val:", len(val_items))
```

Mechanical meaning:

- each class contributes to validation
- `max_per_class` keeps cloud drills fast
- the split uses a seed for repeatability

### 10.3 Split Count Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
train_counts = Counter(class_name for _, class_name in train_items)
val_counts = Counter(class_name for _, class_name in val_items)

print("train counts")
for class_name in class_names:
    print(f"{class_name:25s} {train_counts[class_name]}")

print("\nval counts")
for class_name in class_names:
    print(f"{class_name:25s} {val_counts[class_name]}")
```

Mechanical meaning:

- validation should not be empty for any class
- severe imbalance should be noted before interpreting accuracy

### 10.4 DataLoader Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from torch.utils.data import DataLoader

BATCH_SIZE = 32
NUM_WORKERS = 2

train_ds = PathImageDataset(train_items, class_to_idx, transform=train_transform)
val_ds = PathImageDataset(val_items, class_to_idx, transform=eval_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

images, labels, paths = next(iter(train_loader))

print("images:", images.shape)
print("labels:", labels.shape)
print("paths:", len(paths))
print("first path:", paths[0])
```

Expected pattern:

```text
images: torch.Size([32, 3, 96, 96])
labels: torch.Size([32])
paths: 32
```

Mechanical meaning:

- DataLoader stacks individual dataset outputs into batches
- string paths become a tuple/list of path strings
- `shuffle=True` belongs on training loader, not validation loader

Failure checks:

- worker crash: set `NUM_WORKERS = 0`
- inconsistent tensor shapes: transforms did not force fixed size
- CUDA pinning warning: harmless, or set `pin_memory=False`

### 10.5 Batch Visualization Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def show_batch(images, labels, n=8):
    n = min(n, images.size(0))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_list = [axes] if rows == 1 and cols == 1 else axes.ravel()

    for ax, image_tensor, label in zip(axes_list, images[:n], labels[:n]):
        display = unnormalize(image_tensor).permute(1, 2, 0)
        ax.imshow(display)
        ax.set_title(idx_to_class[int(label)])
        ax.axis("off")

    for ax in axes_list[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


show_batch(images, labels, n=8)
```

Mechanical meaning:

- this checks labels and images together
- this catches wrong class-folder assumptions
- this catches accidental normalization display errors

## 11. Phase 7: CNN Mechanics In Practice

### 11.1 Goal

Practice CNN shape mechanics on real batch-shaped tensors.

### 11.2 Convolution Shape Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from torch import nn

x = images[:4]
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
y = conv(x)

print("input:", x.shape)
print("output:", y.shape)
print("conv weight:", conv.weight.shape)
```

Expected output:

```text
input: torch.Size([4, 3, 96, 96])
output: torch.Size([4, 16, 96, 96])
conv weight: torch.Size([16, 3, 3, 3])
```

Mechanical meaning:

- output channels equal number of filters
- padding keeps spatial size when kernel is 3 and stride is 1
- conv weights are `[out_channels, in_channels, kernel_h, kernel_w]`

### 11.3 Pooling Shape Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(y)

print("before pool:", y.shape)
print("after pool:", pooled.shape)
```

Expected output:

```text
before pool: torch.Size([4, 16, 96, 96])
after pool: torch.Size([4, 16, 48, 48])
```

Mechanical meaning:

- pooling does not change channel count
- stride 2 halves spatial size
- later layers become cheaper

### 11.4 Adaptive Pooling Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
adaptive = nn.AdaptiveAvgPool2d((1, 1))
pooled_global = adaptive(pooled)
flat = pooled_global.flatten(1)

print("global pooled:", pooled_global.shape)
print("flat:", flat.shape)
```

Expected output:

```text
global pooled: torch.Size([4, 16, 1, 1])
flat: torch.Size([4, 16])
```

Mechanical meaning:

- global average pooling removes spatial dimensions
- dense classifier input depends on channels, not image size

## 12. Phase 8: Model

### 12.1 Goal

Build a small CNN that is fast enough for cloud notebook drills but real enough to train on image folders.

Architecture:

```text
ConvBNReLU
-> MaxPool
-> ConvBNReLU
-> MaxPool
-> ConvBNReLU
-> AdaptiveAvgPool
-> Linear head
```

### 12.2 Model Block Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
```

Mechanical meaning:

- Conv2d learns local features
- BatchNorm stabilizes channel activation scale
- ReLU adds nonlinearity

### 12.3 TinyCNN File-Free Model

Notebook: `experiments.ipynb`.

Action:

Run:

```python
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNReLU(3, 32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


num_classes = len(class_names)
model = TinyCNN(num_classes)

logits = model(images[:4])

print("num classes:", num_classes)
print("logits:", logits.shape)
```

Expected output:

```text
logits: torch.Size([4, num_classes])
```

Mechanical meaning:

- final output size is controlled by `num_classes`
- logits are raw class scores
- no softmax is needed before `CrossEntropyLoss`

### 12.4 Parameter Count Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("parameters:", parameter_count(model))
```

Mechanical meaning:

- parameter count tells you model capacity
- this CNN is intentionally small

## 13. Phase 9: One-Batch Training Proof

### 13.1 Goal

Before full training, prove one optimizer step changes model parameters.

### 13.2 One-Batch Update Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyCNN(num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

batch_images, batch_labels, _ = next(iter(train_loader))
batch_images = batch_images.to(device)
batch_labels = batch_labels.to(device)

first_param = next(model.parameters())
before = first_param.detach().clone()

logits = model(batch_images)
loss = loss_fn(logits, batch_labels)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

after = first_param.detach().clone()

print("logits:", logits.shape)
print("loss:", float(loss.detach()))
print("changed:", not torch.allclose(before, after))
```

Expected output:

```text
changed: True
```

Mechanical meaning:

- model and batch are moved to the same device
- logits feed into `CrossEntropyLoss`
- backward computes gradients
- optimizer mutates parameters

Failure checks:

- device mismatch: images, labels, and model must be on same device
- loss error: labels must be integer class IDs
- `changed: False`: check `loss.backward()` and `optimizer.step()`

## 14. Phase 10: Evaluation Helpers

### 14.1 Goal

Write reusable training and evaluation functions in the notebook.

This is still notebook-based, but it is structured like real project code.

### 14.2 Accuracy Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def accuracy_from_logits(logits, labels):
    predicted = logits.argmax(dim=1)
    return float((predicted == labels).float().mean().item())


demo_logits = torch.tensor([[1.0, 3.0], [5.0, 0.0]])
demo_labels = torch.tensor([1, 1])

print(accuracy_from_logits(demo_logits, demo_labels))
```

Expected output:

```text
0.5
```

Mechanical meaning:

- `argmax(dim=1)` selects a class per example
- accuracy compares predicted class IDs to labels

### 14.3 Train/Eval Helper Cell

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_images, batch_labels, _paths in loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_images)
        loss = loss_fn(logits, batch_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = batch_images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_images, batch_labels, _paths in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_images)
            loss = loss_fn(logits, batch_labels)

            batch_size = batch_images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
```

Mechanical meaning:

- training mode enables BatchNorm/Dropout training behavior
- eval mode freezes BatchNorm/Dropout behavior
- `torch.no_grad()` prevents graph construction during evaluation
- metrics are weighted by batch size

## 15. Phase 11: Artifacts

### 15.1 Goal

Save evidence from the run.

Kaggle tutorials often show final accuracy but skip artifact discipline.

This tutorial saves:

- run config
- metrics JSONL
- metrics plot
- class mapping JSON
- checkpoint
- predictions CSV
- handoff notes

### 15.2 Run Directory Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
from datetime import datetime
import json

RUN_ROOT = OUTPUT_ROOT / "cloud_cv_runs"
RUN_NAME = "tinycnn_cloud_image_folder"
RUN_DIR = RUN_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=False)

print(RUN_DIR)
```

Mechanical meaning:

- every run gets a unique output folder
- outputs live in cloud writable storage
- on Kaggle, files in `/kaggle/working` can be downloaded after the run

### 15.3 JSON And JSONL Helpers

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
```

Mechanical meaning:

- JSON stores one object
- JSONL stores one row per epoch
- JSONL is easy to append during training

### 15.4 Save Config And Label Map

Notebook: `experiments.ipynb`.

Action:

Run:

```python
run_config = {
    "data_root": str(DATA_ROOT),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "num_classes": num_classes,
    "class_names": class_names,
    "max_per_class": 300,
    "model": "TinyCNN",
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-2,
}

save_json(RUN_DIR / "run_config.json", run_config)
save_json(RUN_DIR / "class_to_idx.json", class_to_idx)

print("saved:", RUN_DIR / "run_config.json")
```

Mechanical meaning:

- inference needs the same class mapping
- future readers need the dataset root and preprocessing choices

## 16. Phase 12: Full Training Run

### 16.1 Goal

Train the small CNN for a few epochs and save metrics.

This is not a benchmark. It is a complete cloud training loop.

### 16.2 Training Loop Cell

Notebook: `experiments.ipynb`.

Action:

Run:

```python
EPOCHS = 5

model = TinyCNN(num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

metrics_path = RUN_DIR / "metrics.jsonl"
best_val_accuracy = -1.0
best_checkpoint_path = RUN_DIR / "best.pt"

for epoch in range(1, EPOCHS + 1):
    train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_metrics = evaluate(model, val_loader, loss_fn, device)

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
    }
    append_jsonl(metrics_path, row)
    print(row)

    if val_metrics["accuracy"] > best_val_accuracy:
        best_val_accuracy = val_metrics["accuracy"]
        torch.save(
            {
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "image_size": IMAGE_SIZE,
                "mean": MEAN,
                "std": STD,
                "epoch": epoch,
                "val_accuracy": best_val_accuracy,
            },
            best_checkpoint_path,
        )

print("best checkpoint:", best_checkpoint_path)
print("best val accuracy:", best_val_accuracy)
```

Mechanical meaning:

- validation decides which checkpoint is best
- checkpoint stores class mapping and preprocessing constants
- metrics are saved every epoch

Failure checks:

- training accuracy does not move: dataset may be too hard, too small, or labels may be wrong
- validation accuracy much lower than training: possible overfitting
- checkpoint missing: loop may have crashed before validation

### 16.3 Plot Metrics

Notebook: `experiments.ipynb`.

Action:

Run:

```python
history = read_jsonl(metrics_path)

epochs = [row["epoch"] for row in history]
train_acc = [row["train_accuracy"] for row in history]
val_acc = [row["val_accuracy"] for row in history]
train_loss = [row["train_loss"] for row in history]
val_loss = [row["val_loss"] for row in history]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(epochs, train_loss, label="train")
axes[0].plot(epochs, val_loss, label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].legend()

axes[1].plot(epochs, train_acc, label="train")
axes[1].plot(epochs, val_acc, label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(RUN_DIR / "metrics.png", dpi=150)
plt.show()
```

Mechanical meaning:

- plots make learning dynamics visible
- saved plot becomes a handoff artifact

## 17. Phase 13: Checkpoint Reload

### 17.1 Goal

Prove the saved model can be reused after training.

This is the first deployment-adjacent skill:

```text
train now
load later
predict without retraining
```

### 17.2 Reload Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
payload = torch.load(best_checkpoint_path, map_location=device)

loaded_model = TinyCNN(num_classes=len(payload["class_to_idx"])).to(device)
loaded_model.load_state_dict(payload["model_state"])
loaded_model.eval()

print("loaded epoch:", payload["epoch"])
print("loaded val accuracy:", payload["val_accuracy"])
print("loaded classes:", payload["idx_to_class"])
```

Mechanical meaning:

- code defines the architecture
- checkpoint provides the trained weights
- class mapping travels with the checkpoint

Failure checks:

- missing keys: architecture changed after saving
- wrong class count: class mapping does not match model head
- CPU/GPU load issue: use `map_location=device`

### 17.3 Prediction Consistency Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
eval_images, eval_labels, eval_paths = next(iter(val_loader))
eval_images_device = eval_images.to(device)

with torch.no_grad():
    logits = loaded_model(eval_images_device)
    probs = logits.softmax(dim=1)
    confidence, predicted = probs.max(dim=1)

for i in range(min(5, eval_images.size(0))):
    print({
        "path": eval_paths[i],
        "true": idx_to_class[int(eval_labels[i])],
        "pred": idx_to_class[int(predicted[i].cpu())],
        "confidence": float(confidence[i].cpu()),
    })
```

Mechanical meaning:

- logits become probabilities only for human-readable inference
- `argmax` or `max(dim=1)` selects predicted class
- confidence is the winning softmax value

## 18. Phase 14: Confusion Matrix And Errors

### 18.1 Goal

Accuracy is not enough.

You need to inspect:

- which classes are confused
- whether mistakes are visually reasonable
- whether labels look suspicious

### 18.2 Confusion Matrix Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def compute_confusion(model, loader, num_classes, device):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()

    with torch.no_grad():
        for batch_images, batch_labels, _paths in loader:
            logits = model(batch_images.to(device))
            predicted = logits.argmax(dim=1).cpu()
            labels_cpu = batch_labels.cpu()

            flat = labels_cpu * num_classes + predicted
            cm += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    return cm


cm = compute_confusion(loaded_model, val_loader, num_classes, device)
print(cm)
```

Mechanical meaning:

- rows are true classes
- columns are predicted classes
- off-diagonal cells are mistakes

### 18.3 Save Confusion Matrix Plot

Notebook: `experiments.ipynb`.

Action:

Run:

```python
fig, ax = plt.subplots(figsize=(8, 8))
image = ax.imshow(cm.numpy())
fig.colorbar(image, ax=ax)

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Validation Confusion Matrix")

plt.tight_layout()
plt.savefig(RUN_DIR / "confusion_matrix.png", dpi=150)
plt.show()
```

### 18.4 Misclassification Grid Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def collect_misclassified(model, loader, device, limit=12):
    model.eval()
    examples = []

    with torch.no_grad():
        for batch_images, batch_labels, batch_paths in loader:
            logits = model(batch_images.to(device))
            predicted = logits.argmax(dim=1).cpu()

            for image_tensor, true_label, pred_label, path in zip(batch_images, batch_labels, predicted, batch_paths):
                if int(true_label) != int(pred_label):
                    examples.append((image_tensor, int(true_label), int(pred_label), path))

                if len(examples) >= limit:
                    return examples

    return examples


mistakes = collect_misclassified(loaded_model, val_loader, device, limit=12)
print("mistakes:", len(mistakes))
```

Then:

```python
def show_mistakes(mistakes):
    if not mistakes:
        print("No mistakes collected.")
        return

    cols = 4
    rows = (len(mistakes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_list = [axes] if rows == 1 and cols == 1 else axes.ravel()

    for ax, (image_tensor, true_label, pred_label, path) in zip(axes_list, mistakes):
        display = unnormalize(image_tensor).permute(1, 2, 0)
        ax.imshow(display)
        ax.set_title(f"T: {idx_to_class[true_label]}\nP: {idx_to_class[pred_label]}")
        ax.axis("off")

    for ax in axes_list[len(mistakes):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(RUN_DIR / "misclassified.png", dpi=150)
    plt.show()


show_mistakes(mistakes)
```

Mechanical meaning:

- error analysis connects metrics back to actual images
- some errors are model failures
- some errors are ambiguous images or bad labels

## 19. Phase 15: Batch Inference

### 19.1 Goal

Deployment is not only "model trained."

A useful model must score new inputs and write outputs in a form another process or person can consume.

This phase writes:

```text
predictions.csv
```

with:

```text
image_path,predicted_class,predicted_index,confidence
```

### 19.2 Single-Image Prediction Function

Notebook: `experiments.ipynb`.

Action:

Run:

```python
def predict_one_image(model, image_path, transform, idx_to_class, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = logits.softmax(dim=1)
        confidence, predicted = probs.max(dim=1)

    pred_idx = int(predicted.item())
    return {
        "image_path": str(image_path),
        "predicted_class": idx_to_class[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(confidence.item()),
    }


print(predict_one_image(loaded_model, val_items[0][0], eval_transform, idx_to_class, device))
```

Mechanical meaning:

- inference uses eval transform, not random training transform
- `unsqueeze(0)` adds batch dimension
- output is a plain dictionary that can become a row in CSV

### 19.3 Batch Prediction Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
import csv


def predict_paths_to_csv(model, image_paths, transform, idx_to_class, device, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in image_paths:
        rows.append(predict_one_image(model, image_path, transform, idx_to_class, device))

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "predicted_class", "predicted_index", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


sample_paths = [path for path, _class_name in val_items[:25]]
prediction_rows = predict_paths_to_csv(
    loaded_model,
    sample_paths,
    eval_transform,
    idx_to_class,
    device,
    RUN_DIR / "predictions.csv",
)

print(prediction_rows[:3])
print("saved:", RUN_DIR / "predictions.csv")
```

Mechanical meaning:

- this is a batch scoring job
- scoring output is a stable file
- downstream users do not need your notebook state

Failure checks:

- predictions all same class: model may be undertrained or class imbalance is severe
- confidence always near 1.0 but wrong: model may be overconfident
- CSV missing: check `RUN_DIR`

## 20. Phase 16: Deployment/FDE-Lite Handoff

### 20.1 Goal

This is not full production deployment.

It is a file/data/execution handoff:

```text
given a trained model and a folder of images,
produce reproducible prediction artifacts
with enough metadata for another person or system to use them
```

This prepares you for later production work without introducing Docker, APIs, or CI/CD yet.

### 20.2 Handoff Bundle Drill

Notebook: `experiments.ipynb`.

Action:

Run:

```python
handoff = f"""# Cloud CV Handoff

## Run

- run_dir: {RUN_DIR}
- data_root: {DATA_ROOT}
- model: TinyCNN
- image_size: {IMAGE_SIZE}
- num_classes: {num_classes}
- best_val_accuracy: {best_val_accuracy}

## Required Files

- best.pt: model checkpoint with class mapping and preprocessing constants
- class_to_idx.json: class name to integer mapping
- run_config.json: training and preprocessing config
- metrics.jsonl: epoch-level training and validation metrics
- metrics.png: loss and accuracy plot
- confusion_matrix.png: validation confusion matrix
- misclassified.png: sampled validation mistakes
- predictions.csv: batch prediction output

## Prediction CSV Schema

- image_path: input image path
- predicted_class: predicted class name
- predicted_index: predicted class integer
- confidence: softmax confidence for predicted class

## Inference Rule

Use eval_transform, not train_transform.
The training transform includes random augmentation and should not be used for deployment scoring.
"""

(RUN_DIR / "HANDOFF.md").write_text(handoff, encoding="utf-8")
print((RUN_DIR / "HANDOFF.md").read_text(encoding="utf-8"))
```

Mechanical meaning:

- handoff notes make artifacts understandable outside the notebook
- the checkpoint alone is not enough
- consumers need preprocessing and label mapping

### 20.3 Final Artifact Checklist

Notebook: `experiments.ipynb`.

Action:

Run:

```python
for path in sorted(RUN_DIR.iterdir()):
    print(path.name)
```

Expected files:

```text
HANDOFF.md
best.pt
class_to_idx.json
confusion_matrix.png
metrics.jsonl
metrics.png
misclassified.png
predictions.csv
run_config.json
```

Failure checks:

- missing `best.pt`: training loop did not save checkpoint
- missing `predictions.csv`: batch inference was not run
- missing `HANDOFF.md`: handoff cell was not run

## 21. Final Mini Project

### 21.1 Goal

Repeat the full workflow on a dataset you choose.

You should be able to do this without new code from scratch:

```text
attach dataset
discover root
build path table
train model
save artifacts
reload checkpoint
batch-score images
write handoff
```

### 21.2 Final Project Requirements

In `experiments.ipynb`, complete:

- attach one Kaggle image dataset
- choose `DATA_ROOT`
- show class counts
- visualize one batch
- train `TinyCNN` for at least 5 epochs
- save `metrics.jsonl`
- save `metrics.png`
- save `best.pt`
- save `confusion_matrix.png`
- save `misclassified.png`
- save `predictions.csv`
- save `HANDOFF.md`

### 21.3 Reflection Questions

Answer in markdown at the end of `experiments.ipynb`.

```text
1. What was the mounted dataset path?
2. How did folder names become labels?
3. What shape did one training batch have?
4. Which transform choices affected training?
5. What did the validation curve show?
6. Which classes were confused?
7. What files are required to reuse the trained model?
8. Why is predictions.csv a deployment-adjacent artifact?
9. What would break if class_to_idx were lost?
10. What would you change before turning this into a package?
```

## 22. Done Criteria

You are done when:

- all cells in `experiments.ipynb` run on a cloud notebook
- data comes from a cloud-mounted dataset or cloud runtime source
- no local Windows path is required
- you can explain `DATA_ROOT`
- you can explain path-to-label mapping
- dataloaders return `[batch, 3, 96, 96]`
- one-batch training changes weights
- full training writes metrics
- checkpoint reload works
- batch inference writes `predictions.csv`
- handoff notes explain the output bundle

## 23. What This Prepares You For

This tutorial prepares the practical side of Project 1.

Project 1 will add:

- package source files
- CLI commands
- tests
- multiple architectures
- ablation configs
- stricter reproducibility

This tutorial focuses on the cloud notebook mechanics that Kaggle-style lessons often hide:

```text
where data lives
how labels are created
how batches are formed
where outputs go
how checkpoints are reused
how predictions are handed off
```
