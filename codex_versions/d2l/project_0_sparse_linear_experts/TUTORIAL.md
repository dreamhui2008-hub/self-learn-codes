# Project 0 Tutorial: Sparse Linear Experts Under Distribution Shift

Generated: 2026-07-20

## 0. Purpose

This project is the integration checkpoint after D2L Chapters 2-4.

The goal is not to build a big model. The goal is to make the basic pieces hard to fake:

- tensors
- shapes
- matrix multiplication
- train/test split
- loss
- gradients
- manual SGD
- softmax classification
- weight decay
- evaluation
- distribution shift
- sparse routing
- selected-parameter updates

The project should feel like a small simulator. It borrows language from the PFC/router idea and the local-learning topology, but it does not try to validate either architecture.

## 1. How To Use This Tutorial

Do not copy-paste the full project all at once.

Recommended workflow:

1. Read one phase.
2. Type the code by hand into your own files.
3. Run only the cells or commands marked as runnable.
4. Print tensor shapes.
5. Break one thing deliberately.
6. Explain in `notes.md` what broke and why.
7. Move to the next phase.

This is a tutorial/syllabus, not a finished codebase.

When you later create the project, use a separate working folder such as:

```text
project_0_sparse_linear_experts/
```

Suggested files:

```text
project_0_sparse_linear_experts/
    .venv/
    data.py
    models.py
    router.py
    train.py
    metrics.py
    experiments.ipynb
    notes.md
```

Important:

Do not type every code block into one file.

Use this rule:

- reusable functions and classes go into `.py` files
- experiment runs go into `experiments.ipynb` or small `run_phase_XX.py` scripts
- explanations, hypotheses, and results go into `notes.md`

## 1.1 Prerequisite Libraries and Virtual Environment

This project only needs a small Python environment.

Required:

- Python 3.10 or newer
- PyTorch

Recommended:

- Jupyter or VS Code notebooks
- `ipykernel` so the notebook can use the project venv
- `matplotlib` for optional plots

Not required:

- scikit-learn
- pandas
- torchvision
- CUDA
- GPU
- D2L package

Git rule:

Do not commit `.venv/`. A virtual environment contains installed packages and generated executables. It is large, machine-specific, and can contain path structures that Git has trouble indexing on Windows/WSL.

Commit this instead:

```text
requirements.txt
```

Then recreate the venv on each machine from that package list.

PyTorch's official install page currently says latest stable PyTorch requires Python 3.10 or later. Python's official `venv` docs recommend `python -m venv` for creating virtual environments.

### Option A: WSL / Bash Setup

Use this if you are working from the same Linux-style path Codex sees.

```bash
cd /mnt/c/Users/zihui/self-learn-codes
mkdir -p project_0_sparse_linear_experts
cd project_0_sparse_linear_experts
```

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install libraries:

```bash
python -m pip install -r requirements.txt
```

Register a notebook kernel:

```bash
python -m ipykernel install --user --name project0-sparse-experts --display-name "Project 0 Sparse Experts"
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.rand(2, 3)); print('cuda', torch.cuda.is_available())"
```

### Option B: Windows PowerShell Setup

Use this if you want to work from native Windows Python.

```powershell
cd C:\Users\zihui\self-learn-codes
mkdir project_0_sparse_linear_experts
cd project_0_sparse_linear_experts
```

Create the virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can avoid changing execution policy by calling the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch matplotlib jupyter ipykernel
```

If activation works, install normally:

```powershell
python -m pip install --upgrade pip
python -m pip install torch matplotlib jupyter ipykernel
```

Register a notebook kernel:

```powershell
python -m ipykernel install --user --name project0-sparse-experts --display-name "Project 0 Sparse Experts"
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.rand(2, 3)); print('cuda', torch.cuda.is_available())"
```

### Mechanical Setup Notes

Pick one environment for the project and stay inside it.

If you install PyTorch in WSL but run the notebook with Windows Python, the notebook will still say `ModuleNotFoundError: No module named 'torch'`.

If you install PyTorch in Windows PowerShell but run scripts from WSL, WSL will not see the Windows venv.

For this project, CPU PyTorch is enough. If the generic `pip install torch` command fails, use the official PyTorch install selector for your OS and compute platform:

```text
https://pytorch.org/get-started/locally/
```

Check which Python your terminal is using:

```bash
python -c "import sys; print(sys.executable)"
```

Check whether a notebook is using the right kernel:

```python
import sys
print(sys.executable)
```

Expected idea:

```text
the printed path should include project_0_sparse_linear_experts/.venv
```

## 1.2 Exact File Typing Map

Type reusable code into the project files according to this map.

```text
data.py
    make_regression_data
    train_test_split
    make_region_table
    make_region_rules
    make_sparse_regression_data
    train_test_split_with_regions
    make_region_class_rules
    make_sparse_classification_data

models.py
    predict_regression
    squared_loss
    routed_regression_loss
    routed_predict_regression
    routed_classification_logits
    top2_routed_predict_regression

router.py
    normalize_rows
    route_topk
    random_routes
    similarity_routes

train.py
    sgd
    l2_penalty
    rescale_expert_weights
    ReplayBuffer

metrics.py
    accuracy
    per_region_mse
    confusion_matrix

experiments.ipynb
    imports
    manual seed
    parameter initialization
    training loops
    evaluation blocks
    experiment tables
    plots, if you add them

notes.md
    hypotheses
    observed losses
    observed accuracies
    bugs
    explanations
    final writeup
```

Simple rule:

If the tutorial block starts with `def` or `class`, type it into the mapped `.py` file.

If the tutorial block creates data, initializes parameters, runs `for epoch in ...`, prints metrics, or records an experiment, type it into `experiments.ipynb` or a `run_phase_XX.py` script.

Run rule:

- after typing code into a `.py` file, save the file; do not run that file unless the tutorial explicitly says to
- after typing a cell into `experiments.ipynb`, run that notebook cell immediately
- if a notebook cell depends on a function you just changed in a `.py` file, restart the notebook kernel or rerun the import cell
- if the section says "Run now", execute it before moving on
- if a cell only defines variables or trains silently, no output is normal until the next evaluation or print cell

## 1.2.1 Execution Checklist

Use this checklist when you are unsure whether to run something.

```text
Setup commands
    Run in terminal.

Starter headers in data.py/models.py/router.py/train.py/metrics.py
    Type and save only.

Function definitions in .py files
    Type and save only.
    Rerun the notebook import cell after saving.

Class definitions in .py files
    Type and save only.
    Rerun the notebook import cell after saving.

experiments.ipynb cells that create tensors or variables
    Run now.

experiments.ipynb cells with for epoch loops
    Run now.
    No output is normal unless the cell contains print().

experiments.ipynb cells with print()
    Run now.
    They are usually checkpoints.

notes.md templates
    Type or paste into notes.md.
    Do not run.

Experiment tables
    Fill in by rerunning earlier training/evaluation cells with changed settings.
    They are not single code blocks unless the tutorial provides one.

Debugging snippets
    Run only when you need to inspect a bug or verify a shape.
```

Section-by-section:

```text
1.1 setup commands: run in terminal
1.3 starter file headers: type/save only
1.5 sanity_check.py: type file, then run in terminal
6.3 first imports: run in notebook
6.4 first shape checks: run in notebook
7.3-7.5 function code: type/save only
7.6 training loop: run in notebook
7.7 evaluation: run in notebook
8.2-8.4 function code: type/save only
8.6 checkpoint prints: run in notebook
8.8 function code: type/save only
9.2 data generation/training/evaluation: run in notebook
10.3 router functions: type/save only
10.4 router test: run in notebook
11.2 expert parameter init: run in notebook
11.3 routed loss: type/save only
11.4 routed training: run in notebook
11.5 routed predict function: type/save only; evaluation block: run in notebook
12.2 random/similarity route functions: type/save only; oracle route line: run inside notebook experiment
12.3 experiment table: run variants and fill notes/table
13.2 l2_penalty: type/save only; loss-composition lines: run inside modified training loops
13.3 norm check: run in notebook after each experiment
14.2-14.3 shifted data generation: run in notebook
14.4 per_region_mse: type/save only
15.2-15.3 classification data functions: type/save only
15.4 global softmax training: run in notebook; accuracy function: type/save only
15.5 routed logits function: type/save only; routed training loop: run in notebook
15.7 experiment table: run variants and fill notes/table
16.3 top2 function: type/save only
16.4 top-2 comparison: run variants in notebook
17.2 local update gate: run inside a modified routed training loop
18.2 rescale function: type/save only; rescale call: run inside a modified training loop
19.2 ReplayBuffer class: type/save only
19.3 replay comparison: run variants in notebook
20 required experiment matrix: checklist, not code
21 confusion_matrix: type/save only
22 notes template: write in notes.md
23 debugging snippets: run only when debugging
```

## 1.3 Starter File Headers

Before typing the functions, create the files and put these headers at the top.

In `data.py`:

```python
import torch
```

In `models.py`:

```python
import torch
```

In `router.py`:

```python
import torch
```

In `train.py`:

```python
import torch
```

In `metrics.py`:

```python
import torch
```

In `experiments.ipynb`, start with a safe import cell.

This cell tries to import every function in the final project map. If a function has not been typed yet, it skips that function and prints its name. That lets you keep one import cell while building the project gradually.

```python
import torch
import torch.nn.functional as F
from importlib import import_module, reload

def safe_import(module_name, names):
    try:
        module = import_module(module_name)
        module = reload(module)
    except Exception as error:
        print(f"SKIP module {module_name}: {error}")
        return

    for name in names:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)
        else:
            print(f"SKIP {module_name}.{name}: not typed yet")

safe_import("data", [
    "make_regression_data",
    "train_test_split",
    "make_region_table",
    "make_region_rules",
    "make_sparse_regression_data",
    "train_test_split_with_regions",
    "make_region_class_rules",
    "make_sparse_classification_data",
])

safe_import("models", [
    "predict_regression",
    "squared_loss",
    "routed_regression_loss",
    "routed_predict_regression",
    "routed_classification_logits",
    "top2_routed_predict_regression",
])

safe_import("router", [
    "route_topk",
    "random_routes",
    "similarity_routes",
])

safe_import("train", [
    "sgd",
    "l2_penalty",
    "rescale_expert_weights",
    "ReplayBuffer",
])

safe_import("metrics", [
    "accuracy",
    "per_region_mse",
    "confusion_matrix",
])

torch.manual_seed(0)
```

Run this cell whenever you add or change a function in a `.py` file.

Expected behavior:

```text
functions you have typed become available
functions you have not typed yet are printed as SKIP
```

If a whole module is skipped, that usually means the file has a syntax error or a top-level import error.

Important:

Do not put future-only imports at the top of a module. For example, early `models.py` should not import `route_topk` before `router.py` defines it. Later, when a model function needs `route_topk`, import it inside that function or add the import after `route_topk` exists.

## 1.4 Phase-By-Phase Typing Plan

Use this as the mechanical sequence.

```text
Phase 0
    terminal:
        create folder
        create venv
        install libraries
    experiments.ipynb:
        first imports
        first shape checks

Phase 1
    data.py:
        make_regression_data
        train_test_split
    models.py:
        predict_regression
        squared_loss
    train.py:
        sgd
    experiments.ipynb:
        global regression training loop
        global regression evaluation

Phase 2
    data.py:
        make_region_table
        make_region_rules
        make_sparse_regression_data
        train_test_split_with_regions
    experiments.ipynb:
        create region table
        create multi-region data
        print shape checks

Phase 3
    experiments.ipynb:
        global model on multi-region data
        baseline train/test metrics

Phase 4
    router.py:
        normalize_rows
        route_topk
    experiments.ipynb:
        router accuracy check

Phase 5
    models.py:
        routed_regression_loss
        routed_predict_regression
    experiments.ipynb:
        routed regression expert training loop
        routed regression evaluation

Phase 6
    router.py:
        random_routes
        similarity_routes
    experiments.ipynb:
        oracle vs random vs similarity routing table

Phase 7
    train.py:
        l2_penalty
    experiments.ipynb:
        weight decay runs
        weight norm logging

Phase 8
    metrics.py:
        per_region_mse
    experiments.ipynb:
        mixture shift
        noise shift
        per-region evaluation

Phase 9
    data.py:
        make_region_class_rules
        make_sparse_classification_data
    metrics.py:
        accuracy
    models.py:
        routed_classification_logits
    experiments.ipynb:
        global softmax classifier
        routed softmax classifier

Phase 10
    models.py:
        top2_routed_predict_regression
    experiments.ipynb:
        top-1 vs top-2 comparison

Phase 11
    experiments.ipynb:
        local update gate experiment

Phase 12
    train.py:
        rescale_expert_weights
    experiments.ipynb:
        homeostatic scaling comparison

Phase 13
    train.py:
        ReplayBuffer
    experiments.ipynb:
        replay/no-replay comparison

Final
    notes.md:
        final writeup
```

## 1.5 Minimal Sanity Script

Before using a notebook, you can test the environment with a tiny script named `sanity_check.py`.

Type this into `sanity_check.py`:

```python
import torch

X = torch.randn(4, 6)
w = torch.randn(6)
b = torch.tensor(0.5)
y_hat = X @ w + b

print("torch version:", torch.__version__)
print("X:", X.shape)
print("w:", w.shape)
print("y_hat:", y_hat.shape)
print(y_hat)
```

Run it:

```bash
python sanity_check.py
```

Expected:

```text
torch imports successfully
X is [4, 6]
w is [6]
y_hat is [4]
```

## 2. Project Mental Model

The whole system can be summarized as:

```text
input vector
-> router scores input against region table
-> router selects top-k experts
-> selected expert(s) produce prediction
-> loss compares prediction to label
-> backward computes gradients
-> optimizer updates selected parameters
-> evaluation measures whether learning generalized
```

For the first version, every expert is just a linear model.

For regression:

```text
y_hat = x @ w + b
```

For classification:

```text
logits = x @ W + b
probabilities = softmax(logits)
```

The core question:

> If different data regions follow different rules, does a routed set of small linear experts behave differently from one global linear model?

## 3. What This Project Is Not

Do not add these yet:

- MLPs
- CNNs
- transformers
- real MoE auxiliary load-balancing loss
- LoRA
- vector databases
- real hardware simulation
- FastAPI
- UI
- deployment

The point is to understand sparse participation and learning mechanics using only Chapters 2-4.

## 4. D2L Chapter Connections

Chapter 2 connections:

- tensors store batches, parameters, labels, losses, and metrics
- broadcasting appears in bias addition and normalization
- linear algebra appears in dot products, cosine similarity, and matrix multiplication
- calculus appears in gradients
- autodiff computes gradients from scalar losses
- probability appears in synthetic data generation and train/test sampling

Chapter 3 connections:

- linear regression is the simplest expert
- squared loss gives a clean regression training signal
- manual SGD shows exactly how parameters move
- weight decay controls parameter size
- generalization gap shows whether training success transfers to test data

Chapter 4 connections:

- softmax regression is the classification expert
- cross-entropy trains class probabilities
- accuracy evaluates class decisions
- distribution shift tests whether the learned system is brittle

## 5. Core Vocabulary

Region:

A synthetic domain. Each region has its own data pattern and its own hidden rule.

Expert:

A small model assigned to a region. In this project, an expert is only linear regression or softmax regression.

Router:

A function that scores an input vector against region embeddings and selects one or more regions.

Region table:

A matrix where each row is a learned or fixed vector representing one region.

Active expert:

An expert selected for the current input or batch.

Dormant expert:

An expert not selected for the current input or batch. It should not receive meaningful gradient updates during sparse training.

Distribution shift:

The training data and test data are generated differently. For example, the training set mostly contains region 0 and 1, while the test set mostly contains region 2 and 3.

Generalization gap:

The difference between training performance and test performance.

## 6. Phase 0: Environment and Ground Rules

### 6.1 Intuition

Before building routing, you need a small reliable training sandbox.

The environment should be boring:

- PyTorch
- optional matplotlib
- no external datasets
- no downloads
- no scikit-learn required

The data will be synthetic because synthetic data lets you know the hidden truth.

### 6.2 Why This Exists

If a real dataset fails, you often do not know whether the issue is data quality, model capacity, training code, label noise, preprocessing, or evaluation.

Synthetic data lets you ask cleaner questions:

- Did the model learn the true weights?
- Did the router select the right region?
- Did the selected expert receive the update?
- Did the inactive experts stay unchanged?
- Did distribution shift hurt?

### 6.3 First Imports

Type this at the top of your early experiment file:

Action:

Run this in `experiments.ipynb`.

```python
import torch
```

Optional later:

Action:

Run this only when you start plotting.

```python
import matplotlib.pyplot as plt
```

Set a seed:

Action:

Run this in `experiments.ipynb`.

```python
torch.manual_seed(0)
```

### 6.4 First Shape Checks

Before modeling, make sure you can inspect shapes:

Action:

Run this in `experiments.ipynb`.

```python
X = torch.randn(4, 6)
w = torch.randn(6)
b = torch.tensor(0.5)
y_hat = X @ w + b

print(X.shape)
print(w.shape)
print(y_hat.shape)
```

Expected shapes:

```text
X:     [4, 6]
w:     [6]
y_hat: [4]
```

### 6.5 Common Confusion Points

- `X @ w` means each row of `X` gets dot-producted with `w`.
- The bias `b` is scalar, so PyTorch broadcasts it across all 4 predictions.
- A batch dimension is not a feature dimension.
- If `X` has shape `[batch, features]`, then `w` should usually have shape `[features]` for scalar regression.

## 7. Phase 1: One Global Linear Regression Baseline

### 7.1 Intuition

Start with the simplest possible world:

```text
one dataset
one true rule
one linear model
one loss
one optimizer
```

If this does not work, routing will only hide the bug.

### 7.2 Dataset Rule

Create data from a known hidden function:

```text
y = X @ true_w + true_b + noise
```

The model should learn weights close to `true_w` and bias close to `true_b`.

For this first baseline, use exactly 6 features. The hardcoded `true_w` below has length 6.

### 7.3 Code To Type In `data.py`

```python
def make_regression_data(num_examples, num_features, noise_std=0.1):
    true_w = torch.tensor([2.0, -3.0, 1.5, 0.0, 0.5, -1.0])
    true_b = torch.tensor(0.7)
    X = torch.randn(num_examples, num_features)
    noise = torch.randn(num_examples) * noise_std
    y = X @ true_w + true_b + noise
    return X, y, true_w, true_b
```

Shape contract:

```text
X:      [num_examples, num_features]
y:      [num_examples]
true_w: [num_features]
true_b: scalar
```

### 7.4 Train/Test Split In `data.py`

```python
def train_test_split(X, y, train_fraction=0.8):
    n = X.shape[0]
    shuffled = torch.randperm(n)
    train_size = int(n * train_fraction)
    train_idx = shuffled[:train_size]
    test_idx = shuffled[train_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
```

Why this exists:

Training loss tells you whether the model fits examples it saw.

Test loss tells you whether the model learned a rule that transfers to examples it did not see.

### 7.5 Model, Loss, and SGD

Type `predict_regression` and `squared_loss` into `models.py`.

```python
def predict_regression(X, w, b):
    return X @ w + b
```

```python
def squared_loss(y_hat, y):
    return ((y_hat - y) ** 2).mean()
```

Type `sgd` into `train.py`.

```python
def sgd(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()
```

### 7.6 Training Loop In `experiments.ipynb`

Run this cell immediately after typing it.

Expected visible output:

```text
none
```

No output is normal. This cell trains the parameters, but it does not print anything.

```python
X, y, true_w, true_b = make_regression_data(200, 6)
X_train, y_train, X_test, y_test = train_test_split(X, y)

w = torch.randn(6, requires_grad=True)
b = torch.zeros((), requires_grad=True)

for epoch in range(50):
    y_hat = predict_regression(X_train, w, b)
    loss = squared_loss(y_hat, y_train)
    loss.backward()
    sgd([w, b], lr=0.05)
```

### 7.7 Evaluation In `experiments.ipynb`

Run this cell immediately after the training loop.

Expected visible output:

```text
train loss number
test loss number
learned weights
learned bias
true weights
true bias
```

```python
with torch.no_grad():
    train_loss = squared_loss(predict_regression(X_train, w, b), y_train)
    test_loss = squared_loss(predict_regression(X_test, w, b), y_test)

print(train_loss.item())
print(test_loss.item())
print(w)
print(b)
print(true_w)
print(true_b)
```

Run checkpoint:

Do not move to Phase 2 until this evaluation cell prints losses and learned parameters.

### 7.8 Step-by-Step Breakdown

`make_regression_data` creates a world with a known linear rule.

`train_test_split` separates examples used for learning from examples used for evaluation.

`w` and `b` are the learnable parameters.

`requires_grad=True` tells PyTorch to track operations involving those tensors.

`loss.backward()` computes gradients of the scalar loss with respect to `w` and `b`.

`sgd` moves each parameter opposite its gradient.

`torch.no_grad()` prevents the parameter update itself from being tracked by autograd.

`p.grad.zero_()` clears gradients so the next epoch starts cleanly.

### 7.9 Break It Deliberately

Try these one at a time:

- remove `p.grad.zero_()`
- set `lr=5.0`
- set `noise_std=5.0`
- use only 10 training examples
- evaluate on training data only

Write what happens in `notes.md`.

### 7.10 Checkpoint

You are ready to move on only when you can answer:

- What is the shape of `X`?
- What is the shape of `w`?
- Why must the loss be scalar before calling `backward()`?
- Why do gradients need to be cleared?
- How do train loss and test loss differ?
- What does it mean if `w` is close to `true_w`?

## 8. Phase 2: Multiple Regions With Different Hidden Rules

### 8.1 Intuition

Now make the world less convenient.

Instead of one global rule, create several regions:

```text
region 0 has one hidden linear rule
region 1 has another hidden linear rule
region 2 has another hidden linear rule
region 3 has another hidden linear rule
```

One global linear model now has a harder job. It has to average across conflicting rules.

### 8.2 Region Prototypes In `data.py`

Each region gets a prototype vector.

The prototype is not the label rule. It is the routing clue.

```python
def make_region_table(num_regions, num_features):
    table = torch.randn(num_regions, num_features)
    table = table / table.norm(dim=1, keepdim=True)
    return table
```

Shape contract:

```text
region_table: [num_regions, num_features]
```

Each row is one region embedding.

### 8.3 Region-Specific Regression Rules In `data.py`

```python
def make_region_rules(num_regions, num_features):
    true_W = torch.randn(num_regions, num_features)
    true_b = torch.randn(num_regions)
    return true_W, true_b
```

Shape contract:

```text
true_W: [num_regions, num_features]
true_b: [num_regions]
```

For region `r`, the hidden rule is:

```text
y = x @ true_W[r] + true_b[r] + noise
```

### 8.4 Generate Region Data In `data.py`

```python
def make_sparse_regression_data(
    num_examples,
    region_table,
    true_W,
    true_b,
    mixture,
    feature_noise=0.3,
    label_noise=0.1,
):
    num_regions, num_features = region_table.shape
    region_ids = torch.multinomial(mixture, num_examples, replacement=True)
    X = region_table[region_ids] + torch.randn(num_examples, num_features) * feature_noise
    y = (X * true_W[region_ids]).sum(dim=1) + true_b[region_ids]
    y = y + torch.randn(num_examples) * label_noise
    return X, y, region_ids
```

Shape contract:

```text
X:          [num_examples, num_features]
y:          [num_examples]
region_ids: [num_examples]
```

### 8.5 Why This Dataset Matters

This dataset has two levels:

Input geometry:

```text
examples near region_table[0] probably belong to region 0
examples near region_table[1] probably belong to region 1
```

Label rule:

```text
region 0 uses true_W[0], true_b[0]
region 1 uses true_W[1], true_b[1]
```

So the router can use geometry to guess which expert should handle the example.

### 8.6 Checkpoint Prints In `experiments.ipynb`

Run this cell immediately after typing it.

Expected visible output:

```text
X shape
y shape
region_ids shape
first few region IDs
```

```python
num_regions = 4
num_features = 6
region_table = make_region_table(num_regions, num_features)
true_W, true_b = make_region_rules(num_regions, num_features)
mixture = torch.tensor([0.25, 0.25, 0.25, 0.25])

X, y, region_ids = make_sparse_regression_data(
    20, region_table, true_W, true_b, mixture
)

print(X.shape)
print(y.shape)
print(region_ids.shape)
print(region_ids[:10])
```

Expected:

```text
X is [20, 6]
y is [20]
region_ids is [20]
region_ids contains integers from 0 to 3
```

### 8.7 Common Confusion Points

- `region_table` helps create inputs and route inputs.
- `true_W` creates labels.
- `region_ids` are known only because this is synthetic data.
- In real data, you usually do not get perfect region IDs.
- Do not train on `region_ids` as labels unless the experiment explicitly asks for oracle routing.

### 8.8 Region-Aware Train/Test Split In `data.py`

Once `region_ids` exist, split them together with `X` and `y`.

If you split `X` and `y` but forget to split `region_ids` with the same shuffled indices, your per-region metrics and oracle routing will silently become wrong.

```python
def train_test_split_with_regions(X, y, region_ids, train_fraction=0.8):
    n = X.shape[0]
    shuffled = torch.randperm(n)
    train_size = int(n * train_fraction)
    train_idx = shuffled[:train_size]
    test_idx = shuffled[train_size:]
    return (
        X[train_idx],
        y[train_idx],
        region_ids[train_idx],
        X[test_idx],
        y[test_idx],
        region_ids[test_idx],
    )
```

## 9. Phase 3: Global Model On Multi-Region Data

### 9.1 Intuition

Before sparse experts, train one global model on all regions.

This is the baseline.

If the global model already works perfectly, sparse routing has little to prove.

### 9.2 Code To Type In `experiments.ipynb`

Use the same `predict_regression`, `squared_loss`, and `sgd` from Phase 1.

Generate data:

Run this data-generation cell before the training cell.

```python
torch.manual_seed(0)

num_regions = 4
num_features = 6
region_table = make_region_table(num_regions, num_features)
true_W, true_b = make_region_rules(num_regions, num_features)
mixture = torch.tensor([0.25, 0.25, 0.25, 0.25])

X, y, region_ids = make_sparse_regression_data(
    500, region_table, true_W, true_b, mixture
)
(
    X_train,
    y_train,
    train_region_ids,
    X_test,
    y_test,
    test_region_ids,
) = train_test_split_with_regions(X, y, region_ids)
```

Train:

Run this training cell after the data-generation cell.

Expected visible output:

```text
none
```

```python
w = torch.randn(num_features, requires_grad=True)
b = torch.zeros((), requires_grad=True)

for epoch in range(200):
    y_hat = predict_regression(X_train, w, b)
    loss = squared_loss(y_hat, y_train)
    loss.backward()
    sgd([w, b], lr=0.03)
```

Evaluate:

Run this evaluation cell immediately after training.

```python
with torch.no_grad():
    train_loss = squared_loss(predict_regression(X_train, w, b), y_train)
    test_loss = squared_loss(predict_regression(X_test, w, b), y_test)

print("global train loss:", train_loss.item())
print("global test loss:", test_loss.item())
```

### 9.3 What You Should Expect

The global model may improve, but it is trying to fit several different hidden rules with one set of weights.

That means it may become a compromise model.

This is the first lesson:

> A model can be too simple not because linear regression is broken, but because one linear rule is being asked to represent several incompatible local rules.

### 9.4 Experiment Prompt

In `notes.md`, write:

```text
Experiment: global linear model on multi-region regression

What I expected:

What happened:

Train loss:

Test loss:

Why one global model may struggle:
```

## 10. Phase 4: Similarity Router

### 10.1 Intuition

The router answers:

```text
Which region does this input look closest to?
```

It does not predict the label.

It only selects which expert gets to work.

### 10.2 Cosine Similarity

Cosine similarity compares direction rather than raw length.

For two vectors:

```text
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
```

If two vectors point in similar directions, cosine similarity is high.

### 10.3 Code To Type In `router.py`

```python
def normalize_rows(X):
    return X / (X.norm(dim=1, keepdim=True) + 1e-8)
```

```python
def route_topk(X, region_table, k=1):
    X_norm = normalize_rows(X)
    table_norm = normalize_rows(region_table)
    scores = X_norm @ table_norm.T
    top_scores, top_ids = torch.topk(scores, k=k, dim=1)
    return top_ids, top_scores, scores
```

Shape contract:

```text
X:            [batch, features]
region_table: [regions, features]
scores:       [batch, regions]
top_ids:      [batch, k]
top_scores:   [batch, k]
```

### 10.4 Test The Router In `experiments.ipynb`

Run this cell after `route_topk` exists in `router.py` and after you have generated `X`, `region_table`, and `region_ids`.

Expected visible output:

```text
one router accuracy number
```

```python
top_ids, top_scores, scores = route_topk(X, region_table, k=1)
predicted_regions = top_ids.squeeze(1)
router_accuracy = (predicted_regions == region_ids).float().mean()

print(router_accuracy.item())
```

Because the synthetic inputs were generated near their region prototypes, router accuracy should be meaningfully above random.

For 4 regions, random top-1 routing is around:

```text
1 / 4 = 0.25
```

### 10.5 Step-by-Step Breakdown

`normalize_rows(X)` divides each row by its own length.

`region_table.T` turns `[regions, features]` into `[features, regions]`.

`X_norm @ table_norm.T` produces every input-to-region score.

`torch.topk(..., dim=1)` selects the best region per input row.

`squeeze(1)` turns `[batch, 1]` into `[batch]`.

### 10.6 Break It Deliberately

Try:

- remove normalization
- increase `feature_noise` from `0.3` to `2.0`
- set all region prototypes to the same vector
- use `k=2`

Write what happens.

### 10.7 Common Confusion Points

- Routing accuracy is not model accuracy.
- The router can be right while the expert is untrained.
- The expert can learn if routing is noisy, but the job becomes harder.
- The router uses input geometry, not labels.

## 11. Phase 5: Routed Regression Experts

### 11.1 Intuition

Now create one linear model per region.

Instead of:

```text
one global w, one global b
```

use:

```text
W[0], b[0] for region 0
W[1], b[1] for region 1
W[2], b[2] for region 2
W[3], b[3] for region 3
```

Each expert only sees examples routed to it.

### 11.2 Expert Parameter Shapes In `experiments.ipynb`

Use:

Action:

Run this in `experiments.ipynb` before the routed regression training loop.

```python
expert_W = torch.randn(num_regions, num_features, requires_grad=True)
expert_b = torch.zeros(num_regions, requires_grad=True)
```

Shape contract:

```text
expert_W: [regions, features]
expert_b: [regions]
```

For region `r`:

```text
prediction = X_for_region_r @ expert_W[r] + expert_b[r]
```

### 11.3 Routed Loss In `models.py`

```python
def routed_regression_loss(X, y, expert_W, expert_b, route_ids):
    losses = []
    counts = []

    for r in range(expert_W.shape[0]):
        mask = route_ids == r
        if mask.any():
            y_hat = X[mask] @ expert_W[r] + expert_b[r]
            errors = (y_hat - y[mask]) ** 2
            losses.append(errors.sum())
            counts.append(errors.numel())

    total_loss = torch.stack(losses).sum()
    total_count = sum(counts)
    return total_loss / total_count
```

Important:

This averages over examples, not over regions.

If you average each region loss equally, a region with 2 examples gets the same weight as a region with 200 examples. That may be useful for some experiments, but it is not the default baseline.

### 11.4 Training With Similarity Routing In `experiments.ipynb`

Run this cell after `routed_regression_loss` exists in `models.py`.

Expected visible output:

```text
none
```

```python
expert_W = torch.randn(num_regions, num_features, requires_grad=True)
expert_b = torch.zeros(num_regions, requires_grad=True)
usage = torch.zeros(num_regions)

for epoch in range(200):
    top_ids, _, _ = route_topk(X_train, region_table, k=1)
    route_ids = top_ids.squeeze(1)

    loss = routed_regression_loss(
        X_train, y_train, expert_W, expert_b, route_ids
    )

    loss.backward()
    sgd([expert_W, expert_b], lr=0.03)

    with torch.no_grad():
        usage += torch.bincount(route_ids, minlength=num_regions)
```

### 11.5 Evaluation Function

Type `routed_predict_regression` into `models.py`.

```python
def routed_predict_regression(X, expert_W, expert_b, region_table):
    top_ids, _, _ = route_topk(X, region_table, k=1)
    route_ids = top_ids.squeeze(1)
    y_hat = torch.zeros(X.shape[0])

    for r in range(expert_W.shape[0]):
        mask = route_ids == r
        if mask.any():
            y_hat[mask] = X[mask] @ expert_W[r] + expert_b[r]

    return y_hat, route_ids
```

Type the evaluation block into `experiments.ipynb`.

Run this evaluation cell immediately after the routed training loop.

```python
with torch.no_grad():
    train_pred, train_routes = routed_predict_regression(
        X_train, expert_W, expert_b, region_table
    )
    test_pred, test_routes = routed_predict_regression(
        X_test, expert_W, expert_b, region_table
    )

    train_loss = squared_loss(train_pred, y_train)
    test_loss = squared_loss(test_pred, y_test)

print("routed train loss:", train_loss.item())
print("routed test loss:", test_loss.item())
print("usage:", usage)
```

### 11.6 What Should Happen

If routing is decent and each region has enough examples, routed experts should often beat the global model on this synthetic task.

Why:

The global model has one weight vector.

The routed system has one weight vector per region.

If the hidden rules differ by region, the routed system has a better inductive bias.

### 11.7 Important Autograd Detail

Even though `expert_W` is one tensor, indexing can produce sparse-like gradient behavior.

If only `expert_W[2]` participates in the computation graph for a batch, then only region 2's slice receives meaningful gradient from that batch.

You can inspect this:

Action:

Run this in `experiments.ipynb` only when you want to inspect gradients after `loss.backward()` and before `sgd(...)`.

```python
print(expert_W.grad)
```

Expected idea:

```text
rows for active regions: nonzero or meaningful gradients
rows for inactive regions: zero gradients
```

### 11.8 Checkpoint

You are ready to move on when you can explain:

- why `expert_W` has shape `[regions, features]`
- why `route_ids` has shape `[batch]`
- why masks are needed
- why some experts may receive no update
- why this is not the same as training four separate global models manually

## 12. Phase 6: Oracle Routing, Random Routing, Similarity Routing

### 12.1 Intuition

You need routing baselines.

Three useful routing modes:

```text
oracle routing:     use true synthetic region_ids
random routing:     choose random regions
similarity routing: use cosine similarity to region table
```

Oracle routing tells you the best case.

Random routing tells you the bad baseline.

Similarity routing tells you whether your actual router adds value.

### 12.2 Code To Type In `router.py`

```python
def random_routes(num_examples, num_regions):
    return torch.randint(0, num_regions, (num_examples,))
```

```python
def similarity_routes(X, region_table):
    top_ids, _, _ = route_topk(X, region_table, k=1)
    return top_ids.squeeze(1)
```

Oracle routes are just:

Type this directly into `experiments.ipynb` when running the oracle experiment.

Action:

Run this as part of the oracle-routing experiment cell. Do not put it in `router.py`.

```python
route_ids = region_ids
```

### 12.3 Experiment Table

Run the same training loop with:

```text
global model
routed experts with oracle routing
routed experts with random routing
routed experts with similarity routing
```

Record:

```text
model_type | routing_type | train_loss | test_loss | router_accuracy
```

Run checkpoint:

Do not move to weight decay until you have at least rough numbers for global, oracle-routed, random-routed, and similarity-routed regression.

### 12.4 What To Learn

If oracle routing is much better than similarity routing:

The expert model can work, but the router is weak.

If similarity routing is much better than random routing:

The region table carries useful information.

If global model matches routed experts:

The regions may not actually have different enough rules, or the task is too easy.

If all models fail:

The learning rate, data, labels, or loss code may be broken.

## 13. Phase 7: Add Weight Decay

### 13.1 Intuition

Weight decay penalizes large weights.

It changes the training objective from:

```text
prediction error
```

to:

```text
prediction error + penalty for large weights
```

### 13.2 Manual Weight Decay

Type `l2_penalty` into `train.py`.

For the global model:

```python
def l2_penalty(w):
    return (w ** 2).sum() / 2
```

Type the loss-composition lines into `experiments.ipynb` inside the relevant training loop.

Action:

Do not run these two snippets by themselves. Insert them into the training loop variant where you are testing weight decay, then run that whole training loop.

```python
loss = squared_loss(y_hat, y_train) + wd * l2_penalty(w)
```

For routed experts:

```python
loss = routed_regression_loss(
    X_train, y_train, expert_W, expert_b, route_ids
)
loss = loss + wd * l2_penalty(expert_W)
```

### 13.3 What To Measure In `experiments.ipynb`

Record:

```text
wd | train_loss | test_loss | weight_norm
```

Compute weight norm:

Run this after each weight-decay experiment.

```python
with torch.no_grad():
    norm = expert_W.norm().item()
```

### 13.4 What To Learn

Weight decay usually increases training loss slightly.

It can improve test loss if the model was overfitting.

It can hurt if the task needs large weights or the model was not overfitting.

### 13.5 Common Confusion Points

- Weight decay is not dropout.
- Weight decay is not a learning rate.
- Weight decay changes the objective.
- A smaller weight norm is not automatically better.

## 14. Phase 8: Distribution Shift

### 14.1 Intuition

Distribution shift means the world changes between training and testing.

Examples in this project:

- train mostly on region 0 and 1, test mostly on region 2 and 3
- train with low feature noise, test with high feature noise
- train with balanced regions, test with imbalanced regions
- test with shifted region prototypes

### 14.2 Mixture Shift In `experiments.ipynb`

Training mixture:

Action:

Run this in `experiments.ipynb` when starting the mixture-shift experiment.

```python
train_mixture = torch.tensor([0.45, 0.45, 0.05, 0.05])
```

Test mixture:

Action:

Run this immediately after the training mixture cell.

```python
test_mixture = torch.tensor([0.05, 0.05, 0.45, 0.45])
```

Generate:

Run this before training the shifted-distribution experiment.

```python
X_train, y_train, train_region_ids = make_sparse_regression_data(
    500, region_table, true_W, true_b, train_mixture
)

X_test, y_test, test_region_ids = make_sparse_regression_data(
    200, region_table, true_W, true_b, test_mixture
)
```

### 14.3 Noise Shift In `experiments.ipynb`

Train:

Run these cells when you intentionally want feature noise to differ between training and testing.

```python
X_train, y_train, train_region_ids = make_sparse_regression_data(
    500, region_table, true_W, true_b, train_mixture, feature_noise=0.2
)
```

Test:

Action:

Run this immediately after the shifted test-mixture/noise setup.

```python
X_test, y_test, test_region_ids = make_sparse_regression_data(
    200, region_table, true_W, true_b, test_mixture, feature_noise=1.0
)
```

### 14.4 What To Measure

Record:

```text
shift_type | model_type | train_loss | test_loss | router_accuracy | region_usage
```

Also compute per-region test loss:

Type `per_region_mse` into `metrics.py`.

```python
def per_region_mse(y_hat, y, region_ids, num_regions):
    values = []
    for r in range(num_regions):
        mask = region_ids == r
        if mask.any():
            mse = ((y_hat[mask] - y[mask]) ** 2).mean()
            values.append(mse.item())
        else:
            values.append(None)
    return values
```

### 14.5 What To Learn

Average test loss can hide failures.

Example:

```text
region 0 loss: low
region 1 loss: low
region 2 loss: terrible
region 3 loss: terrible
average: looks only moderately bad
```

Per-region metrics tell you where the model fails.

### 14.6 Connection To Chapter 4.7

Chapter 4.7 says the environment can change.

This phase makes that concrete:

- same model
- same training code
- different test distribution
- different result

That is distribution shift in mechanical form.

## 15. Phase 9: Classification Version

### 15.1 Intuition

Regression predicts a number.

Classification predicts a class.

The routed idea is the same:

```text
route input -> selected classifier expert -> logits -> cross-entropy -> update
```

### 15.2 Region-Specific Classification Rules In `data.py`

```python
def make_region_class_rules(num_regions, num_features, num_classes):
    true_W = torch.randn(num_regions, num_features, num_classes)
    true_b = torch.randn(num_regions, num_classes)
    return true_W, true_b
```

Shape contract:

```text
true_W: [regions, features, classes]
true_b: [regions, classes]
```

### 15.3 Generate Classification Data In `data.py`

```python
def make_sparse_classification_data(
    num_examples,
    region_table,
    true_W,
    true_b,
    mixture,
    feature_noise=0.3,
):
    num_regions, num_features = region_table.shape
    region_ids = torch.multinomial(mixture, num_examples, replacement=True)
    X = region_table[region_ids] + torch.randn(num_examples, num_features) * feature_noise
    logits = torch.zeros(num_examples, true_b.shape[1])

    for r in range(num_regions):
        mask = region_ids == r
        if mask.any():
            logits[mask] = X[mask] @ true_W[r] + true_b[r]

    y = logits.argmax(dim=1)
    return X, y, region_ids
```

Shape contract:

```text
X:          [examples, features]
y:          [examples]
region_ids: [examples]
logits:     [examples, classes]
```

### 15.4 Global Softmax Classifier

Type the import and training loop into `experiments.ipynb`.

Run the classification data-generation cell first, then run this training loop.

Classification data-generation cell:

Action:

Run this in `experiments.ipynb` before the global softmax training loop.

```python
num_classes = 3
true_class_W, true_class_b = make_region_class_rules(
    num_regions, num_features, num_classes
)
X, y, region_ids = make_sparse_classification_data(
    500, region_table, true_class_W, true_class_b, mixture
)
(
    X_train,
    y_train,
    train_region_ids,
    X_test,
    y_test,
    test_region_ids,
) = train_test_split_with_regions(X, y, region_ids)
```

Use logits directly.

Do not manually apply softmax before `torch.nn.functional.cross_entropy`.

Action:

Run this import if it has not already been run in the notebook import cell.

```python
import torch.nn.functional as F
```

Action:

Run this training loop in `experiments.ipynb`.

```python
W = torch.randn(num_features, num_classes, requires_grad=True)
b = torch.zeros(num_classes, requires_grad=True)

for epoch in range(200):
    logits = X_train @ W + b
    loss = F.cross_entropy(logits, y_train)
    loss.backward()
    sgd([W, b], lr=0.05)
```

Accuracy:

Type `accuracy` into `metrics.py`.

```python
def accuracy(logits, y):
    predictions = logits.argmax(dim=1)
    return (predictions == y).float().mean()
```

Evaluation:

Action:

Run this in `experiments.ipynb` immediately after the global softmax training loop.

```python
with torch.no_grad():
    train_logits = X_train @ W + b
    test_logits = X_test @ W + b
    train_acc = accuracy(train_logits, y_train)
    test_acc = accuracy(test_logits, y_test)

print("global train accuracy:", train_acc.item())
print("global test accuracy:", test_acc.item())
```

### 15.5 Routed Softmax Classifier

Type the expert parameter initialization into `experiments.ipynb`.

Run this after the global softmax classifier baseline works.

Expert parameter shapes:

Action:

Run this in `experiments.ipynb`.

```python
expert_W = torch.randn(
    num_regions, num_features, num_classes, requires_grad=True
)
expert_b = torch.zeros(num_regions, num_classes, requires_grad=True)
```

Routed logits:

Type `routed_classification_logits` into `models.py`.

```python
def routed_classification_logits(X, expert_W, expert_b, route_ids):
    num_examples = X.shape[0]
    num_classes = expert_b.shape[1]
    logits = torch.zeros(num_examples, num_classes)

    for r in range(expert_W.shape[0]):
        mask = route_ids == r
        if mask.any():
            logits[mask] = X[mask] @ expert_W[r] + expert_b[r]

    return logits
```

Training:

Type this loop into `experiments.ipynb`.

Run this loop after `routed_classification_logits` exists in `models.py`.

```python
for epoch in range(200):
    route_ids = similarity_routes(X_train, region_table)
    logits = routed_classification_logits(
        X_train, expert_W, expert_b, route_ids
    )
    loss = F.cross_entropy(logits, y_train)
    loss.backward()
    sgd([expert_W, expert_b], lr=0.05)
```

### 15.6 Important Softmax Confusion

For learning:

```text
logits -> cross_entropy
```

For interpretation:

```text
logits -> softmax -> probabilities
```

Do not do this during training:

Action:

Do not run this. It is shown as an anti-example.

```python
loss = F.cross_entropy(torch.softmax(logits, dim=1), y)
```

That is conceptually redundant and numerically worse.

### 15.7 Classification Experiment Table

Record:

```text
model_type | routing_type | train_accuracy | test_accuracy | test_loss | router_accuracy
```

Compare:

- global classifier
- oracle-routed classifier
- random-routed classifier
- similarity-routed classifier

Run checkpoint:

Do not move to top-2 routing until the global classifier and similarity-routed classifier both run without shape errors.

## 16. Phase 10: Top-2 Routing

### 16.1 Intuition

Top-1 routing chooses one expert.

Top-2 routing lets two experts vote.

This is closer to sparse MoE behavior, but still simple.

### 16.2 Regression Version

For each input:

```text
choose top 2 experts
get 2 predictions
average the predictions
```

This is not necessarily optimal. It is just the simplest ballot.

### 16.3 Code Sketch In `models.py`

```python
def top2_routed_predict_regression(X, expert_W, expert_b, region_table):
    top_ids, _, _ = route_topk(X, region_table, k=2)
    preds = torch.zeros(X.shape[0], 2)

    for j in range(2):
        route_ids = top_ids[:, j]
        for r in range(expert_W.shape[0]):
            mask = route_ids == r
            if mask.any():
                preds[mask, j] = X[mask] @ expert_W[r] + expert_b[r]

    return preds.mean(dim=1), top_ids
```

### 16.4 What To Test

Compare:

```text
top-1 similarity routing
top-2 similarity routing
top-2 oracle plus one distractor
top-2 random routing
```

Action:

Run these as experiment variants in `experiments.ipynb` after `top2_routed_predict_regression` exists in `models.py`. This section is not one standalone cell; it tells you which variants to compare.

### 16.5 What To Learn

Top-2 may help when routing is uncertain.

Top-2 may hurt when the second expert is wrong and its vote corrupts the result.

This connects to an important systems idea:

> More active compute is not automatically better. It depends on whether the extra compute is relevant.

## 17. Phase 11: Local Update Gate

### 17.1 Intuition

This is the first tiny connection to the algorithm topology.

Instead of updating every time, add a crude gate:

```text
expert participated AND loss is high enough -> update
otherwise -> skip durable update
```

This is not real biology. It is a mechanical toy analog for:

```text
local participation x global salience
```

### 17.2 Simple Batch-Level Gate In `experiments.ipynb`

Run this only after ordinary routed regression training works. This is an experiment variant, not required for the first successful baseline.

Action:

Do not run this as a standalone cell unless `route_ids`, `expert_W`, and `expert_b` already exist from the current experiment. Usually, insert this logic inside a modified routed training loop, replacing the ordinary `loss.backward()` and `sgd(...)` part.

```python
threshold = 0.5

loss = routed_regression_loss(
    X_train, y_train, expert_W, expert_b, route_ids
)

if loss.item() > threshold:
    loss.backward()
    sgd([expert_W, expert_b], lr=0.03)
else:
    with torch.no_grad():
        for p in [expert_W, expert_b]:
            if p.grad is not None:
                p.grad.zero_()
```

### 17.3 Why This Is Crude

This gate uses one scalar loss for the whole batch.

The biological story is more local and asynchronous:

- some synapses participate
- some local traces decay
- broad signals arrive independently
- durable changes happen only where signals coincide
- homeostatic scaling can run separately
- replay can run separately
- retrieval-triggered updates are not necessarily in the same timeline

So this phase is only a mechanical sketch.

### 17.4 What To Test

Run:

```text
no gate
threshold = low
threshold = medium
threshold = high
```

Record:

```text
threshold | number_of_updates | train_loss | test_loss | per_region_loss
```

### 17.5 What To Learn

If the threshold is too low:

The gate behaves almost like ordinary training.

If the threshold is too high:

The model may barely learn.

If the threshold is useful:

It may reduce unnecessary updates while preserving performance.

This is where the research instinct starts:

> When should a system be allowed to write?

## 18. Phase 12: Optional Homeostatic Scaling

### 18.1 Intuition

Homeostatic scaling is a local stabilizer idea.

In this toy project, make it very simple:

```text
periodically rescale each expert's weight vector toward a target norm
```

This is not content-aware. It does not know whether predictions are correct.

### 18.2 Code Sketch

Type `rescale_expert_weights` into `train.py`.

```python
def rescale_expert_weights(expert_W, target_norm=1.0):
    with torch.no_grad():
        norms = expert_W.norm(dim=1, keepdim=True)
        scale = target_norm / (norms + 1e-8)
        expert_W *= scale
```

Use every 20 epochs:

Type this call into the relevant training loop in `experiments.ipynb`.

Action:

Do not run this snippet by itself. Insert it inside the training loop variant where you are testing homeostatic scaling, then run that whole training loop.

```python
if epoch % 20 == 0:
    rescale_expert_weights(expert_W, target_norm=1.0)
```

### 18.3 What To Test

Compare:

```text
no weight decay
weight decay
homeostatic scaling
weight decay + homeostatic scaling
```

### 18.4 What To Learn

Weight decay changes the loss objective.

Homeostatic scaling changes weights outside the loss objective.

They are not the same operation.

This distinction matters for your algorithm topology because Stage 4 is not the same as Stage 3.

## 19. Phase 13: Optional Replay Buffer

### 19.1 Intuition

A replay buffer stores some old examples.

During later training, you mix old examples with current examples so the model does not only train on the newest distribution.

### 19.2 Simple Replay Buffer In `train.py`

Action:

Type and save this in `train.py`. Do not run `train.py`. Keep both code blocks inside the same `ReplayBuffer` class indentation.

```python
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.X = []
        self.y = []

    def add(self, X, y):
        for i in range(X.shape[0]):
            self.X.append(X[i].detach().clone())
            self.y.append(y[i].detach().clone())
        self.X = self.X[-self.max_size:]
        self.y = self.y[-self.max_size:]
```

```python
    def sample(self, batch_size):
        n = len(self.X)
        idx = torch.randint(0, n, (batch_size,))
        X = torch.stack([self.X[i] for i in idx])
        y = torch.stack([self.y[i] for i in idx])
        return X, y
```

If you type this into a real file, keep both methods inside the same class indentation.

### 19.3 What To Test

Create a curriculum:

```text
first train mostly on region 0 and 1
then train mostly on region 2 and 3
```

Compare:

```text
without replay: old-region performance after new-region training
with replay: old-region performance after new-region training
```

Action:

Run this later as an experiment variant in `experiments.ipynb` after the ordinary shifted-distribution experiment works. This section is not one standalone cell yet.

### 19.4 What To Learn

Replay is a different anti-forgetting mechanism from sparsity.

Sparsity says:

```text
do not touch unrelated parameters
```

Replay says:

```text
keep old examples in the training mixture
```

These can complement each other.

## 20. Required Experiment Matrix

At minimum, run these experiments.

Regression:

```text
1. global linear model, stable distribution
2. global linear model, shifted distribution
3. routed experts, oracle routing
4. routed experts, random routing
5. routed experts, similarity routing
6. routed experts, similarity routing, top-2 vote
7. routed experts, similarity routing, with weight decay
8. routed experts, similarity routing, with local update gate
```

Classification:

```text
1. global softmax classifier, stable distribution
2. global softmax classifier, shifted distribution
3. routed softmax experts, oracle routing
4. routed softmax experts, random routing
5. routed softmax experts, similarity routing
```

Optional:

```text
1. homeostatic scaling
2. replay buffer
3. mixture shift followed by recovery
4. per-region active update counts
5. deliberate router corruption
```

## 21. Metrics

Regression metrics:

- train MSE
- test MSE
- per-region MSE
- weight norm
- router accuracy
- region usage count
- number of parameter updates

Classification metrics:

- train accuracy
- test accuracy
- train cross-entropy
- test cross-entropy
- per-region accuracy
- confusion matrix
- router accuracy
- region usage count

Simple confusion matrix:

Type `confusion_matrix` into `metrics.py`.

```python
def confusion_matrix(pred, y, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for true, guessed in zip(y, pred):
        matrix[true, guessed] += 1
    return matrix
```

Rows are true labels.

Columns are predicted labels.

## 22. Notes Template

Use this in `notes.md`:

```text
# Project 0 Notes

## Current Experiment

Date:

Question:

Hypothesis:

Setup:

Model:

Routing:

Training distribution:

Test distribution:

Metrics:

Result:

What changed:

What surprised me:

What I think is happening mechanically:

Next experiment:
```

## 23. Debugging Checklist

Before asking for help, print:

Action:

Run these snippets in `experiments.ipynb` only when debugging.

```python
print("X", X.shape)
print("y", y.shape)
print("region_ids", region_ids.shape)
print("expert_W", expert_W.shape)
print("expert_b", expert_b.shape)
print("route_ids", route_ids.shape)
```

For regression:

```python
print("y_hat", y_hat.shape)
print("loss", loss.item())
```

For classification:

```python
print("logits", logits.shape)
print("y min/max", y.min().item(), y.max().item())
print("loss", loss.item())
```

Check gradients:

Action:

Run this only when debugging gradients. Run it before `sgd(...)`; otherwise the gradient may already have been cleared. Avoid running it repeatedly on the same loss unless you know you want gradient accumulation.

```python
loss.backward()
print(expert_W.grad)
```

Check route usage:

```python
print(torch.bincount(route_ids, minlength=num_regions))
```

## 24. Common Failure Modes

Shape mismatch:

Usually caused by mixing `[batch, features]`, `[features]`, `[features, classes]`, and `[regions, features]`.

Loss does not decrease:

Possible causes:

- learning rate too high
- learning rate too low
- gradients not cleared
- labels generated incorrectly
- model too weak
- routing too noisy
- experts receiving too little data

Test loss much worse than train loss:

Possible causes:

- overfitting
- distribution shift
- train/test split bug
- test regions underrepresented during training

Router accuracy bad:

Possible causes:

- feature noise too high
- region prototypes too similar
- normalization missing
- using the wrong dimension in `topk`

Classification loss weird:

Possible causes:

- applying softmax before `F.cross_entropy`
- labels are not integer class IDs
- labels contain values outside `[0, num_classes - 1]`
- logits shape is not `[batch, classes]`

Inactive experts changing:

Possible causes:

- weight decay applied globally
- optimizer updating all parameters with nonzero stale gradients
- gradients not cleared
- your loss accidentally used all experts

## 25. Final Writeup Prompts

At the end, write a short technical note answering:

1. What did the global model learn?
2. When did routed experts outperform the global model?
3. When did routed experts fail?
4. How much did router quality matter?
5. What did distribution shift do?
6. Which parameters received gradients during sparse training?
7. What did weight decay change?
8. What did per-region metrics reveal that average metrics hid?
9. Which part connects most clearly to D2L Chapters 2-4?
10. Which part feels like a real research question rather than a solved exercise?

## 26. Suggested Build Order

Use this exact order when typing:

```text
1. one global regression model on one linear dataset
2. train/test split
3. metrics
4. multi-region regression data
5. global baseline on multi-region data
6. cosine router
7. routed regression experts
8. oracle/random/similarity routing comparison
9. weight decay
10. distribution shift
11. global softmax classifier
12. routed softmax experts
13. top-2 routing
14. local update gate
15. optional homeostatic scaling
16. optional replay buffer
17. final writeup
```

Do not skip the global baseline.

Do not skip random routing.

Those weak baselines are what make the routed result interpretable.

## 27. Final Done Criteria

You are done with Project 0 when:

- the global regression baseline trains
- the global classification baseline trains
- the router returns top-k region IDs with correct shapes
- routed regression experts train
- routed classification experts train
- oracle routing, random routing, and similarity routing are compared
- at least one distribution shift experiment is run
- per-region metrics are computed
- weight decay is tested
- one local-update-gate experiment is attempted
- `notes.md` contains experiment logs and final explanations

The point is not that sparse experts must win.

The point is that you can explain why they win, lose, or behave differently.
