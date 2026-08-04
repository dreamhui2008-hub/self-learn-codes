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
15.2 exact run order: checklist, not code
15.3 logits/argmax drill: run in notebook
15.4 cross-entropy drill: run in notebook
15.5 bincount drill: run in notebook
15.6 classification data helpers: type/save only
15.7 routed classification model drills: run in notebook; final routed_classification_logits: type/save only
15.8 accuracy: type/save only; rerun notebook import/reload cell
15.9 classification data-shape drill: run in notebook
15.10 full classification experiment cell: run in notebook after reading block grammar
15.11 checkpoint: write answers in notes.md
16.2 exact run order: checklist, not code
16.3 top-k drill: run in notebook
16.4 top-2 prediction table drills: run in notebook; final top2_routed_predict_regression: type/save only; rerun notebook import/reload cell
16.5 full top-1 vs top-2 experiment cell: run in notebook after reading block grammar
16.6 checkpoint: write answers in notes.md
17.2 exact run order: checklist, not code
17.3 threshold gate drill: run in notebook
17.4 gated helper grammar drills: run in notebook; full gated-update experiment cell: run in notebook
17.5 checkpoint: write answers in notes.md
18.2 exact run order: checklist, not code
18.3 norms/rescaling drill: run in notebook
18.4 in-place rescale grammar drill: run in notebook; final rescale_expert_weights: type/save only; rerun notebook import/reload cell
18.5 full scaling experiment cell: run in notebook after reading block grammar
18.6 checkpoint: write answers in notes.md
19.2 exact run order: checklist, not code
19.3 torch.cat drill: run in notebook
19.4 replay storage grammar drills: run in notebook; final ReplayBuffer class: type/save only; rerun notebook import/reload cell
19.5 replay add/sample drill: run in notebook
19.6 batch-mixing grammar drill: run in notebook; full curriculum replay experiment cell: run in notebook
19.7 checkpoint: write answers in notes.md
20 required experiment matrix: checklist, not code
21.3 confusion_matrix: type/save only; rerun notebook import/reload cell
21.4 confusion matrix drill: run in notebook
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

Enter the following into `experiments.ipynb`: 

```python
# Clean normal data setup
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

def routed_predict_with_routes(X, expert_W, expert_b, route_ids):
    y_hat = torch.zeros(X.shape[0])

    for r in range(expert_W.shape[0]):
        mask = route_ids == r
        if mask.any():
            y_hat[mask] = X[mask] @ expert_W[r] + expert_b[r]

    return y_hat

# Glboal model
w = torch.randn(num_features, requires_grad=True)
b = torch.zeros((), requires_grad=True)

for epoch in range(200):
    y_hat = predict_regression(X_train, w, b)
    loss = squared_loss(y_hat, y_train)
    loss.backward()
    sgd([w, b], lr=0.01)

with torch.no_grad():
    global_train_loss = squared_loss(predict_regression(X_train, w, b), y_train)
    global_test_loss = squared_loss(predict_regression(X_test, w, b), y_test)

def run_routed_experiment(routing_type):
    expert_W = torch.randn(num_regions, num_features, requires_grad=True)
    expert_b = torch.zeros(num_regions, requires_grad=True)

    for epoch in range(200):
        if routing_type == "oracle":
            route_ids = train_region_ids
        elif routing_type == "random":
            route_ids = random_routes(X_train.shape[0], num_regions)
        elif routing_type == "similarity":
            route_ids = similarity_routes(X_train, region_table)

        loss = routed_regression_loss(
            X_train, y_train, expert_W, expert_b, route_ids
        )
        loss.backward()
        sgd([expert_W, expert_b], lr=0.03)

    with torch.no_grad():
        if routing_type == "oracle":
            train_routes = train_region_ids
            test_routes = test_region_ids
            router_acc = 1.0

        elif routing_type == "random":
            train_routes = random_routes(X_train.shape[0], num_regions)
            test_routes = random_routes(X_test.shape[0], num_regions)
            router_acc = (test_routes == test_region_ids).float().mean().item()

        elif routing_type == "similarity":
            train_routes = similarity_routes(X_train, region_table)
            test_routes = similarity_routes(X_test, region_table)
            router_acc = (test_routes == test_region_ids).float().mean().item()

        train_pred = routed_predict_with_routes(
            X_train, expert_W, expert_b, train_routes
        )
        test_pred = routed_predict_with_routes(
            X_test, expert_W, expert_b, test_routes
        )
        train_loss = squared_loss(train_pred, y_train)
        test_loss = squared_loss(test_pred, y_test)

    return train_loss.item(), test_loss.item(), router_acc

results = []

results.append([
    "global",
    "none",
    global_train_loss.item(),
    global_test_loss.item(),
    None,
])

for routing_type in ["oracle", "random", "similarity"]:
    train_loss, test_loss, router_acc = run_routed_experiment(routing_type)
    results.append([
        "routed experts",
        routing_type,
        train_loss,
        test_loss,
        router_acc,
    ])

for row in results:
    print(row)
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
Type this into `experiments.ipynb`.

```python
wd_values = [0.0, 0.001, 0.01, 0.1]

def run_global_weight_decay(wd):
    w = torch.randn(num_features, requires_grad=True)
    b = torch.zeros((), requires_grad=True)

    for epoch in range(200):
        y_hat = predict_regression(X_train, w, b)
        loss = squared_loss(y_hat, y_train) + wd * l2_penalty(w)
        loss.backward()
        sgd([w, b], lr=0.01)

    with torch.no_grad():
        train_loss = squared_loss(predict_regression(X_train, w, b), y_train)
        test_loss = squared_loss(predict_regression(X_test, w, b), y_test)
        weight_norm = w.norm().item()

    return train_loss.item(), test_loss.item(), weight_norm

def run_similarity_routed_weight_decay(wd):
    expert_W = torch.randn(num_regions, num_features, requires_grad=True)
    expert_b = torch.zeros(num_regions, requires_grad=True)

    for epoch in range(200):
        route_ids = similarity_routes(X_train, region_table)

        loss = routed_regression_loss(
            X_train, y_train, expert_W, expert_b, route_ids
        )
        loss = loss + wd * l2_penalty(expert_W)

        loss.backward()
        sgd([expert_W, expert_b], lr=0.03)

    with torch.no_grad():
        train_routes = similarity_routes(X_train, region_table)
        test_routes = similarity_routes(X_test, region_table)

        train_pred = routed_predict_with_routes(
            X_train, expert_W, expert_b, train_routes
        )
        test_pred = routed_predict_with_routes(
            X_test, expert_W, expert_b, test_routes
        )

        train_loss = squared_loss(train_pred, y_train)
        test_loss = squared_loss(test_pred, y_test)
        weight_norm = expert_W.norm().item()

    return train_loss.item(), test_loss.item(), weight_norm

results = []

for wd in wd_values:
    train_loss, test_loss, weight_norm = run_global_weight_decay(wd)
    results.append(["global", wd, train_loss, test_loss, weight_norm])

for wd in wd_values:
    train_loss, test_loss, weight_norm = run_similarity_routed_weight_decay(wd)
    results.append(["similarity routed", wd, train_loss, test_loss, weight_norm])

for row in results:
    print(row)
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

### 14.2 What To Measure

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

### 14.3 Mixture Shift In `experiments.ipynb`

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

### 14.4 Noise Shift In `experiments.ipynb`

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

Type this into `experiments.ipynb`.

```python
def run_global_on_current_data():
    w = torch.randn(num_features, requires_grad=True)
    b = torch.zeros((), requires_grad=True)

    for epoch in range(200):
        y_hat = predict_regression(X_train, w, b)
        loss = squared_loss(y_hat, y_train)
        loss.backward()
        sgd([w, b], lr=0.01)

    with torch.no_grad():
        train_pred = predict_regression(X_train, w, b)
        test_pred = predict_regression(X_test, w, b)

        train_loss = squared_loss(train_pred, y_train)
        test_loss = squared_loss(test_pred, y_test)
        per_region_test_loss = per_region_mse(
            test_pred, y_test, test_region_ids, num_regions
        )

    return train_loss.item(), test_loss.item(), None, None, per_region_test_loss


def run_similarity_routed_on_current_data():
    expert_W = torch.randn(num_regions, num_features, requires_grad=True)
    expert_b = torch.zeros(num_regions, requires_grad=True)

    for epoch in range(200):
        route_ids = similarity_routes(X_train, region_table)

        loss = routed_regression_loss(
            X_train, y_train, expert_W, expert_b, route_ids
        )

        loss.backward()
        sgd([expert_W, expert_b], lr=0.03)

    with torch.no_grad():
        train_routes = similarity_routes(X_train, region_table)
        test_routes = similarity_routes(X_test, region_table)

        train_pred = routed_predict_with_routes(
            X_train, expert_W, expert_b, train_routes
        )
        test_pred = routed_predict_with_routes(
            X_test, expert_W, expert_b, test_routes
        )

        train_loss = squared_loss(train_pred, y_train)
        test_loss = squared_loss(test_pred, y_test)
        router_accuracy = (test_routes == test_region_ids).float().mean().item()
        region_usage = torch.bincount(train_routes, minlength=num_regions).tolist()
        per_region_test_loss = per_region_mse(
            test_pred, y_test, test_region_ids, num_regions
        )

    return (
        train_loss.item(),
        test_loss.item(),
        router_accuracy,
        region_usage,
        per_region_test_loss,
    )


def run_shift_experiment(shift_type, train_mixture, test_mixture, train_feature_noise=0.3, test_feature_noise=0.3):
    global X_train, y_train, train_region_ids
    global X_test, y_test, test_region_ids

    X_train, y_train, train_region_ids = make_sparse_regression_data(
        500, region_table, true_W, true_b, train_mixture, feature_noise=train_feature_noise
    )

    X_test, y_test, test_region_ids = make_sparse_regression_data(
        200, region_table, true_W, true_b, test_mixture, feature_noise=test_feature_noise
    )

    rows = []

    train_loss, test_loss, router_accuracy, region_usage, per_region_loss = run_global_on_current_data()
    rows.append([shift_type, "global", train_loss, test_loss, router_accuracy, region_usage, per_region_loss])

    train_loss, test_loss, router_accuracy, region_usage, per_region_loss = run_similarity_routed_on_current_data()
    rows.append([shift_type, "similarity routed", train_loss, test_loss, router_accuracy, region_usage, per_region_loss])

    return rows


train_mixture = torch.tensor([0.45, 0.45, 0.05, 0.05])
test_mixture = torch.tensor([0.05, 0.05, 0.45, 0.45])

results = []

results += run_shift_experiment(
    "mixture shift",
    train_mixture,
    test_mixture,
)

results += run_shift_experiment(
    "mixture + noise shift",
    train_mixture,
    test_mixture,
    train_feature_noise=0.2,
    test_feature_noise=1.0,
)

for row in results:
    print(row)
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

### 15.0 Why This Phase Is More Structured

From this point forward, each phase uses the same pattern:

```text
concept -> syntax preflight -> file edits -> exact experiment cell -> checkpoint
```

Do not infer missing glue code. If a section says to run an experiment, the full cell appears in that section.

### 15.1 Goal

Regression predicts one number.

Classification predicts one class.

The routed idea stays the same:

```text
route input -> selected classifier expert -> logits -> cross-entropy -> update
```

New mechanics in this phase:

- `logits`
- integer class labels
- `argmax(dim=1)`
- `F.cross_entropy`
- `torch.bincount`
- routed classifier weight shape `[regions, features, classes]`

### 15.2 Exact Run Order

Use this order:

```text
1. Run the notebook import/reload cell.
2. Run the logits/argmax drill.
3. Run the cross-entropy drill.
4. Run the bincount drill.
5. Add classification helpers to data.py.
6. Run the routed classification model drills.
7. Add classification helpers to models.py and metrics.py.
8. Rerun the notebook import/reload cell.
9. Run the classification data-shape drill.
10. Run the full classification experiment cell.
11. Write the checkpoint answers in notes.md.
```

### 15.3 Syntax Preflight: Logits And `argmax`

Purpose:

Understand classification output shapes before routing is involved.

Action:

Run this in `experiments.ipynb`.

```python
X_small = torch.randn(5, 6)       # [batch, features]
W_small = torch.randn(6, 3)       # [features, classes]
b_small = torch.zeros(3)          # [classes]

logits = X_small @ W_small + b_small
pred_classes = logits.argmax(dim=1)

print("X_small:", X_small.shape)
print("W_small:", W_small.shape)
print("logits:", logits.shape)
print("pred_classes:", pred_classes.shape)
print(pred_classes)
```

Expected shape pattern:

```text
X_small: torch.Size([5, 6])
W_small: torch.Size([6, 3])
logits: torch.Size([5, 3])
pred_classes: torch.Size([5])
```

Mechanical meaning:

- `logits[i]` contains one raw score per class for example `i`.
- `argmax(dim=1)` picks the highest-scoring class across columns.
- `dim=1` is the class dimension because `logits` has shape `[batch, classes]`.

### 15.4 Syntax Preflight: Cross Entropy

Purpose:

Use PyTorch's classification loss directly.

Action:

Run this in `experiments.ipynb` after the logits drill.

```python
labels = torch.tensor([0, 2, 1, 1, 0])
loss = F.cross_entropy(logits, labels)

print("labels:", labels.shape)
print("loss:", loss)
```

Shape contract:

```text
logits: [batch, classes]
labels: [batch]
loss: scalar
```

Important:

- `F.cross_entropy` expects raw logits.
- Do not apply softmax before `F.cross_entropy`.
- labels must be integer class IDs such as `0`, `1`, `2`.

Anti-example:

Do not train with this:

```python
loss = F.cross_entropy(torch.softmax(logits, dim=1), labels)
```

That applies probability conversion before a function that already handles the needed log-softmax step internally.

### 15.5 Syntax Preflight: `torch.bincount`

Purpose:

Understand region usage counts before they appear in the experiment table.

Action:

Run this in `experiments.ipynb`.

```python
routes_small = torch.tensor([0, 2, 2, 1, 0, 3, 3, 3])
usage_small = torch.bincount(routes_small, minlength=4)

print("routes_small:", routes_small)
print("usage_small:", usage_small)
```

Expected output:

```text
usage_small: tensor([2, 1, 2, 3])
```

Mechanical meaning:

- region `0` appears 2 times
- region `1` appears 1 time
- region `2` appears 2 times
- region `3` appears 3 times

### 15.6 File Edit: Classification Data In `data.py`

Action:

Add these functions to `data.py`.

```python
def make_region_class_rules(num_regions, num_features, num_classes):
    # Create one hidden class-weight matrix per region.
    # Shape: [regions, features, classes]
    true_W = torch.randn(num_regions, num_features, num_classes)

    # Create one hidden class-bias vector per region.
    # Shape: [regions, classes]
    true_b = torch.randn(num_regions, num_classes)

    # These are answer-key parameters used to generate labels, not trained parameters.
    return true_W, true_b


def make_sparse_classification_data(
    num_examples,
    region_table,
    true_W,
    true_b,
    mixture,
    feature_noise=0.3,
):
    # region_table shape is [regions, features].
    # Unpack those dimensions so the rest of the function stays shape-driven.
    num_regions, num_features = region_table.shape

    # Sample one true region ID per example using the mixture probabilities.
    # Shape: [num_examples]
    region_ids = torch.multinomial(mixture, num_examples, replacement=True)

    # Build input rows near their assigned region prototype.
    # region_table[region_ids] shape: [num_examples, features]
    # noise shape: [num_examples, features]
    # X shape: [num_examples, features]
    X = region_table[region_ids] + torch.randn(num_examples, num_features) * feature_noise

    # Allocate the hidden logits table that will become class labels.
    # true_W.shape[2] is num_classes.
    # logits shape: [num_examples, classes]
    logits = torch.zeros(num_examples, true_W.shape[2])

    # Fill logits region by region so each example uses its own region's hidden rule.
    for r in range(num_regions):
        # mask shape: [num_examples]
        # True entries mark examples whose hidden region is r.
        mask = region_ids == r

        # Skip empty regions so X[mask] never becomes an empty training block here.
        if mask.any():
            # X[mask] shape: [examples_for_r, features]
            # true_W[r] shape: [features, classes]
            # true_b[r] shape: [classes]
            # logits[mask] shape: [examples_for_r, classes]
            logits[mask] = X[mask] @ true_W[r] + true_b[r]

    # Convert hidden class scores into integer class IDs.
    # y shape: [num_examples]
    y = logits.argmax(dim=1)

    # Return inputs, class labels, and true synthetic region IDs.
    return X, y, region_ids
```

Shape contract:

```text
true_W:     [regions, features, classes]
true_b:     [regions, classes]
X:          [examples, features]
y:          [examples]
region_ids: [examples]
logits:     [examples, classes]
```

Why this exists:

- The synthetic classification data still has hidden regions.
- Each region has its own hidden classifier.
- The target label is the class with the highest hidden logit.

### 15.7 File Edit: Classification Model In `models.py`

This section is a block-grammar drill.

Do not start by memorizing the full function.

Learn this repeated pattern first:

```text
allocate output table
for each expert r:
    build mask for rows routed to r
    run only those rows through expert r
    write those rows back into the output table
return the filled output table
```

The function you will type at the end is only this pattern packaged inside `def`.

#### 15.7.1 Tiny Routed Classification Shapes

Purpose:

Create the smallest concrete version of the tensors used by routed classification.

Action:

Run this in `experiments.ipynb`.

```python
torch.manual_seed(0)

# Five examples, each with six input features.
X_small = torch.randn(5, 6)

# Four experts.
# Each expert maps six input features to three class logits.
expert_W_small = torch.randn(4, 6, 3)

# Four experts.
# Each expert has one bias per class.
expert_b_small = torch.zeros(4, 3)

# One route ID per example.
# These are expert IDs, not class labels.
route_ids_small = torch.tensor([0, 2, 2, 1, 3])

# The output must have one class-score row per input example.
num_examples = X_small.shape[0]
num_classes = expert_b_small.shape[1]
logits_small = torch.zeros(num_examples, num_classes)

print("X_small:", X_small.shape)
print("expert_W_small:", expert_W_small.shape)
print("expert_b_small:", expert_b_small.shape)
print("route_ids_small:", route_ids_small.shape)
print("logits_small:", logits_small.shape)
```

Expected shape pattern:

```text
X_small: torch.Size([5, 6])
expert_W_small: torch.Size([4, 6, 3])
expert_b_small: torch.Size([4, 3])
route_ids_small: torch.Size([5])
logits_small: torch.Size([5, 3])
```

Mechanical meaning:

- `X_small[i]` is one input example.
- `route_ids_small[i]` says which expert handles `X_small[i]`.
- `expert_W_small[r]` is the classifier weight matrix for expert `r`.
- `logits_small[i]` will store three raw class scores for `X_small[i]`.

Important distinction:

```text
route_ids: expert IDs
labels/y:  class IDs
```

They are both integer vectors, but they are not the same thing.

#### 15.7.2 Boolean Mask Drill

Purpose:

Select only the examples routed to one expert.

Action:

Run this in `experiments.ipynb`.

```python
r = 2
mask = route_ids_small == r

print("route_ids_small:", route_ids_small)
print("mask:", mask)
print("X_small[mask] shape:", X_small[mask].shape)
print("selected routes:", route_ids_small[mask])
```

Expected pattern:

```text
mask: tensor([False,  True,  True, False, False])
X_small[mask] shape: torch.Size([2, 6])
selected routes: tensor([2, 2])
```

Mechanical meaning:

- `mask` has one boolean per example.
- `True` means "this row goes to expert `r`."
- `X_small[mask]` keeps only the rows handled by expert `2`.

#### 15.7.3 One-Expert Logit Drill

Purpose:

Run the selected rows through one expert's classifier.

Action:

Run this in `experiments.ipynb`.

```python
r = 2
mask = route_ids_small == r

X_for_r = X_small[mask]
W_for_r = expert_W_small[r]
b_for_r = expert_b_small[r]

logits_for_r = X_for_r @ W_for_r + b_for_r

print("X_for_r:", X_for_r.shape)
print("W_for_r:", W_for_r.shape)
print("b_for_r:", b_for_r.shape)
print("logits_for_r:", logits_for_r.shape)
```

Expected shape pattern:

```text
X_for_r: torch.Size([2, 6])
W_for_r: torch.Size([6, 3])
b_for_r: torch.Size([3])
logits_for_r: torch.Size([2, 3])
```

Mechanical meaning:

```text
[examples_for_r, features] @ [features, classes] -> [examples_for_r, classes]
```

So expert `2` produces one three-class logit row for each example routed to expert `2`.

#### 15.7.4 Fill-Back Drill

Purpose:

Write one expert's computed logits back into the correct rows of the shared output table.

Action:

Run this in `experiments.ipynb`.

```python
logits_small = torch.zeros(X_small.shape[0], expert_b_small.shape[1])

r = 2
mask = route_ids_small == r
logits_for_r = X_small[mask] @ expert_W_small[r] + expert_b_small[r]

logits_small[mask] = logits_for_r

print("filled row indices:", torch.where(mask)[0])
print("logits_small:", logits_small)
```

Mechanical meaning:

- `logits_for_r` has shape `[examples_for_r, classes]`.
- `logits_small[mask]` selects the same number of output rows.
- The assignment fills only rows routed to expert `r`.
- Other rows are still zero because no other expert has filled them yet.

#### 15.7.5 Full Loop Drill

Purpose:

Run the whole routed forward pass once, but not inside a function yet.

Action:

Run this in `experiments.ipynb`.

```python
logits_small = torch.zeros(X_small.shape[0], expert_b_small.shape[1])

for r in range(expert_W_small.shape[0]):
    mask = route_ids_small == r
    print("expert", r, "count", mask.sum().item())

    if mask.any():
        logits_small[mask] = X_small[mask] @ expert_W_small[r] + expert_b_small[r]

pred_small = logits_small.argmax(dim=1)

print("logits_small shape:", logits_small.shape)
print("pred_small shape:", pred_small.shape)
print("pred_small:", pred_small)
```

Expected shape pattern:

```text
logits_small shape: torch.Size([5, 3])
pred_small shape: torch.Size([5])
```

Mechanical meaning:

- The loop visits every expert ID: `0`, `1`, `2`, `3`.
- Each expert only processes rows whose `route_ids_small` equal that expert ID.
- Every valid input row gets exactly one output logit row.
- `argmax(dim=1)` is only for converting logits to predicted classes after logits exist.

Do not put `argmax` inside `routed_classification_logits`.

The model helper should return raw logits because `F.cross_entropy` needs raw logits.

#### 15.7.6 File Edit: Full Function

Action:

Add this function to `models.py`.

```python
def routed_classification_logits(X, expert_W, expert_b, route_ids):
    # X shape: [batch, features].
    # The output needs one row of class scores per input row.
    num_examples = X.shape[0]

    # expert_b shape is [regions, classes], so dimension 1 is num_classes.
    num_classes = expert_b.shape[1]

    # Allocate output logits before filling routed rows.
    # Shape: [batch, classes]
    logits = torch.zeros(num_examples, num_classes)

    # Loop over each expert/region ID.
    for r in range(expert_W.shape[0]):
        # Select examples routed to expert r.
        # mask shape: [batch]
        mask = route_ids == r

        # Only compute for experts that actually received examples.
        if mask.any():
            # X[mask] shape: [examples_for_r, features]
            # expert_W[r] shape: [features, classes]
            # expert_b[r] shape: [classes]
            # logits[mask] shape: [examples_for_r, classes]
            logits[mask] = X[mask] @ expert_W[r] + expert_b[r]

    # Return raw class scores, not softmax probabilities.
    return logits
```

Shape contract:

```text
X:         [batch, features]
expert_W:  [regions, features, classes]
expert_b:  [regions, classes]
route_ids: [batch]
logits:    [batch, classes]
```

Comparison with routed regression:

```text
regression expert_W[r]:     [features]
classification expert_W[r]: [features, classes]

regression output:          [batch]
classification output:      [batch, classes]
```

#### 15.7.7 Trace Checklist

Before moving on, trace the function with these questions:

```text
1. What does one row of X represent?
2. What does one value in route_ids represent?
3. What is the shape of expert_W[r]?
4. Why does X[mask] @ expert_W[r] produce class logits?
5. Why does logits[mask] have the same row count as X[mask]?
6. Why does this function return logits instead of predicted classes?
7. Why are route IDs not the same thing as class labels?
```

Common failure modes:

- using class labels as `route_ids`
- returning `argmax` predictions instead of logits
- applying softmax inside the helper
- forgetting `if mask.any()`
- making `expert_W` shape `[regions, classes, features]` instead of `[regions, features, classes]`

### 15.8 File Edit: Accuracy In `metrics.py`

Action:

Add this function to `metrics.py`.

```python
def accuracy(logits, y):
    predictions = logits.argmax(dim=1)
    return (predictions == y).float().mean()
```

Then rerun the notebook import/reload cell.

Expected import output:

```text
No SKIP lines for:
data.make_region_class_rules
data.make_sparse_classification_data
models.routed_classification_logits
metrics.accuracy
```

If those names still show `SKIP`, the notebook is using stale imports or the function was not added to the expected file.

### 15.9 Data Shape Drill

Action:

Run this in `experiments.ipynb` after editing files and rerunning the import cell.

```python
torch.manual_seed(0)

num_regions = 4
num_features = 6
num_classes = 3

region_table = make_region_table(num_regions, num_features)
true_W_class, true_b_class = make_region_class_rules(
    num_regions, num_features, num_classes
)
mixture = torch.tensor([0.25, 0.25, 0.25, 0.25])

X_cls, y_cls, region_ids_cls = make_sparse_classification_data(
    20, region_table, true_W_class, true_b_class, mixture
)

print("X_cls:", X_cls.shape)
print("y_cls:", y_cls.shape)
print("region_ids_cls:", region_ids_cls.shape)
print("labels:", y_cls[:10])
```

Expected shape pattern:

```text
X_cls: torch.Size([20, 6])
y_cls: torch.Size([20])
region_ids_cls: torch.Size([20])
```

Suspicious output:

- `y_cls` has shape `[20, 3]`: labels are wrong; labels should be class IDs, not one-hot rows.
- label values are outside `0..2`: `num_classes` or label generation is wrong.

### 15.10 Full Classification Experiment Cell

Purpose:

Compare global classification against oracle, random, and similarity-routed classifier experts.

Block grammar:

Do not read this as one large code blob.

Read it as five smaller blocks:

```text
1. Build one synthetic classification dataset.
2. Split the dataset into train/test tensors.
3. Define run_global_classifier().
4. Define run_routed_classifier(routing_type).
5. Run each condition and print rows.
```

The two helper functions have the same internal shape:

```text
initialize fresh trainable parameters
for epoch in range(200):
    compute logits
    compute cross-entropy loss
    backward
    sgd
evaluate without gradients
return plain Python numbers
```

The routed helper adds one extra step inside training:

```text
routing_type -> route_ids -> routed_classification_logits -> cross_entropy
```

Trace before running:

```text
global classifier:
    X_train @ W + b -> logits

routed classifier:
    route_ids choose expert rows
    routed_classification_logits(...) -> logits

both:
    logits + y_train -> cross_entropy
    logits.argmax(dim=1) -> accuracy
```

Action:

Run this as one cell in `experiments.ipynb`.

```python
# 15.10 Full Classification Experiment Cell

# Fix PyTorch randomness so the generated data and initialization are repeatable.
torch.manual_seed(0)

# Use four synthetic regions, so valid expert IDs are 0, 1, 2, and 3.
num_regions = 4

# Each input example has six numeric features, so X rows have length 6.
num_features = 6

# Each classifier predicts one of three possible classes: 0, 1, or 2.
num_classes = 3

# Build the routing prototypes.
# Shape: [4, 6], one six-feature prototype vector per region.
region_table = make_region_table(num_regions, num_features)

# Build the hidden answer-key classifier for each region.
# true_W_class shape: [4, 6, 3]
# true_b_class shape: [4, 3]
true_W_class, true_b_class = make_region_class_rules(
    num_regions, num_features, num_classes
)

# Balanced mixture means each synthetic region is sampled with equal probability.
mixture = torch.tensor([0.25, 0.25, 0.25, 0.25])

# Generate 500 classification examples.
# X_cls shape: [500, 6]
# y_cls shape: [500], with integer class IDs in 0..2
# region_ids_cls shape: [500], with true synthetic region IDs in 0..3
X_cls, y_cls, region_ids_cls = make_sparse_classification_data(
    500, region_table, true_W_class, true_b_class, mixture
)

# Split X, y, and region IDs together so every row keeps its matching label and region.
(
    X_train,
    y_train,
    train_region_ids,
    X_test,
    y_test,
    test_region_ids,
) = train_test_split_with_regions(X_cls, y_cls, region_ids_cls)


# Train one global softmax classifier.
# This baseline has no experts and no routing; one W,b pair must fit all regions.
def run_global_classifier():
    # W maps six input features to three class logits.
    # Shape: [features, classes] = [6, 3]
    W = torch.randn(num_features, num_classes, requires_grad=True)

    # b has one bias per class.
    # Shape: [classes] = [3]
    b = torch.zeros(num_classes, requires_grad=True)

    # Run 200 full-batch SGD steps on the training set.
    for epoch in range(200):
        # Compute raw class scores.
        # X_train shape: [train_examples, 6]
        # W shape: [6, 3]
        # logits shape: [train_examples, 3]
        logits = X_train @ W + b

        # Cross entropy compares raw logits against integer class labels.
        # y_train shape: [train_examples]
        # loss is a scalar.
        loss = F.cross_entropy(logits, y_train)

        # Compute gradients for W and b from the scalar loss.
        loss.backward()

        # Apply one manual SGD update and clear gradients.
        sgd([W, b], lr=0.05)

    # Evaluation should not build a gradient graph.
    with torch.no_grad():
        # Recompute logits on train and test data using the final W,b.
        train_logits = X_train @ W + b
        test_logits = X_test @ W + b

        # accuracy() uses argmax(dim=1) internally to convert logits to classes.
        train_acc = accuracy(train_logits, y_train)
        test_acc = accuracy(test_logits, y_test)

        # Keep test cross-entropy too, because accuracy hides confidence mistakes.
        test_loss = F.cross_entropy(test_logits, y_test)

    # Convert scalar tensors to Python floats for table printing.
    return train_acc.item(), test_acc.item(), test_loss.item()


# Train one routed classifier condition.
# routing_type decides where route_ids come from.
def run_routed_classifier(routing_type):
    # Create one classifier weight matrix per expert.
    # Shape: [regions, features, classes] = [4, 6, 3]
    expert_W = torch.randn(
        num_regions, num_features, num_classes, requires_grad=True
    )

    # Create one class-bias vector per expert.
    # Shape: [regions, classes] = [4, 3]
    expert_b = torch.zeros(num_regions, num_classes, requires_grad=True)

    # Train the routed experts for 200 full-batch epochs.
    for epoch in range(200):
        # Oracle routing uses the synthetic answer-key region IDs.
        # This is only possible in synthetic experiments.
        if routing_type == "oracle":
            route_ids = train_region_ids

        # Random routing ignores X and assigns each example to a random expert.
        # route_ids shape: [train_examples]
        elif routing_type == "random":
            route_ids = random_routes(X_train.shape[0], num_regions)

        # Similarity routing computes route IDs from X_train and region_table.
        # route_ids shape: [train_examples]
        elif routing_type == "similarity":
            route_ids = similarity_routes(X_train, region_table)

        # Compute routed logits.
        # Each example is processed by exactly one expert selected by route_ids.
        # logits shape: [train_examples, 3]
        logits = routed_classification_logits(
            X_train, expert_W, expert_b, route_ids
        )

        # Train against integer class labels.
        loss = F.cross_entropy(logits, y_train)

        # Backprop through only the expert rows used by the routed forward pass.
        loss.backward()

        # Update expert_W/expert_b and clear gradients.
        sgd([expert_W, expert_b], lr=0.05)

    # Evaluate the trained routed classifier.
    with torch.no_grad():
        # For oracle evaluation, test routes are the true synthetic test regions.
        if routing_type == "oracle":
            train_routes = train_region_ids
            test_routes = test_region_ids
            router_acc = 1.0

        # For random evaluation, create fresh random routes for train/test.
        elif routing_type == "random":
            train_routes = random_routes(X_train.shape[0], num_regions)
            test_routes = random_routes(X_test.shape[0], num_regions)

            # Router accuracy compares random routes against true test region IDs.
            router_acc = (test_routes == test_region_ids).float().mean().item()

        # For similarity evaluation, route each row by cosine similarity.
        elif routing_type == "similarity":
            train_routes = similarity_routes(X_train, region_table)
            test_routes = similarity_routes(X_test, region_table)

            # Router accuracy measures how often similarity recovers the true synthetic region.
            router_acc = (test_routes == test_region_ids).float().mean().item()

        # Compute train logits using the chosen train routes.
        train_logits = routed_classification_logits(
            X_train, expert_W, expert_b, train_routes
        )

        # Compute test logits using the chosen test routes.
        test_logits = routed_classification_logits(
            X_test, expert_W, expert_b, test_routes
        )

        # Convert logits to accuracy metrics.
        train_acc = accuracy(train_logits, y_train)
        test_acc = accuracy(test_logits, y_test)

        # Keep test cross-entropy as a loss metric.
        test_loss = F.cross_entropy(test_logits, y_test)

        # Count how many train examples each expert handled.
        # train_routes shape: [train_examples]
        # region_usage length: 4
        region_usage = torch.bincount(train_routes, minlength=num_regions).tolist()

    # Return one table row worth of metrics.
    return train_acc.item(), test_acc.item(), test_loss.item(), router_acc, region_usage


# Accumulate rows before printing so all conditions share one output format.
results = []

# Run the global baseline first.
train_acc, test_acc, test_loss = run_global_classifier()

# Global has no router, so router_accuracy and region_usage are None.
results.append([
    "global classifier",
    "none",
    train_acc,
    test_acc,
    test_loss,
    None,
    None,
])

# Run the same routed classifier code with three route sources.
for routing_type in ["oracle", "random", "similarity"]:
    # Train and evaluate a fresh routed classifier for this routing_type.
    train_acc, test_acc, test_loss, router_acc, region_usage = run_routed_classifier(
        routing_type
    )

    # Store the result row.
    results.append([
        "routed classifier",
        routing_type,
        train_acc,
        test_acc,
        test_loss,
        router_acc,
        region_usage,
    ])

# Print the comparison table one row at a time.
for row in results:
    print(row)
```

Output columns:

```text
model_type | routing_type | train_accuracy | test_accuracy | test_loss | router_accuracy | region_usage
```

Expected rough pattern:

- Random class baseline is about `1 / 3 = 0.333`.
- Oracle routed should usually be strong.
- Similarity routed should be better than random if router accuracy is high.
- Global classifier may be weaker than good routed experts because one classifier is trying to fit multiple region-specific rules.

Failure checks:

- `NameError: make_region_class_rules`: rerun import cell after editing `data.py`.
- `NameError: routed_classification_logits`: rerun import cell after editing `models.py`.
- `RuntimeError` from `F.cross_entropy`: check `logits.shape == [batch, classes]` and `y.shape == [batch]`.
- `IndexError: Target ... is out of bounds`: check labels are integers in `[0, num_classes - 1]`.

### 15.11 Checkpoint

You are ready to move on when you can explain:

- why classification labels have shape `[batch]`
- why logits have shape `[batch, classes]`
- why `F.cross_entropy(logits, y)` receives raw logits
- why `argmax(dim=1)` converts logits into predicted class IDs
- how routed classification differs from routed regression
- why classification `expert_W` has shape `[regions, features, classes]`

Suggested notes format:

```markdown
# 15 Checkpoint

* why classification labels have shape [batch]
> Each example has one correct class ID, so `y[i]` is the target class for `X[i]`.

* why logits have shape [batch, classes]
> Each example needs one raw score per possible class.

* why F.cross_entropy receives raw logits
> PyTorch's cross entropy applies the needed log-softmax internally, so passing softmax probabilities would duplicate that step.

* why argmax(dim=1) gives predictions
> `dim=1` is the class dimension, so argmax picks the highest-scoring class for each example.

* how routed classification differs from routed regression
> Routed regression returns one number per example. Routed classification returns one vector of class scores per example.

* why expert_W is [regions, features, classes]
> Each region has one classifier expert. For region `r`, `expert_W[r]` maps input features to class logits, so it has shape `[features, classes]`.
```

## 16. Phase 10: Top-2 Routing

### 16.1 Goal

Top-1 routing chooses one expert.

Top-2 routing chooses two experts and averages their predictions.

This phase asks a concrete question:

```text
Does activating more experts help, or does the extra expert add noise?
```

### 16.2 Exact Run Order

Use this order:

```text
1. Run the notebook import/reload cell.
2. Run the top-k shape drill.
3. Run the top-2 prediction table drills.
4. Add top2_routed_predict_regression to models.py.
5. Rerun the notebook import/reload cell.
6. Run the full top-1 vs top-2 experiment cell.
7. Write the checkpoint answers in notes.md.
```

### 16.3 Syntax Preflight: `route_topk(..., k=2)`

Action:

Run this in `experiments.ipynb`.

```python
torch.manual_seed(0)

X_small = torch.randn(5, 6)
region_table_small = make_region_table(4, 6)

top_ids, top_scores, scores = route_topk(X_small, region_table_small, k=2)

print("X_small:", X_small.shape)
print("region_table_small:", region_table_small.shape)
print("scores:", scores.shape)
print("top_ids:", top_ids.shape)
print("top_scores:", top_scores.shape)
print(top_ids)
```

Expected shape pattern:

```text
scores: torch.Size([5, 4])
top_ids: torch.Size([5, 2])
top_scores: torch.Size([5, 2])
```

Mechanical meaning:

- `top_ids[i, 0]` is the best expert for example `i`.
- `top_ids[i, 1]` is the second-best expert for example `i`.
- `top_ids[:, 0]` is a `[batch]` vector of first-choice routes.
- `top_ids[:, 1]` is a `[batch]` vector of second-choice routes.

### 16.4 File Edit: Top-2 Regression Predictor In `models.py`

This section is another block-grammar drill.

The repeated pattern is:

```text
ask router for two route columns
allocate two prediction columns
for each top-k slot j:
    pull route_ids = top_ids[:, j]
    fill preds[:, j] using the routed regression pattern
average the two columns
return averaged predictions and top_ids
```

The new piece is not routing itself.

The new piece is keeping two route columns and two prediction columns aligned.

#### 16.4.1 Top-2 Route Columns

Purpose:

Separate the `[batch, 2]` top-k table into two ordinary `[batch]` route vectors.

Action:

Run this in `experiments.ipynb` after the `16.3` top-k drill.

```python
first_routes = top_ids[:, 0]
second_routes = top_ids[:, 1]

print("top_ids:", top_ids.shape)
print("first_routes:", first_routes.shape, first_routes)
print("second_routes:", second_routes.shape, second_routes)
```

Expected shape pattern:

```text
top_ids: torch.Size([5, 2])
first_routes: torch.Size([5])
second_routes: torch.Size([5])
```

Mechanical meaning:

- `top_ids[:, 0]` can be used exactly like ordinary top-1 `route_ids`.
- `top_ids[:, 1]` is another ordinary route vector for the second-choice expert.
- Top-2 prediction repeats the routed regression fill pattern once per column.

#### 16.4.2 One Prediction Column Drill

Purpose:

Fill only one column of a two-column prediction table.

Action:

Run this in `experiments.ipynb`.

```python
torch.manual_seed(0)

expert_W_small = torch.randn(4, 6)
expert_b_small = torch.zeros(4)
preds_small = torch.zeros(X_small.shape[0], 2)

j = 0
route_ids = top_ids[:, j]

for r in range(expert_W_small.shape[0]):
    mask = route_ids == r

    if mask.any():
        preds_small[mask, j] = X_small[mask] @ expert_W_small[r] + expert_b_small[r]

print("route_ids:", route_ids.shape)
print("preds_small:", preds_small.shape)
print(preds_small)
```

Mechanical meaning:

- `j = 0` means fill the first-choice prediction column.
- `preds_small[:, 1]` stays zero because this drill has not filled the second-choice column yet.
- The inner loop is the same mask/fill grammar used in routed regression.

#### 16.4.3 Full Two-Column Drill

Purpose:

Fill both top-2 prediction columns, then average them.

Action:

Run this in `experiments.ipynb`.

```python
preds_small = torch.zeros(X_small.shape[0], 2)

for j in range(2):
    route_ids = top_ids[:, j]

    for r in range(expert_W_small.shape[0]):
        mask = route_ids == r

        if mask.any():
            preds_small[mask, j] = X_small[mask] @ expert_W_small[r] + expert_b_small[r]

top2_pred_small = preds_small.mean(dim=1)

print("preds_small:", preds_small.shape)
print("top2_pred_small:", top2_pred_small.shape)
```

Expected shape pattern:

```text
preds_small: torch.Size([5, 2])
top2_pred_small: torch.Size([5])
```

Mechanical meaning:

- `preds_small[i, 0]` is example `i`'s first expert prediction.
- `preds_small[i, 1]` is example `i`'s second expert prediction.
- `preds_small.mean(dim=1)` collapses the two expert predictions into one regression prediction per example.

#### 16.4.4 File Edit: Full Function

Action:

Add this function to `models.py`.

```python
def top2_routed_predict_regression(X, expert_W, expert_b, region_table):
    # Ask the router for the best two experts per input row.
    # top_ids shape: [batch, 2]
    top_ids, _, _ = route_topk(X, region_table, k=2)

    # Allocate two prediction columns per example.
    # Column 0 stores the first expert's prediction.
    # Column 1 stores the second expert's prediction.
    # preds shape: [batch, 2]
    preds = torch.zeros(X.shape[0], 2)

    # j selects which top-k slot we are filling: 0 for best expert, 1 for second-best.
    for j in range(2):
        # Pull one route column out of top_ids.
        # route_ids shape: [batch]
        route_ids = top_ids[:, j]

        # For this route column, compute predictions expert by expert.
        for r in range(expert_W.shape[0]):
            # mask selects examples whose j-th route is expert r.
            mask = route_ids == r

            # Skip experts that no examples selected in this slot.
            if mask.any():
                # X[mask] shape: [examples_for_r, features]
                # expert_W[r] shape: [features]
                # preds[mask, j] shape: [examples_for_r]
                preds[mask, j] = X[mask] @ expert_W[r] + expert_b[r]

    # Average the two expert predictions for each example.
    # preds.mean(dim=1) shape: [batch]
    return preds.mean(dim=1), top_ids
```

Then rerun the notebook import/reload cell.

Expected import output:

```text
No SKIP line for models.top2_routed_predict_regression
```

Trace checklist:

```text
1. Why does top_ids have two columns?
2. Why is top_ids[:, j] a normal route_ids vector?
3. What does preds[i, 0] store?
4. What does preds[i, 1] store?
5. Why does preds.mean(dim=1) return shape [batch]?
6. Why does this function return top_ids as well as predictions?
```

### 16.5 Full Top-1 Vs Top-2 Experiment Cell

Purpose:

Train the usual similarity-routed regression experts, then evaluate the same trained experts with top-1 and top-2 prediction.

Block grammar:

Read this full cell as four blocks:

```text
1. Build one regression dataset.
2. Train ordinary top-1 similarity-routed experts.
3. Evaluate the same trained experts two ways:
       top-1 prediction
       top-2 averaged prediction
4. Print two comparable result rows.
```

The important experiment design detail:

```text
training uses top-1
evaluation compares top-1 vs top-2
```

So if top-2 improves or worsens the result, that change comes from the evaluation-time prediction rule, not from retraining different experts.

Action:

Run this as one cell in `experiments.ipynb`.

```python
# 16.5 Top-1 vs Top-2 Regression Experiment

# Fix randomness so the data, initialization, and routes are reproducible.
torch.manual_seed(0)

# Use four routed regression experts.
num_regions = 4

# Each input row has six features.
num_features = 6

# Create the router's prototype table.
# Shape: [4, 6]
region_table = make_region_table(num_regions, num_features)

# Create the hidden true regression rule for each synthetic region.
# true_W shape: [4, 6]
# true_b shape: [4]
true_W, true_b = make_region_rules(num_regions, num_features)

# Generate a balanced dataset so every region should appear often enough.
mixture = torch.tensor([0.25, 0.25, 0.25, 0.25])

# Generate one multi-region regression dataset.
# X shape: [500, 6]
# y shape: [500]
# region_ids shape: [500]
X, y, region_ids = make_sparse_regression_data(
    500, region_table, true_W, true_b, mixture
)

# Split examples, labels, and true region IDs together.
(
    X_train,
    y_train,
    train_region_ids,
    X_test,
    y_test,
    test_region_ids,
) = train_test_split_with_regions(X, y, region_ids)

# Create trainable expert weights.
# Shape: [regions, features] = [4, 6]
expert_W = torch.randn(num_regions, num_features, requires_grad=True)

# Create one trainable bias per expert.
# Shape: [regions] = [4]
expert_b = torch.zeros(num_regions, requires_grad=True)

# Train experts using ordinary top-1 similarity routing.
# Top-2 is only used later at evaluation time in this experiment.
for epoch in range(200):
    # route_ids shape: [train_examples]
    # Each value is the single nearest region/expert.
    route_ids = similarity_routes(X_train, region_table)

    # Compute MSE using only the expert selected for each training example.
    loss = routed_regression_loss(
        X_train, y_train, expert_W, expert_b, route_ids
    )

    # Build gradients for the active expert rows.
    loss.backward()

    # Update expert parameters and clear gradients.
    sgd([expert_W, expert_b], lr=0.03)

# Evaluation should not track gradients.
with torch.no_grad():
    # Top-1 prediction uses the same single-expert route used during training.
    # top1_train_pred shape: [train_examples]
    # top1_train_routes shape: [train_examples]
    top1_train_pred, top1_train_routes = routed_predict_regression(
        X_train, expert_W, expert_b, region_table
    )

    # Evaluate top-1 prediction on test examples.
    top1_test_pred, top1_test_routes = routed_predict_regression(
        X_test, expert_W, expert_b, region_table
    )

    # Top-2 prediction asks for two experts per example and averages their outputs.
    # top2_train_pred shape: [train_examples]
    # train_top2_ids shape: [train_examples, 2]
    top2_train_pred, train_top2_ids = top2_routed_predict_regression(
        X_train, expert_W, expert_b, region_table
    )

    # Same top-2 evaluation on test examples.
    # test_top2_ids shape: [test_examples, 2]
    top2_test_pred, test_top2_ids = top2_routed_predict_regression(
        X_test, expert_W, expert_b, region_table
    )

    # Plain prediction MSE for top-1.
    top1_train_loss = squared_loss(top1_train_pred, y_train)
    top1_test_loss = squared_loss(top1_test_pred, y_test)

    # Plain prediction MSE for top-2 averaging.
    top2_train_loss = squared_loss(top2_train_pred, y_train)
    top2_test_loss = squared_loss(top2_test_pred, y_test)

    # Top-1 router accuracy checks whether the one selected route equals the true region.
    top1_router_acc = (top1_test_routes == test_region_ids).float().mean()

    # Top-2 router accuracy checks whether the true region appears in either slot.
    # test_region_ids[:, None] changes [test_examples] to [test_examples, 1]
    # so it can broadcast against test_top2_ids with shape [test_examples, 2].
    top2_router_acc = (test_top2_ids == test_region_ids[:, None]).any(dim=1).float().mean()

    # Count how many examples each expert handled under top-1 routing.
    # Length is 4 because minlength=num_regions.
    top1_usage = torch.bincount(top1_train_routes, minlength=num_regions).tolist()

    # Count top-2 expert usage.
    # train_top2_ids has two routes per example, so reshape(-1) flattens [batch, 2]
    # into [batch * 2] before counting expert IDs.
    top2_usage = torch.bincount(train_top2_ids.reshape(-1), minlength=num_regions).tolist()

# Build two output rows with the same column order.
results = [
    [
        "top-1 similarity",
        top1_train_loss.item(),
        top1_test_loss.item(),
        top1_router_acc.item(),
        top1_usage,
    ],
    [
        "top-2 similarity average",
        top2_train_loss.item(),
        top2_test_loss.item(),
        top2_router_acc.item(),
        top2_usage,
    ],
]

# Print the comparison rows.
for row in results:
    print(row)
```

Output columns:

```text
routing_type | train_loss | test_loss | router_accuracy | region_usage
```

Expected rough pattern:

- Top-2 router accuracy should be at least as high as top-1 router accuracy because the correct region only needs to appear in either of two slots.
- Top-2 prediction loss may improve or worsen.
- If the second expert is often wrong, averaging can hurt even when top-2 router accuracy is higher.

Failure checks:

- `NameError: top2_routed_predict_regression`: rerun import cell after editing `models.py`.
- `RuntimeError` around `test_region_ids[:, None]`: check that `test_region_ids` has shape `[batch]`.
- Bad `top2_usage` length: check `minlength=num_regions`.

### 16.6 Checkpoint

You are ready to move on when you can explain:

- why `top_ids` has shape `[batch, 2]`
- why `top_ids[:, 0]` and `top_ids[:, 1]` each have shape `[batch]`
- why top-2 router accuracy can improve while prediction loss gets worse
- why "more active experts" is not automatically better

## 17. Phase 11: Local Update Gate

### 17.1 Goal

This phase adds a crude update gate:

```text
if batch loss is high enough:
    allow the model to update
else:
    skip the optimizer step
```

This is a toy version of:

```text
local participation x global salience
```

It is not biologically realistic. It is a mechanical experiment about when a system should write updates.

### 17.2 Exact Run Order

Use this order:

```text
1. Run the notebook import/reload cell.
2. Run the threshold drill.
3. Run the gated helper grammar drill.
4. Run the full gated-update experiment cell.
5. Write the checkpoint answers in notes.md.
```

### 17.3 Syntax Preflight: Threshold Gate

Action:

Run this in `experiments.ipynb`.

```python
loss_values = [1.2, 0.7, 0.3]
threshold = 0.5

for loss_value in loss_values:
    if loss_value > threshold:
        decision = "update"
    else:
        decision = "skip"

    print(loss_value, decision)
```

Expected output:

```text
1.2 update
0.7 update
0.3 skip
```

Mechanical meaning:

- The gate uses `loss.item()` during training because the decision is ordinary Python control flow.
- If the gate skips, there is no `backward()` call and no `sgd(...)` call for that epoch.

### 17.4 Full Gated-Update Experiment Cell

Purpose:

Compare normal routed regression against threshold-gated routed regression.

Block grammar:

The full cell has two loops at different levels:

```text
inner loop:
    one model trains for 200 possible epochs
    threshold decides whether each epoch updates

outer loop:
    run a fresh model for each threshold condition
    collect one result row per condition
```

The helper function has this shape:

```text
def run_gated_similarity_regression(threshold):
    initialize fresh expert_W, expert_b
    update_count = 0

    for epoch in range(200):
        compute route_ids
        compute loss

        if threshold is None or loss.item() > threshold:
            backward
            sgd
            update_count += 1

    evaluate final model
    return update_count and metrics
```

#### 17.4.1 Gated Helper Grammar Drill

Purpose:

Separate "possible epochs" from "actual updates."

Action:

Run this in `experiments.ipynb`.

```python
threshold = 0.5
loss_values = [1.2, 0.4, 0.8, 0.3]
update_count = 0

for epoch, loss_value in enumerate(loss_values):
    if loss_value > threshold:
        update_count += 1
        decision = "backward + sgd"
    else:
        decision = "skip"

    print(epoch, loss_value, decision, "updates so far:", update_count)
```

Expected pattern:

```text
0 1.2 backward + sgd updates so far: 1
1 0.4 skip updates so far: 1
2 0.8 backward + sgd updates so far: 2
3 0.3 skip updates so far: 2
```

Mechanical meaning:

- The loop still runs four possible epochs.
- Only two of those epochs update parameters.
- `update_count` measures optimizer steps, not loop iterations.

#### 17.4.2 Threshold Condition Table

Purpose:

See why `threshold=None` means ordinary training.

Action:

Run this in `experiments.ipynb`.

```python
loss_value = 0.3

for threshold in [None, 0.2, 0.5]:
    should_update = threshold is None or loss_value > threshold
    print("threshold:", threshold, "should_update:", should_update)
```

Mechanical meaning:

- `threshold is None` bypasses the comparison.
- A numeric threshold only updates when the current loss is larger than the threshold.
- High thresholds can skip too many updates.

Action:

Run this as one cell in `experiments.ipynb`.

```python
# 17.4 Local Update Gate Experiment

# Fix randomness so all threshold conditions start from reproducible data.
torch.manual_seed(0)

# Use four routed experts.
num_regions = 4

# Each input has six features.
num_features = 6

# Build the router prototype table.
# Shape: [4, 6]
region_table = make_region_table(num_regions, num_features)

# Build the hidden regression rule for each region.
# true_W shape: [4, 6]
# true_b shape: [4]
true_W, true_b = make_region_rules(num_regions, num_features)

# Use a balanced region mixture for this gate experiment.
mixture = torch.tensor([0.25, 0.25, 0.25, 0.25])

# Generate a standard multi-region regression dataset.
X, y, region_ids = make_sparse_regression_data(
    500, region_table, true_W, true_b, mixture
)

# Split X, y, and true region IDs together.
(
    X_train,
    y_train,
    train_region_ids,
    X_test,
    y_test,
    test_region_ids,
) = train_test_split_with_regions(X, y, region_ids)


# Train one similarity-routed model with a specific update threshold.
# threshold=None means ordinary training: always update.
def run_gated_similarity_regression(threshold):
    # Create one trainable weight vector per expert.
    # Shape: [regions, features] = [4, 6]
    expert_W = torch.randn(num_regions, num_features, requires_grad=True)

    # Create one trainable bias per expert.
    # Shape: [regions] = [4]
    expert_b = torch.zeros(num_regions, requires_grad=True)

    # Count how many epochs actually performed backward + SGD.
    update_count = 0

    # Run 200 possible training epochs.
    for epoch in range(200):
        # Route each training example by similarity.
        # route_ids shape: [train_examples]
        route_ids = similarity_routes(X_train, region_table)

        # Compute the routed training loss for this epoch.
        # loss is a scalar tensor.
        loss = routed_regression_loss(
            X_train, y_train, expert_W, expert_b, route_ids
        )

        # If threshold is None, always update.
        # Otherwise, convert loss to a Python float and compare against threshold.
        if threshold is None or loss.item() > threshold:
            # Only this branch builds gradients.
            loss.backward()

            # Only this branch changes expert_W and expert_b.
            sgd([expert_W, expert_b], lr=0.03)

            # Track how many optimizer updates actually happened.
            update_count += 1

        # If the branch is skipped, no backward pass occurs and parameters stay unchanged.

    # Evaluate the final model without tracking gradients.
    with torch.no_grad():
        # routed_predict_regression computes similarity routes internally.
        train_pred, train_routes = routed_predict_regression(
            X_train, expert_W, expert_b, region_table
        )

        # Same prediction path on the test set.
        test_pred, test_routes = routed_predict_regression(
            X_test, expert_W, expert_b, region_table
        )

        # Plain train/test MSE, not including any threshold logic.
        train_loss = squared_loss(train_pred, y_train)
        test_loss = squared_loss(test_pred, y_test)

        # Measure whether the similarity router is still assigning correct regions.
        router_acc = (test_routes == test_region_ids).float().mean().item()

        # Break test MSE down by true region to see uneven failures.
        per_region_loss = per_region_mse(
            test_pred, y_test, test_region_ids, num_regions
        )

    # Return one result row worth of values.
    return (
        update_count,
        train_loss.item(),
        test_loss.item(),
        router_acc,
        per_region_loss,
    )


# Compare ordinary training against increasingly strict update thresholds.
thresholds = [None, 0.2, 0.5, 1.0]

# Collect one row per threshold.
results = []

# Run a fresh model for each threshold condition.
for threshold in thresholds:
    # Train/evaluate one condition.
    update_count, train_loss, test_loss, router_acc, per_region_loss = run_gated_similarity_regression(
        threshold
    )

    # Store threshold, number of actual updates, and final metrics.
    results.append([
        threshold,
        update_count,
        train_loss,
        test_loss,
        router_acc,
        per_region_loss,
    ])

# Print comparison rows.
for row in results:
    print(row)
```

Output columns:

```text
threshold | number_of_updates | train_loss | test_loss | router_accuracy | per_region_test_loss
```

Expected rough pattern:

- `threshold = None` should update 200 times.
- Very low thresholds should behave close to normal training.
- Very high thresholds may skip too many updates and hurt learning.
- A useful threshold would reduce updates without badly hurting test loss.

Failure checks:

- `NameError: per_region_mse`: add it to `metrics.py` from Phase 14 and rerun the import cell.
- `number_of_updates = 0` for every threshold: thresholds are too high or loss is already below threshold.
- Loss does not change at all: check that `loss.backward()` and `sgd(...)` are inside the update branch.

### 17.5 Checkpoint

You are ready to move on when you can explain:

- why this gate is batch-level rather than per-example
- why skipped epochs do not update parameters
- why a low threshold behaves like ordinary training
- why a high threshold can cause underfitting
- why this is only a toy analog of local gated learning

## 18. Phase 12: Optional Homeostatic Scaling

### 18.1 Goal

Homeostatic scaling is a stabilizer.

In this project, use the simplest version:

```text
periodically rescale each expert weight vector toward a target norm
```

This is different from weight decay:

```text
weight decay: changes the loss objective
homeostatic scaling: changes weights outside the loss objective
```

### 18.2 Exact Run Order

Use this order:

```text
1. Run the norm/rescale drill.
2. Run the in-place rescale grammar drill.
3. Add rescale_expert_weights to train.py.
4. Rerun the notebook import/reload cell.
5. Run the full scaling experiment cell.
6. Write the checkpoint answers in notes.md.
```

### 18.3 Syntax Preflight: Norms And Rescaling

Action:

Run this in `experiments.ipynb`.

```python
W_small = torch.tensor([
    [3.0, 4.0],
    [6.0, 8.0],
])

norms = W_small.norm(dim=1, keepdim=True)
target_norm = 1.0
scale = target_norm / (norms + 1e-8)
W_scaled = W_small * scale

print("norms before:", W_small.norm(dim=1))
print("scale:", scale)
print("norms after:", W_scaled.norm(dim=1))
```

Expected output pattern:

```text
norms before: tensor([ 5., 10.])
norms after: tensor([1., 1.])
```

Mechanical meaning:

- `dim=1` computes one norm per expert row.
- `keepdim=True` keeps shape `[regions, 1]`, so scaling broadcasts across features.

### 18.4 File Edit: `rescale_expert_weights` In `train.py`

This helper is short, but it introduces a new kind of operation:

```text
change the parameter tensor directly outside autograd
```

That is why the function uses `with torch.no_grad()`.

#### 18.4.1 In-Place Rescale Grammar Drill

Purpose:

Distinguish "create a scaled copy" from "modify the original tensor."

Action:

Run this in `experiments.ipynb`.

```python
W_demo = torch.tensor([
    [3.0, 4.0],
    [6.0, 8.0],
])

norms = W_demo.norm(dim=1, keepdim=True)
scale = 1.0 / (norms + 1e-8)

W_copy = W_demo * scale

print("original after copy scaling:", W_demo)
print("scaled copy:", W_copy)

W_demo *= scale

print("original after in-place scaling:", W_demo)
print("row norms:", W_demo.norm(dim=1))
```

Mechanical meaning:

- `W_copy = W_demo * scale` creates a new tensor.
- `W_demo *= scale` changes `W_demo` itself.
- `rescale_expert_weights` uses the in-place version because it is meant to modify trained expert weights.

#### 18.4.2 File Edit: Full Function

Action:

Add this function to `train.py`.

```python
def rescale_expert_weights(expert_W, target_norm=1.0):
    # This function directly changes expert_W values, so do it outside autograd.
    with torch.no_grad():
        # Compute one L2 norm per expert row.
        # If expert_W shape is [regions, features], norms shape is [regions, 1].
        norms = expert_W.norm(dim=1, keepdim=True)

        # Compute one multiplier per expert row.
        # The small 1e-8 prevents division by zero.
        scale = target_norm / (norms + 1e-8)

        # Multiply each expert row by its own scale.
        # Broadcasting works because scale shape is [regions, 1].
        expert_W *= scale
```

Then rerun the notebook import/reload cell.

Expected import output:

```text
No SKIP line for train.rescale_expert_weights
```

### 18.5 Full Scaling Experiment Cell

Purpose:

Compare ordinary routed regression, weight decay, homeostatic scaling, and both together.

Block grammar:

Read this full cell as:

```text
1. Build one regression dataset.
2. Define run_scaling_experiment(wd, use_scaling).
3. Create a config table.
4. Run a fresh model for each config.
5. Print comparable rows.
```

The condition table controls two independent switches:

```text
wd:
    adds weight decay to the loss before backward()

use_scaling:
    directly rescales expert_W after sgd()
```

Inside each epoch, the order matters:

```text
prediction loss
plus optional weight decay
backward
sgd
optional direct rescale
```

Action:

Run this as one cell in `experiments.ipynb`.

```python
# 18.5 Weight Decay vs Homeostatic Scaling

# Fix randomness so each condition is easier to compare.
torch.manual_seed(0)

# Use four routed experts.
num_regions = 4

# Each example has six features.
num_features = 6

# Create the router prototype table.
# Shape: [4, 6]
region_table = make_region_table(num_regions, num_features)

# Create hidden region-specific regression rules.
true_W, true_b = make_region_rules(num_regions, num_features)

# Use a balanced dataset for this comparison.
mixture = torch.tensor([0.25, 0.25, 0.25, 0.25])

# Generate one synthetic regression dataset.
X, y, region_ids = make_sparse_regression_data(
    500, region_table, true_W, true_b, mixture
)

# Split examples, labels, and true region IDs together.
(
    X_train,
    y_train,
    train_region_ids,
    X_test,
    y_test,
    test_region_ids,
) = train_test_split_with_regions(X, y, region_ids)


# Train one routed model under one regularization/scaling condition.
# wd controls the weight-decay penalty.
# use_scaling controls direct periodic norm rescaling.
def run_scaling_experiment(wd, use_scaling):
    # expert_W has one six-feature weight vector per expert.
    # Shape: [4, 6]
    expert_W = torch.randn(num_regions, num_features, requires_grad=True)

    # expert_b has one scalar bias per expert.
    # Shape: [4]
    expert_b = torch.zeros(num_regions, requires_grad=True)

    # Run 200 full-batch training epochs.
    for epoch in range(200):
        # Route each training example to the nearest expert.
        route_ids = similarity_routes(X_train, region_table)

        # Compute routed prediction MSE.
        loss = routed_regression_loss(
            X_train, y_train, expert_W, expert_b, route_ids
        )

        # Add weight decay to the training objective.
        # If wd is 0.0, this adds nothing.
        # This penalty contributes gradients during loss.backward().
        loss = loss + wd * l2_penalty(expert_W)

        # Backpropagate through prediction loss plus optional weight penalty.
        loss.backward()

        # Apply the gradient update and clear gradients.
        sgd([expert_W, expert_b], lr=0.03)

        # Homeostatic scaling is not part of the loss.
        # It directly modifies expert_W after the optimizer step.
        # epoch % 20 == 0 means epochs 0, 20, 40, ... trigger rescaling.
        if use_scaling and epoch % 20 == 0:
            rescale_expert_weights(expert_W, target_norm=1.0)

    # Evaluate final model without building gradient history.
    with torch.no_grad():
        # Predict train outputs using similarity routing.
        train_pred, train_routes = routed_predict_regression(
            X_train, expert_W, expert_b, region_table
        )

        # Predict test outputs using similarity routing.
        test_pred, test_routes = routed_predict_regression(
            X_test, expert_W, expert_b, region_table
        )

        # Report plain MSE only.
        # Do not include the weight-decay penalty in evaluation loss.
        train_loss = squared_loss(train_pred, y_train)
        test_loss = squared_loss(test_pred, y_test)

        # Overall norm of the whole expert_W tensor.
        weight_norm = expert_W.norm().item()

        # One norm per expert row.
        # Shape before .tolist(): [4]
        per_expert_norms = expert_W.norm(dim=1).tolist()

    # Return metrics for this condition.
    return train_loss.item(), test_loss.item(), weight_norm, per_expert_norms


# Each row defines one experimental condition:
# [condition name, weight decay coefficient, whether to apply scaling]
configs = [
    ["none", 0.0, False],
    ["weight decay", 0.1, False],
    ["homeostatic scaling", 0.0, True],
    ["weight decay + scaling", 0.1, True],
]

# Collect one output row per condition.
results = []

# Train and evaluate a fresh model for each condition.
for name, wd, use_scaling in configs:
    # Run one condition.
    train_loss, test_loss, weight_norm, per_expert_norms = run_scaling_experiment(
        wd, use_scaling
    )

    # Store condition metadata plus final metrics.
    results.append([
        name,
        wd,
        use_scaling,
        train_loss,
        test_loss,
        weight_norm,
        per_expert_norms,
    ])

# Print the comparison table.
for row in results:
    print(row)
```

Output columns:

```text
condition | wd | use_scaling | train_loss | test_loss | weight_norm | per_expert_norms
```

Expected rough pattern:

- Scaling should push per-expert norms closer to the target.
- Weight decay should usually reduce norms through the training objective.
- Scaling and weight decay can affect loss differently because they act through different mechanisms.

Failure checks:

- `NameError: rescale_expert_weights`: rerun import cell after editing `train.py`.
- Per-expert norms do not change under scaling: check `epoch % 20 == 0` block is inside the training loop.
- All norms become zero or explode: check `target_norm` and learning rate.

### 18.6 Checkpoint

You are ready to move on when you can explain:

- why `expert_W.norm(dim=1)` gives one norm per expert
- why `keepdim=True` matters for broadcasting
- why homeostatic scaling is not the same as weight decay
- why smaller weight norm is not automatically better

## 19. Phase 13: Optional Replay Buffer

### 19.1 Goal

Replay stores old examples and mixes them into later training.

The question:

```text
When the training distribution moves from old regions to new regions,
does replay help preserve old-region performance?
```

Replay is different from sparse routing:

```text
sparse routing: avoid touching unrelated experts
replay: keep old examples in the training stream
```

### 19.2 Exact Run Order

Use this order:

```text
1. Run the torch.cat drill.
2. Run the replay storage grammar drills.
3. Add ReplayBuffer to train.py.
4. Rerun the notebook import/reload cell.
5. Run the replay buffer drill.
6. Run the full curriculum experiment cell.
7. Write the checkpoint answers in notes.md.
```

### 19.3 Syntax Preflight: `torch.cat`

Action:

Run this in `experiments.ipynb`.

```python
current_X = torch.randn(3, 6)
replay_X = torch.randn(2, 6)

mixed_X = torch.cat([current_X, replay_X], dim=0)

print("current_X:", current_X.shape)
print("replay_X:", replay_X.shape)
print("mixed_X:", mixed_X.shape)
```

Expected output:

```text
current_X: torch.Size([3, 6])
replay_X: torch.Size([2, 6])
mixed_X: torch.Size([5, 6])
```

Mechanical meaning:

- `dim=0` stacks more examples into the batch.
- Feature count must match.

### 19.4 File Edit: `ReplayBuffer` In `train.py`

This section is a block-grammar drill for a small class.

The repeated pattern is:

```text
add:
    store individual detached rows
    trim lists to max_size

sample:
    choose random list positions
    stack selected rows back into tensors
```

The buffer stores examples as Python lists because it is easier to append and trim one example at a time.

#### 19.4.1 Store Individual Rows Drill

Purpose:

See what `ReplayBuffer.add` does before it is inside a class.

Action:

Run this in `experiments.ipynb`.

```python
X_demo = torch.arange(24, dtype=torch.float32).reshape(4, 6)
y_demo = torch.tensor([0.0, 1.0, 2.0, 3.0])

stored_X = []
stored_y = []

for i in range(X_demo.shape[0]):
    stored_X.append(X_demo[i].detach().clone())
    stored_y.append(y_demo[i].detach().clone())

print("stored rows:", len(stored_X))
print("first stored X row:", stored_X[0])
print("first stored y:", stored_y[0])
```

Mechanical meaning:

- The list stores one example row at a time.
- Each `stored_X[i]` has shape `[features]`, not `[1, features]`.
- `detach().clone()` turns a training tensor row into stored data.

#### 19.4.2 Max-Size Trim Drill

Purpose:

See how the buffer forgets oldest examples when it gets too large.

Action:

Run this in `experiments.ipynb`.

```python
max_size = 2

stored_X = stored_X[-max_size:]
stored_y = stored_y[-max_size:]

print("stored rows after trim:", len(stored_X))
print("remaining y values:", stored_y)
```

Expected pattern:

```text
stored rows after trim: 2
```

Mechanical meaning:

- `stored_X[-max_size:]` keeps the last `max_size` rows.
- Older rows are dropped.
- This buffer is a recent-example replay buffer, not an all-history archive.

#### 19.4.3 Sample And Stack Drill

Purpose:

Rebuild a tensor batch from sampled list positions.

Action:

Run this in `experiments.ipynb`.

```python
idx = torch.tensor([1, 0, 1])

sample_X = torch.stack([stored_X[int(i)] for i in idx])
sample_y = torch.stack([stored_y[int(i)] for i in idx])

print("idx:", idx)
print("sample_X:", sample_X.shape)
print("sample_y:", sample_y.shape)
```

Expected shape pattern:

```text
sample_X: torch.Size([3, 6])
sample_y: torch.Size([3])
```

Mechanical meaning:

- `idx` chooses positions inside the stored Python lists.
- `torch.stack(...)` turns individual stored rows back into a batch.
- Sampling with repeated indices is allowed.

#### 19.4.4 File Edit: Full Class

Action:

Add this class to `train.py`.

```python
class ReplayBuffer:
    def __init__(self, max_size):
        # Store at most this many examples.
        self.max_size = max_size

        # Python list of individual X rows.
        # Each stored item has shape [features].
        self.X = []

        # Python list of individual labels.
        # For regression, each stored item is a scalar tensor.
        self.y = []

    def add(self, X, y):
        # Add examples one row at a time so the buffer can keep/replay individual examples.
        for i in range(X.shape[0]):
            # detach() removes old computation graph history.
            # clone() gives the buffer its own copy of the tensor values.
            self.X.append(X[i].detach().clone())
            self.y.append(y[i].detach().clone())

        # Keep only the most recent max_size examples.
        # If max_size is 300 and 500 examples were added, the oldest 200 are dropped.
        self.X = self.X[-self.max_size:]
        self.y = self.y[-self.max_size:]

    def sample(self, batch_size):
        # n is the number of examples currently stored.
        n = len(self.X)

        # Randomly choose batch_size integer positions from 0 to n - 1.
        # idx shape: [batch_size]
        idx = torch.randint(0, n, (batch_size,))

        # Rebuild a batch tensor from the sampled stored rows.
        # X shape: [batch_size, features]
        X = torch.stack([self.X[int(i)] for i in idx])

        # Rebuild the matching label tensor.
        # y shape: [batch_size]
        y = torch.stack([self.y[int(i)] for i in idx])

        # Return replay examples and labels in matching order.
        return X, y
```

Then rerun the notebook import/reload cell.

Expected import output:

```text
No SKIP line for train.ReplayBuffer
```

### 19.5 Syntax Preflight: Replay Add And Sample

Action:

Run this in `experiments.ipynb`.

```python
buffer = ReplayBuffer(max_size=5)

X_demo = torch.randn(8, 6)
y_demo = torch.randn(8)

buffer.add(X_demo, y_demo)
sample_X, sample_y = buffer.sample(3)

print("stored examples:", len(buffer.X))
print("sample_X:", sample_X.shape)
print("sample_y:", sample_y.shape)
```

Expected output:

```text
stored examples: 5
sample_X: torch.Size([3, 6])
sample_y: torch.Size([3])
```

Failure check:

- If `sample()` fails with `n = 0`, the buffer was sampled before anything was added.

### 19.6 Full Curriculum Replay Experiment Cell

Purpose:

Train first on old regions, then train on new regions. Compare later old-region performance with and without replay.

Block grammar:

Read this full cell as a two-stage curriculum:

```text
data setup:
    old training distribution
    new training distribution
    old test distribution
    new test distribution

stage 1:
    train on old data

stage 2 without replay:
    train on new data only

stage 2 with replay:
    train on new data plus sampled old examples

evaluation:
    measure old-test loss and new-test loss
```

The experiment question is not just "which loss is lower?"

It is:

```text
Does replay reduce forgetting on old_test_loss without ruining new_test_loss?
```

#### 19.6.1 Batch-Mixing Grammar Drill

Purpose:

See exactly what changes when replay is enabled in stage 2.

Action:

Run this in `experiments.ipynb`.

```python
new_X_demo = torch.randn(5, 6)
new_y_demo = torch.randn(5)

replay_X_demo = torch.randn(2, 6)
replay_y_demo = torch.randn(2)

batch_X = torch.cat([new_X_demo, replay_X_demo], dim=0)
batch_y = torch.cat([new_y_demo, replay_y_demo], dim=0)

print("new_X_demo:", new_X_demo.shape)
print("replay_X_demo:", replay_X_demo.shape)
print("batch_X:", batch_X.shape)
print("batch_y:", batch_y.shape)
```

Expected shape pattern:

```text
batch_X: torch.Size([7, 6])
batch_y: torch.Size([7])
```

Mechanical meaning:

- Replay adds more rows to the current training batch.
- It does not add more features.
- `batch_X` and `batch_y` must grow together along `dim=0`.

Action:

Run this as one cell in `experiments.ipynb`.

```python
# 19.6 Curriculum Replay Experiment

# Set PyTorch's random generator so random tensors, routes, and replay samples are repeatable.
torch.manual_seed(0)

# We will create four regions, so valid expert/region IDs are 0, 1, 2, and 3.
num_regions = 4

# Every input row x will have six numeric features, so X has shape [examples, 6].
num_features = 6

# Create one prototype vector per region; shape is [4, 6].
# similarity_routes later compares each input row against these four rows.
region_table = make_region_table(num_regions, num_features)

# Create the hidden answer-key regression weights and biases.
# true_W shape is [4, 6], so true_W[r] is the hidden rule for region r.
# true_b shape is [4], so true_b[r] is the hidden bias for region r.
true_W, true_b = make_region_rules(num_regions, num_features)

# This probability vector controls how often each region appears in old data.
# Regions 0 and 1 are common; regions 2 and 3 are rare.
old_mixture = torch.tensor([0.45, 0.45, 0.05, 0.05])

# This probability vector controls how often each region appears in new data.
# Now regions 2 and 3 are common; this intentionally changes the training distribution.
new_mixture = torch.tensor([0.05, 0.05, 0.45, 0.45])

# Generate old training examples.
# old_X_train shape: [500, 6]
# old_y_train shape: [500]
# old_region_ids_train shape: [500], with mostly 0s and 1s.
old_X_train, old_y_train, old_region_ids_train = make_sparse_regression_data(
    500, region_table, true_W, true_b, old_mixture
)

# Generate new training examples.
# Same shapes as old training data, but region IDs are mostly 2s and 3s.
new_X_train, new_y_train, new_region_ids_train = make_sparse_regression_data(
    500, region_table, true_W, true_b, new_mixture
)

# Generate old-distribution test data.
# This lets us ask: after learning new data, did the model forget old regions?
old_X_test, old_y_test, old_region_ids_test = make_sparse_regression_data(
    200, region_table, true_W, true_b, old_mixture
)

# Generate new-distribution test data.
# This lets us ask: did the model learn the later/new distribution?
new_X_test, new_y_test, new_region_ids_test = make_sparse_regression_data(
    200, region_table, true_W, true_b, new_mixture
)


# This helper performs exactly one full-batch optimization step.
def train_one_routed_epoch(expert_W, expert_b, batch_X, batch_y):
    # Compute route_ids from the batch itself.
    # batch_X shape is [batch_size, 6].
    # route_ids shape is [batch_size], where each value is an expert ID 0..3.
    route_ids = similarity_routes(batch_X, region_table)

    # Compute prediction MSE using only the expert assigned to each example.
    # Inside routed_regression_loss, masks select rows for each expert.
    loss = routed_regression_loss(
        batch_X, batch_y, expert_W, expert_b, route_ids
    )

    # Build gradients for expert_W/expert_b from this loss.
    # Only experts used by route_ids receive meaningful gradients.
    loss.backward()

    # Apply manual SGD to expert_W and expert_b, then zero their gradients.
    # The next epoch starts with updated parameters and clean grad buffers.
    sgd([expert_W, expert_b], lr=0.03)


# This helper measures prediction error without training.
def evaluate_routed_loss(expert_W, expert_b, eval_X, eval_y):
    # routed_predict_regression computes similarity routes internally.
    # pred shape is [eval_examples]; routes shape is [eval_examples].
    pred, routes = routed_predict_regression(
        eval_X, expert_W, expert_b, region_table
    )

    # squared_loss returns average prediction error across this eval set.
    # .item() converts the scalar tensor into a plain Python float for printing.
    return squared_loss(pred, eval_y).item()


# This helper runs the full two-stage training process once.
# use_replay=False means stage 2 trains only on new data.
# use_replay=True means stage 2 trains on new data plus sampled old examples.
def run_curriculum(use_replay):
    # Create trainable expert weights.
    # Shape [4, 6] means four experts, each with one six-feature linear rule.
    expert_W = torch.randn(num_regions, num_features, requires_grad=True)

    # Create trainable expert biases.
    # Shape [4] means one scalar bias per expert.
    expert_b = torch.zeros(num_regions, requires_grad=True)

    # Construct an empty ReplayBuffer object.
    # Mechanically, ReplayBuffer owns two Python lists: buffer.X and buffer.y.
    # max_size=300 means after adding examples, it keeps only the most recent 300.
    # At this line, nothing is stored yet; len(buffer.X) is 0.
    buffer = ReplayBuffer(max_size=300)

    # Stage 1 trains on old_X_train only.
    # This should make experts good at the old distribution, especially regions 0 and 1.
    for epoch in range(100):
        train_one_routed_epoch(expert_W, expert_b, old_X_train, old_y_train)

    # If replay is enabled, copy old examples into the buffer after stage 1.
    # buffer.add loops through old_X_train row-by-row and stores detached clones.
    # Because max_size is 300 and old_X_train has 500 rows, only 300 examples remain stored.
    # Detaching matters because replay examples should be data, not old computation graphs.
    if use_replay:
        buffer.add(old_X_train, old_y_train)

    # Stage 2 trains on the new distribution.
    # This is where forgetting can happen: updates now mostly come from regions 2 and 3.
    for epoch in range(100):
        # Replay branch: mix old examples back into the stage-2 training batch.
        if use_replay:
            # Randomly choose 128 stored old examples from buffer.X and buffer.y.
            # replay_X shape: [128, 6]
            # replay_y shape: [128]
            replay_X, replay_y = buffer.sample(128)

            # Concatenate current new examples and replayed old examples along rows.
            # new_X_train shape: [500, 6]
            # replay_X shape: [128, 6]
            # batch_X shape: [628, 6]
            batch_X = torch.cat([new_X_train, replay_X], dim=0)

            # Concatenate matching labels in the same row order.
            # new_y_train shape: [500]
            # replay_y shape: [128]
            # batch_y shape: [628]
            batch_y = torch.cat([new_y_train, replay_y], dim=0)

        # No-replay branch: stage 2 ignores old data completely.
        else:
            # batch_X is just the new distribution examples, shape [500, 6].
            batch_X = new_X_train

            # batch_y is just the new distribution labels, shape [500].
            batch_y = new_y_train

        # Train one epoch on whichever batch the branch produced.
        # With replay, gradients reflect both new and sampled old examples.
        # Without replay, gradients reflect only new examples.
        train_one_routed_epoch(expert_W, expert_b, batch_X, batch_y)

    # Evaluate final model on old test data.
    # High loss here means the model forgot or never learned the old distribution well.
    old_test_loss = evaluate_routed_loss(expert_W, expert_b, old_X_test, old_y_test)

    # Evaluate final model on new test data.
    # This checks whether adding replay prevented learning the new distribution.
    new_test_loss = evaluate_routed_loss(expert_W, expert_b, new_X_test, new_y_test)

    # Return both losses so the outer loop can compare replay vs no replay.
    return old_test_loss, new_test_loss


# Create a Python list that will hold two rows: one no-replay row and one replay row.
results = []

# Run the same curriculum twice.
# First with use_replay=False, then with use_replay=True.
for use_replay in [False, True]:
    # Train a fresh model under this replay condition and get final test losses.
    old_test_loss, new_test_loss = run_curriculum(use_replay)

    # Store the result as a table row.
    # Column 1: whether replay was used.
    # Column 2: final loss on old-distribution test data.
    # Column 3: final loss on new-distribution test data.
    results.append([
        use_replay,
        old_test_loss,
        new_test_loss,
    ])

# Print the two result rows.
# Lower old_test_loss with replay means replay helped preserve old-distribution performance.
for row in results:
    print(row)
```

Output columns:

```text
use_replay | old_distribution_test_loss | new_distribution_test_loss
```

Expected rough pattern:

- Without replay, old-distribution test loss may worsen after training on the new distribution.
- With replay, old-distribution test loss may improve or degrade less.
- New-distribution performance can sometimes be worse with replay because training time is shared with old examples.

Failure checks:

- `NameError: ReplayBuffer`: rerun import cell after editing `train.py`.
- `RuntimeError` from `torch.cat`: check both tensors have the same number of features.
- Replay has no effect at all: check `use_replay` branch actually builds `batch_X` with `torch.cat`.

### 19.7 Checkpoint

You are ready to move on when you can explain:

- why replay uses `torch.cat(..., dim=0)`
- why replay can help old-region performance
- why replay can sometimes hurt new-region performance
- why replay and sparse routing solve different problems

## 20. Required Experiment Matrix

This matrix lists the minimum experiments for Project 0. Earlier phases already covered most regression rows.

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
2. routed softmax experts, oracle routing
3. routed softmax experts, random routing
4. routed softmax experts, similarity routing
```

Optional:

```text
1. homeostatic scaling
2. replay buffer
3. mixture shift followed by recovery
4. per-region active update counts
5. deliberate router corruption
```

Do not skip the global baseline.

Do not skip random routing.

Those weak baselines make the routed result interpretable.

## 21. Metrics

### 21.1 Regression Metrics

Use these for regression:

- train MSE
- test MSE
- per-region MSE
- weight norm
- router accuracy
- region usage count
- number of parameter updates

### 21.2 Classification Metrics

Use these for classification:

- train accuracy
- test accuracy
- train cross-entropy
- test cross-entropy
- per-region accuracy
- confusion matrix
- router accuracy
- region usage count

### 21.3 File Edit: Confusion Matrix In `metrics.py`

Action:

Add this function to `metrics.py`.

```python
def confusion_matrix(pred, y, num_classes):
    # Create a square count table.
    # Rows are true classes; columns are predicted classes.
    # Shape: [num_classes, num_classes]
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Walk through true/predicted class IDs one example at a time.
    for true, guessed in zip(y, pred):
        # Add one count to the cell for this true/predicted pair.
        matrix[true, guessed] += 1

    # Return the full count table.
    return matrix
```

Rows are true labels.

Columns are predicted labels.

Then rerun the notebook import/reload cell.

### 21.4 Syntax Preflight: Confusion Matrix

Action:

Run this in `experiments.ipynb`.

```python
y_demo = torch.tensor([0, 1, 2, 1, 0])
pred_demo = torch.tensor([0, 2, 2, 1, 1])

matrix = confusion_matrix(pred_demo, y_demo, num_classes=3)
print(matrix)
```

Expected interpretation:

```text
matrix[true_class, predicted_class]
```

For example, `matrix[1, 2]` counts examples whose true class was `1` and predicted class was `2`.

## 22. Notes Template

Use this in `notes.md` for every remaining experiment:

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

Minimum note quality:

- Include exact numbers.
- Include tensor shapes when a new tensor appears.
- Write at least one mechanical explanation, not only a conclusion.

## 23. Debugging Checklist

Run only the relevant debugging cell. Do not run all of these after every experiment.

### 23.1 Shape Debug Cell

Action:

Run this when a shape error appears.

```python
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

if "train_region_ids" in globals():
    print("train_region_ids", train_region_ids.shape)

if "test_region_ids" in globals():
    print("test_region_ids", test_region_ids.shape)

if "route_ids" in globals():
    print("route_ids", route_ids.shape)

if "expert_W" in globals():
    print("expert_W", expert_W.shape)

if "expert_b" in globals():
    print("expert_b", expert_b.shape)
```

### 23.2 Regression Debug Cell

Action:

Run this when regression predictions or loss look wrong.

```python
with torch.no_grad():
    y_hat, route_ids = routed_predict_regression(
        X_train, expert_W, expert_b, region_table
    )
    loss = squared_loss(y_hat, y_train)

print("y_hat", y_hat.shape)
print("loss", loss.item())
print("route usage", torch.bincount(route_ids, minlength=num_regions))
```

### 23.3 Classification Debug Cell

Action:

Run this when classification loss or accuracy looks wrong.

```python
with torch.no_grad():
    route_ids = similarity_routes(X_train, region_table)
    logits = routed_classification_logits(
        X_train, expert_W, expert_b, route_ids
    )
    loss = F.cross_entropy(logits, y_train)
    pred = logits.argmax(dim=1)

print("logits", logits.shape)
print("y_train", y_train.shape)
print("y min/max", y_train.min().item(), y_train.max().item())
print("loss", loss.item())
print("accuracy", (pred == y_train).float().mean().item())
print("route usage", torch.bincount(route_ids, minlength=num_regions))
```

### 23.4 Gradient Debug Cell

Action:

Run this only immediately after `loss.backward()` and before `sgd(...)`.

```python
print("expert_W.grad is None:", expert_W.grad is None)

if expert_W.grad is not None:
    print(expert_W.grad)
```

Important:

After `sgd(...)`, the manual optimizer clears gradients. If you inspect gradients after `sgd(...)`, you will usually see zeros.

## 24. Common Failure Modes

Shape mismatch:

- Check whether the tensor is `[batch, features]`, `[features]`, `[features, classes]`, `[regions, features]`, or `[regions, features, classes]`.
- For matrix multiplication, the right-hand tensor must start with the left-hand tensor's last dimension.

Stale imports:

- If a function was just added to `data.py`, `models.py`, `train.py`, or `metrics.py`, rerun the import/reload cell.
- Old notebook output saying `SKIP ... not typed yet` may be stale. Rerun the cell before trusting it.

Regression loss does not decrease:

- learning rate too high
- learning rate too low
- gradients not cleared
- labels generated incorrectly
- model too weak
- routing too noisy
- experts receiving too little data

Test loss much worse than train loss:

- overfitting
- distribution shift
- train/test split bug
- test regions underrepresented during training

Router accuracy bad:

- feature noise too high
- region prototypes too similar
- normalization missing
- using the wrong dimension in `topk`

Classification loss weird:

- applying softmax before `F.cross_entropy`
- labels are not integer class IDs
- labels contain values outside `[0, num_classes - 1]`
- logits shape is not `[batch, classes]`

Inactive experts changing:

- weight decay applied globally
- optimizer updating all parameters with stale gradients
- gradients not cleared
- loss accidentally used all experts

Replay errors:

- sampled before adding examples
- `torch.cat` used along the wrong dimension
- replay examples and current examples have different feature counts

## 25. Final Writeup Prompts

At the end, write a short technical note answering:

1. What did the global regression model learn?
2. What did the global classification model learn?
3. When did routed experts outperform the global model?
4. When did routed experts fail?
5. How much did router quality matter?
6. What did distribution shift do?
7. Which parameters received gradients during sparse training?
8. What did weight decay change?
9. What did per-region metrics reveal that average metrics hid?
10. Which part connects most clearly to D2L Chapters 2-4?
11. Which part feels like a real research question rather than a solved exercise?

## 26. Suggested Build Order

Use this exact order when typing Project 0 from scratch:

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
11. classification syntax preflight
12. classification data helpers
13. global softmax classifier
14. routed softmax experts
15. top-2 routing
16. local update gate
17. optional homeostatic scaling
18. optional replay buffer
19. final writeup
```

If a future phase requires a new syntax feature, add a drill before the full experiment cell.

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

Optional completion:

- top-2 routing is compared against top-1 routing
- homeostatic scaling is tested
- replay buffer is tested
- confusion matrix is computed for classification

The point is not that sparse experts must win.

The point is that you can explain why they win, lose, or behave differently.
