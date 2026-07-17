# D2L Notebook Generation Running Design Doc

Last updated: 2026-07-05

## Purpose

This document keeps durable project state for the D2L notebook rewrite. It should be updated after each completed chapter, major workflow change, or tooling roadblock.

The goal is to avoid relying on chat history for continuity.

## Current Structure

Root folder:

- `d2l/INSTRUCTIONS.md`: canonical generation instructions and syllabus
- `d2l/RUNNING_DESIGN_DOC.md`: this progress and workflow journal
- `d2l/Chapter 2 - Preliminaries/`: completed Chapter 2 notebooks
- `d2l/Chapter 3 - Linear Neural Networks for Regression/`: completed Chapter 3 notebooks
- `d2l/Chapter 4 - Linear Neural Networks for Classification/`: completed Chapter 4 notebooks
- `d2l/Chapter 5 - Multilayer Perceptrons/`: completed Chapter 5 notebooks
- `d2l/Chapter 6 - Builders' Guide/`: completed Chapter 6 notebooks
- `d2l/Chapter 7 - Convolutional Neural Networks/`: completed Chapter 7 notebooks
- `d2l/generation_scripts/`: persistent Python scripts used to generate or repair notebooks
- `d2l/generation_scripts/generate_chapter_03.py`: persistent generator for Chapter 3
- `d2l/generation_scripts/generate_chapter_04.py`: persistent generator for Chapter 4
- `d2l/generation_scripts/generate_chapter_05.py`: persistent generator for Chapter 5
- `d2l/generation_scripts/generate_chapter_06.py`: persistent generator for Chapter 6
- `d2l/generation_scripts/generate_chapter_07.py`: persistent generator for Chapter 7

## Completed Work

### Chapter 2 - Preliminaries

Completed as separate notebooks:

- `Chapter 2.1 - Data Manipulation.ipynb`
- `Chapter 2.2 - Data Preprocessing.ipynb`
- `Chapter 2.3 - Linear Algebra.ipynb`
- `Chapter 2.4 - Calculus.ipynb`
- `Chapter 2.5 - Automatic Differentiation.ipynb`
- `Chapter 2.6 - Probability and Statistics.ipynb`
- `Chapter 2.7 - Documentation.ipynb`

Validation performed:

- all notebooks parse as JSON
- all notebooks use `nbformat: 4`
- code cells have `execution_count: null`
- code cells have `outputs: []`
- code cells parse with Python `ast.parse`
- files are stored under `d2l/Chapter 2 - Preliminaries/`

Execution limitation:

- Full execution was not verified locally because the environment did not have `torch`, `pandas`, or `matplotlib` installed.
- `numpy` was available.


### Chapter 3 - Linear Neural Networks for Regression

Completed as separate notebooks:

- `Chapter 3.1 - Linear Regression.ipynb`
- `Chapter 3.2 - Object-Oriented Design for Implementation.ipynb`
- `Chapter 3.3 - Synthetic Regression Data.ipynb`
- `Chapter 3.4 - Linear Regression Implementation from Scratch.ipynb`
- `Chapter 3.5 - Concise Implementation of Linear Regression.ipynb`
- `Chapter 3.6 - Generalization.ipynb`
- `Chapter 3.7 - Weight Decay.ipynb`

Generation script kept:

- `d2l/generation_scripts/generate_chapter_03.py`

Validation performed:

- all notebooks parse as JSON
- all notebooks use `nbformat: 4`
- code cells have `execution_count: null`
- code cells have `outputs: []`
- code cells parse with Python `ast.parse`
- no code cell exceeds 10 nonblank lines
- files are stored under `d2l/Chapter 3 - Linear Neural Networks for Regression/`

Execution limitation:

- Full execution was not verified locally because the environment did not have `torch` installed.
- `numpy` was available.


### Chapter 4 - Linear Neural Networks for Classification

Completed as separate notebooks:

- `Chapter 4.1 - Softmax Regression.ipynb`
- `Chapter 4.2 - The Image Classification Dataset.ipynb`
- `Chapter 4.3 - The Base Classification Model.ipynb`
- `Chapter 4.4 - Softmax Regression Implementation from Scratch.ipynb`
- `Chapter 4.5 - Concise Implementation of Softmax Regression.ipynb`
- `Chapter 4.6 - Generalization in Classification.ipynb`
- `Chapter 4.7 - Environment and Distribution Shift.ipynb`

Generation script kept:

- `d2l/generation_scripts/generate_chapter_04.py`

Validation performed:

- all notebooks parse as JSON
- all notebooks use `nbformat: 4`
- code cells have `execution_count: null`
- code cells have `outputs: []`
- code cells parse with Python `ast.parse`
- no code cell exceeds 10 nonblank lines
- files are stored under `d2l/Chapter 4 - Linear Neural Networks for Classification/`

Execution limitation:

- Full execution was not verified locally because the environment did not have `torch` installed.
- `numpy` was available.

Implementation note:

- Chapter 4.2 uses tiny synthetic image tensors instead of downloading Fashion-MNIST-style data, so shape/loading/batching concepts remain inspectable without network access.


### Chapter 5 - Multilayer Perceptrons

Completed as separate notebooks:

- `Chapter 5.1 - Multilayer Perceptrons.ipynb`
- `Chapter 5.2 - Implementation of Multilayer Perceptrons.ipynb`
- `Chapter 5.3 - Forward Propagation, Backward Propagation, and Computational Graphs.ipynb`
- `Chapter 5.4 - Numerical Stability and Initialization.ipynb`
- `Chapter 5.5 - Generalization in Deep Learning.ipynb`
- `Chapter 5.6 - Dropout.ipynb`
- `Chapter 5.7 - Predicting House Prices on Kaggle.ipynb`

Generation script kept:

- `d2l/generation_scripts/generate_chapter_05.py`

Validation performed:

- all notebooks parse as JSON
- all notebooks use `nbformat: 4`
- code cells have `execution_count: null`
- code cells have `outputs: []`
- code cells parse with Python `ast.parse`
- no code cell exceeds 10 nonblank lines
- files are stored under `d2l/Chapter 5 - Multilayer Perceptrons/`

Execution limitation:

- Full execution was not verified locally because the environment did not have `torch` installed.
- `numpy` was available.

Implementation note:

- Chapter 5.7 uses tiny offline house-price rows and submission-like dictionaries instead of requiring Kaggle credentials, API tokens, downloads, or external CSV files.


### Chapter 6 - Builders' Guide

Completed as separate notebooks:

- `Chapter 6.1 - Layers and Modules.ipynb`
- `Chapter 6.2 - Parameter Management.ipynb`
- `Chapter 6.3 - Parameter Initialization.ipynb`
- `Chapter 6.4 - Lazy Initialization.ipynb`
- `Chapter 6.5 - Custom Layers.ipynb`
- `Chapter 6.6 - File I-O.ipynb`
- `Chapter 6.7 - GPUs.ipynb`

Generation script kept:

- `d2l/generation_scripts/generate_chapter_06.py`

Validation performed:

- all notebooks parse as JSON
- all notebooks use `nbformat: 4`
- code cells have `execution_count: null`
- code cells have `outputs: []`
- code cells parse with Python `ast.parse`
- no code cell exceeds 10 nonblank lines
- files are stored under `d2l/Chapter 6 - Builders' Guide/`

Execution limitation:

- Full execution was not verified locally because the environment did not have `torch` installed.
- `numpy` was available.

Implementation notes:

- File I/O examples use temporary directories and do not require persistent external artifacts.
- GPU examples use `torch.cuda.is_available()` and CPU fallback logic.

### Chapter 7 - Convolutional Neural Networks

Completed as separate notebooks:

- `Chapter 7.1 - From Fully Connected Layers to Convolutions.ipynb`
- `Chapter 7.2 - Convolutions for Images.ipynb`
- `Chapter 7.3 - Padding and Stride.ipynb`
- `Chapter 7.4 - Multiple Input and Multiple Output Channels.ipynb`
- `Chapter 7.5 - Pooling.ipynb`
- `Chapter 7.6 - Convolutional Neural Networks (LeNet).ipynb`

Generation script kept:

- `d2l/generation_scripts/generate_chapter_07.py`

Validation performed:

- all notebooks parse as JSON
- all notebooks use `nbformat: 4`
- code cells have `execution_count: null`
- code cells have `outputs: []`
- code cells parse with Python `ast.parse`
- no code cell exceeds 10 nonblank lines
- files are stored under `d2l/Chapter 7 - Convolutional Neural Networks/`

Execution limitation:

- Full execution was not verified locally because the environment did not have `torch` installed.
- `numpy` was available.

Implementation notes:

- CNN examples use tiny synthetic image tensors and manual kernels instead of `torchvision` datasets or downloads.
- Chapter 7 explicitly explains the deep-learning convention of calling cross-correlation layers convolutional layers.
- LeNet training uses one tiny synthetic batch to teach mechanics without network access or long execution.
## Roadblocks And Resolutions

### Windows command-length limit

Problem:

- A large inline PowerShell command that attempted to generate multiple notebooks failed before Python ran.
- The failure appeared as a Windows process creation / command-length issue.

Resolution:

- Use persistent `.py` generator scripts for future chapters instead of huge inline commands.
- Keep those scripts for future reference instead of deleting them.

Instruction update:

- Added notebook production workflow and roadblock retrieval rules to `INSTRUCTIONS.md`.

### Temporary generator script was deleted

Problem:

- A temporary Chapter 2 generator script was created and later removed.
- That made the completed generation less reproducible than it should be.

Resolution for future work:

- Keep chapter generator scripts under `d2l/generation_scripts/`.
- Name scripts by chapter, for example `generate_chapter_03.py`.
- Scripts should be rerunnable and should preserve notebook metadata/output rules.

### `apply_patch` sandbox wrapper failure

Problem:

- `apply_patch` failed on some file updates with a sandbox wrapper error before reading the target file.

Resolution:

- For markdown-only updates, use a small PowerShell or Python file edit and immediately verify the modified content.
- For source code edits, prefer `apply_patch`; if the same sandbox failure blocks it, document the failure and use the smallest safe targeted edit available.
- The same markdown `apply_patch` failure repeated during the Chapter 7 instruction update; a targeted PowerShell edit plus `Select-String` verification worked.


### Chapter 3 generator fallback details

Problem:

- `apply_patch` succeeded for the initial generator file but failed on a later append with the same sandbox wrapper issue.
- The fallback append initially copied patch-style leading `+` characters into the Python script.
- Two generated code cells exceeded the 10-line micro-example limit by one line.
- One generated notebook used `data_iter` before that helper was defined inside the same notebook.

Resolution:

- Removed stray leading `+` markers and verified with `python -m py_compile`.
- Added a self-contained `data_iter` helper to Chapter 3.4.
- Tightened long code cells and added maximum nonblank code-line validation.
- Updated `INSTRUCTIONS.md` with these validation requirements.


### Chapter 4 dataset-generation note

Problem:

- Image-classification dataset sections can tempt the generator to depend on `torchvision`, dataset downloads, or network access.

Resolution:

- Chapter 4.2 used tiny offline synthetic image tensors to teach image shape, labels, minibatches, flattening, and visualization concepts.
- `INSTRUCTIONS.md` was updated to prefer offline synthetic tensors for dataset-shape teaching unless the user explicitly asks for real dataset downloads.


### Chapter 5 Kaggle workflow note

Problem:

- Kaggle sections can tempt the generator to depend on credentials, API tokens, downloads, external CSV files, or live leaderboard behavior.

Resolution:

- Chapter 5.7 used offline Python rows and submission-like dictionaries to teach the workflow without network or credential requirements.
- `INSTRUCTIONS.md` was updated to prefer offline Kaggle-style examples unless the user explicitly approves real downloads or submissions.


### Chapter 6 path, File I/O, and GPU notes

Problem:

- The Chapter 6 folder name contains an apostrophe, which can break fragile shell quoting in validation commands.
- File I/O examples can accidentally create persistent artifacts if paths are not scoped carefully.
- GPU examples can fail on CPU-only machines if they assume CUDA availability.

Resolution:

- Validation used robust Python `Path` construction for `Chapter 6 - Builders' Guide`.
- Chapter 6.6 uses `tempfile.TemporaryDirectory()` for local, disposable file examples.
- Chapter 6.7 uses `torch.cuda.is_available()` and CPU fallback device selection.
- `INSTRUCTIONS.md` was updated with these durable rules.

### Chapter 7 CNN terminology and dataset notes

Problem:

- CNN chapters can tempt the generator to depend on `torchvision`, real image dataset downloads, or long training runs.
- CNN introductions can also blur the distinction between mathematical convolution and deep-learning cross-correlation.

Resolution:

- Chapter 7 used tiny synthetic tensors, manual kernels, and shape checks for convolution, padding, stride, channels, pooling, and LeNet.
- `INSTRUCTIONS.md` was updated to keep CNN chapters offline by default and to explain cross-correlation terminology explicitly.
### Missing local dependencies

Problem:

- `torch`, `pandas`, and `matplotlib` were unavailable in the local Python environment.
- This prevented full notebook execution validation.

Resolution:

- Continue to validate JSON, notebook metadata, no saved outputs, no execution counts, and code syntax.
- Report missing dependencies explicitly.
- Do not install dependencies unless the user asks or approves.

## Current Notebook Generation Pattern

Each subchapter notebook should contain:

- a top-level chapter title markdown cell
- required imports near the beginning
- one section per syllabus subheading
- the six-part required structure:
  - `1. Intuition`
  - `2. Why this exists`
  - `3. Examples`
  - `4. Step-by-step breakdown`
  - `5. Connection to ML systems`
  - `6. Common confusion points`
- code cells without saved outputs
- small examples that respect the learner profile

Layering rule:

- explain with raw intuition first
- show manual Python implementation when relevant
- show NumPy version when relevant
- show PyTorch/D2L framework mapping last

## Next Work Queue

Next chapter to generate:

- `Chapter 8 - Modern Convolutional Neural Networks`

Expected subchapter notebooks:

- `Chapter 8.1 - Deep Convolutional Neural Networks (AlexNet).ipynb`
- `Chapter 8.2 - Networks Using Blocks (VGG).ipynb`
- `Chapter 8.3 - Network in Network (NiN).ipynb`
- `Chapter 8.4 - Multi-Branch Networks (GoogLeNet).ipynb`
- `Chapter 8.5 - Batch Normalization.ipynb`
- `Chapter 8.6 - Residual Networks (ResNet) and ResNeXt.ipynb`
- `Chapter 8.7 - Densely Connected Networks (DenseNet).ipynb`
- `Chapter 8.8 - Designing Convolution Network Architectures.ipynb`

Recommended first step for Chapter 8:

- create `d2l/generation_scripts/generate_chapter_08.py`
- generate notebooks into `d2l/Chapter 8 - Modern Convolutional Neural Networks/`
- keep the generator script after successful generation
- validate JSON, no outputs, null execution counts, Python syntax, and code-cell line limits
- keep architecture examples tiny and shape-focused unless real training or datasets are explicitly approved
- update this document with validation results and any new roadblocks
## Open Decisions

- Whether to install notebook execution dependencies locally for stronger validation.
- Whether to retroactively recreate a reusable Chapter 2 generator script from the generated notebooks for reference.
- Whether future chapters should be generated one subchapter at a time or all subchapters per chapter after user approval.





