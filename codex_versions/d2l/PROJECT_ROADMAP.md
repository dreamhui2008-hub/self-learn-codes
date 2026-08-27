# D2L Project Roadmap

Last updated: 2026-08-24/

## Purpose

This document is the high-level topology for project work around the D2L study path.

The D2L notebooks teach concepts in small controlled pieces. The projects below are where those pieces get forced into one working system: data, tensors, model code, loss, gradients, optimization, evaluation, failure analysis, and iteration.

This is not a full walkthrough yet. Each project should later receive its own tutorial or notebook sequence.

## Learning Intensity Rule

Each project should be dense but bounded.

The project bar should stay production-facing. When a learner struggles, the response is more scaffolding, smaller mechanical steps, clearer shape checks, and more explicit walkthroughs, not a weaker target system. Projects may be scaled down in data size, model size, runtime, or deployment surface, but they should not be watered down into toy exercises once the relevant D2L concepts have been introduced.

Code quality, result quality, and presentation quality matter. Future projects should push toward production-level habits: modular code, reproducible configs, clean experiment logs, focused tests, meaningful baselines, interpretable metrics, readable documentation, and outputs that could be inspected by another engineer. The learning path can move slowly, but the destination should remain serious.

Every project must include:

- a tiny from-scratch drill only when a mechanism is new and worth exposing
- a production-shaped library/module version for the main experiment
- explicit tensor shapes
- explicit train/test split logic
- at least one baseline
- at least one deliberately broken or weak variant
- evaluation metrics
- a short experiment log
- notes on what changed in the model, data, or training loop
- a final "what I now understand" writeup

Every project should avoid:

- chasing leaderboard performance
- adding architecture complexity before the simple version is understood
- using library magic before the mechanism is explained
- keeping hand-written mechanics in large experiments after the mechanism is understood
- building product polish before the learning mechanism works

## Project Tutorial Design Standard

Starting with Project 0 Phase 15, and for all future projects, tutorials must be mechanically explicit. The student should not have to infer missing glue code, hidden execution order, or notebook state from qualitative directions.

Do not treat a tutorial draft as complete just because the concept is described. Before asking the student to type or run a section, check whether all required functions, imports, helpers, data variables, and evaluation blocks exist. If glue code is missing, the tutorial must identify it before the student reaches the experiment.

Each phase must use three layers:

```text
Concept
Small mechanical drill
Full experiment cell
```

The concept layer explains the idea in plain language. The drill layer uses tiny tensors or short snippets to build syntax and shape fluency. The full experiment layer gives a complete runnable block or a clear call to a reusable helper.

Each phase must include a syntax preflight before any large block. If a phase uses mechanics such as `torch.bincount`, `torch.argmax`, `F.cross_entropy`, boolean masks, `torch.no_grad`, optimizer objects, reshaping, indexing, or route IDs, those mechanics should appear first in 5-10 line drills. New syntax should not first appear inside a large experiment cell.

Use PyTorch library code where appropriate. Manual implementations are useful when the lesson is specifically about internals, but production-shaped code should prefer standard PyTorch APIs such as `torch.nn.functional.mse_loss`, `torch.nn.functional.cross_entropy`, and `torch.optim.SGD` once the concept has been taught.

For future projects after Project 0, do not keep using hand-written losses, manual optimizers, or notebook-local helper code after the mechanism has been introduced. The teaching flow should be:

```text
manual miniature version -> explain the mechanism -> production-shaped library/module version
```

For example, hand-written MSE and SGD are acceptable in an early mechanics drill. Later experiment cells should usually use `torch.nn.functional.mse_loss`, `torch.nn.functional.cross_entropy`, `torch.optim.SGD`, `torch.utils.data.DataLoader`, and project-local helper modules unless the phase is explicitly about implementing the internals.

Keep `experiments.ipynb` small and readable. Repeated logic belongs in project modules:

```text
data.py
router.py
models.py
train.py
metrics.py
```

The notebook should mostly run setup cells, tiny drills, experiment calls, and notes. It should not become a long-term storage place for duplicated helper functions.

Large experiment cells must include a mechanics memo. For every nontrivial line or small block, place a short comment immediately above it explaining what state is created or changed, what tensor shape is expected, what mutation happens, or why a branch exists. Avoid comments that merely rename the line. Prefer comments like:

```python
# Mechanically, ReplayBuffer owns two Python lists: buffer.X and buffer.y.
# max_size=300 means after adding examples, it keeps only the most recent 300.
buffer = ReplayBuffer(max_size=300)
```

Do not write vague comments like:

```python
# Create a replay buffer.
buffer = ReplayBuffer(max_size=300)
```

Every phase must provide exact run order:

```text
Step 1: edit this file
Step 2: rerun the import cell
Step 3: run this drill cell
Step 4: run this experiment cell
Step 5: write this checkpoint answer
```

Avoid vague phrases such as "run the same loop", "record these metrics", "insert where appropriate", or "try this variant" unless the exact insertion point and full executable context are shown.

Every phase must include expected outputs and failure checks:

- expected shapes
- expected rough metric pattern
- common stale-import error
- common shape error
- common route ID error
- what result would be suspicious

Scope should slow down when mechanics become new. If a project phase introduces several unfamiliar APIs or tensor patterns at once, add a bridge/tutorial section before the full project experiment. The expectation is that the student enters the main experiment with high-level understanding and enough mechanical fluency to read the code, not that they learn every new syntax item inside one large block.

## Reference-Guided Project Rule

Starting with Project 1, each major project should be guided by at least one reference paper, technical report, publication, or high-quality engineering writeup.

The goal is not full paper reproduction. The goal is bounded imitation: understand the core idea, map it onto the relevant D2L chapters, implement a smaller but serious version, measure behavior, and document where the project intentionally differs from the original reference.

Each project plan should include:

- one primary reference
- optional supporting references only when they answer a project question
- a short "reference-to-implementation map"
- a list of claims or mechanisms being imitated
- a list of claims intentionally not reproduced
- at least one baseline that predates or simplifies the reference idea
- at least one ablation or weak variant
- final notes comparing observed project behavior against the reference

References should be selected close to the project start. Do not overload the roadmap with papers too early. The active rule is: choose enough reference material to make the project serious, but not so much that reading replaces implementation.

## Roadmap Shape

This roadmap is not saying D2L ends at Chapter 7. D2L continues through Chapter 21. The projects are checkpoints inserted between D2L blocks so the material becomes mechanical rather than only readable.

```text
D2L Chapters 2-4
-> Project 0: Sparse Linear Experts Under Distribution Shift
-> D2L Chapters 5-6 as the deep-learning software bridge
-> D2L Chapters 7-8
-> Project 1: Paper-Guided CNN System
   provisional direction: advanced AlexNet-style classifier
-> D2L Chapters 9-10
-> Project 2: Paper-Guided RNN Sequence System
-> D2L Chapters 11-13
-> Project 3: Transformer / Small LLM-Style Training System
-> Optional Project 3B: RAG/API/Agent System
   integrated only after the transformer baseline is stable, or kept as a sibling project
-> D2L Chapter 14
-> Project 4: Paper-Guided Computer Vision System
-> D2L Chapters 15-16
-> Project 5: NLP Representation, Search, and Application System
-> Production ML Systems Capstone
   convert one completed ML project into a deployed service with API, persistence, CI, observability, and evaluation
-> D2L Chapters 17-21 parked until the chapters create a project reason
-> Later: full AI product / startup-style system / research-track experiments
```

## Project 0: Sparse Linear Experts Under Distribution Shift

Timing: after D2L Chapters 2-4, before the D2L Chapters 5-8 continuation.

Role: integration project for tensors, preprocessing, linear algebra, calculus, autodiff, linear regression, softmax regression, generalization, weight decay, and distribution shift.

Core idea:

Build a small simulator where a router chooses among several simple linear experts. Each expert is only a Chapter 3 or Chapter 4 model. The system should feel like a small-scale version of the PFC/router idea, without pretending to be real hardware or real MoE training.

Main learning targets:

- generate synthetic datasets with known ground truth
- build regression and classification baselines
- implement manual SGD
- compare manual training with PyTorch optimizer training
- track train loss, test loss, accuracy, and generalization gap
- introduce distribution shift deliberately
- route examples to top-k experts using vector similarity
- update only participating experts
- compare global learning against sparse local learning

Minimum system components:

- `data.py`: synthetic data generation and train/test splitting
- `models.py`: linear regression, softmax regression, and expert wrappers
- `router.py`: query vector scoring, top-k region selection, routing logs
- `train.py`: manual SGD and concise PyTorch training loops
- `metrics.py`: MSE, accuracy, confusion matrix, loss curves
- `experiments.ipynb`: guided runs and analysis
- `notes.md`: experiment log and explanations in your own words

Experiments to run:

- one global linear model vs multiple routed experts
- random routing vs similarity routing
- top-1 routing vs top-2 routing
- update all experts vs update selected experts only
- no weight decay vs weight decay
- stable train/test distribution vs shifted test distribution
- low-noise data vs high-noise data
- per-region loss tracking vs one averaged global loss

Light connection to the hardware topology:

For Project 0, the hardware topology is only vocabulary and inspiration. The goal is not to validate the single-die design. The goal is to build mechanical comfort with routing, sparsity, active subsets, and measurement.

- PFC router becomes a small CPU-side routing function
- region table becomes a matrix of region embeddings
- active regions become selected experts
- dormant regions are experts skipped for that batch
- priority frequency cache becomes a counter of recent region usage
- active parameter cap becomes a simple top-k or update budget

Connection to the biological/local-learning topology:

The algorithm topology should be treated as the more relevant intermediate-to-long-term research bogey. Unlike the hardware topology, it should not be forced into a single fixed sequence. Several parts are asynchronous, local, or only weakly ordered relative to the others.

- Stage 0 sparse encoding becomes top-k routing
- Stage 1 eligibility trace becomes "this expert participated"
- Stage 2 broadcast becomes a scalar loss/confidence signal
- Stage 3 gated update becomes "update only if participated and loss passes threshold"
- Stage 4 homeostatic scaling can become optional periodic weight normalization
- Stage 5 replay can become optional replay from an old-example buffer
- Stage 6 reconsolidation is intentionally deferred

Important modeling constraint:

Do not overfit the software simulator to the current hardware sequence. A useful Project 0 result is allowed to point away from the hardware design. If the algorithmic experiments suggest async local updates, replay, scaling, or cold retrieval need a different execution model, that should be treated as discovery rather than inconsistency.

Explicit non-goals:

- no MLPs
- no CNNs
- no transformers
- no real MoE load-balancing loss
- no LoRA
- no gem5
- no CXL/NUMA simulation yet
- no production API

Done criteria:

- can train a global baseline
- can train routed experts
- can explain every tensor shape in the forward pass
- can explain which parameters receive gradients and which do not
- can show a distribution shift case where the model degrades
- can show one ablation where sparse routing helps, hurts, or behaves differently
- can write a short technical note connecting the result back to D2L Chapters 2-4

## D2L Chapters 5-8 Bridge

After Project 0, continue with:

- Chapter 5: multilayer perceptrons, nonlinear activations, initialization, dropout, generalization
- Chapter 6: modules, parameters, custom layers, initialization, file I/O, devices
- Chapter 7: convolution, padding, stride, channels, pooling, LeNet
- Chapter 8: modern CNN architectures

The point of this stretch is to upgrade from linear models to reusable deep learning systems and then into serious CNN architecture work.

The mental shift:

- Chapter 3-4: one linear map can learn simple decision boundaries
- Chapter 5: stacked nonlinear layers can learn richer functions
- Chapter 6: models become reusable software objects
- Chapter 7: spatial structure matters, especially for images
- Chapter 8: architecture choices become design decisions that can be tied back to published systems

Important scope note:

Do not start the main CNN project immediately after Chapter 7. Chapter 7 gives the basic mechanics of convolution, padding, stride, channels, pooling, and LeNet. Chapter 8 makes the project more meaningful by adding modern CNN architecture patterns. The provisional Project 1 direction is an advanced AlexNet-style classifier, but the exact reference packet and implementation target should be finalized only after Chapter 8 is complete.

## Full D2L Continuation Map

Use the projects as reinforcement checkpoints while still continuing through D2L.

Foundation block:

- Chapters 2-4: tensors, data, linear algebra, calculus, autodiff, regression, classification, generalization, distribution shift
- Project 0 belongs here

Deep learning software bridge:

- Chapters 5-6: MLPs, modules, parameters, initialization, dropout, custom layers, file I/O, devices
- No major standalone project is required here unless mechanics are weak
- Notes and drills should still be serious: shape contracts, reusable modules, clean training loops, and checkpoint basics

CNN architecture block:

- Chapters 7-8: convolution mechanics, LeNet, and modern CNN architectures
- Project 1 belongs here as a paper-guided CNN system

Sequence block:

- Chapters 9-10: recurrent neural networks and modern recurrent neural networks
- Project 2 belongs here as a paper-guided RNN sequence system

Attention, transformer, and training-systems block:

- Chapter 11: attention mechanisms and transformers
- Chapter 12: optimization algorithms
- Chapter 13: computational performance
- Project 3 belongs here as a transformer or small LLM-style training system
- Optional Project 3B can add RAG/API/agent work after the core transformer baseline is stable

Computer vision application block:

- Chapter 14: computer vision
- Project 4 belongs here as a paper-guided practical vision system

NLP application block:

- Chapters 15-16: NLP pretraining and NLP applications
- Project 5 belongs here, relocating the earlier NLP embeddings/search idea into the proper NLP block

Advanced/specialized block:

- Chapter 17: reinforcement learning
- Chapter 18: Gaussian processes
- Chapter 19: hyperparameter optimization
- Chapter 20: generative adversarial networks
- Chapter 21: recommender systems
- These chapters are parked until the material itself creates a reason for another project

## Project 1: Paper-Guided CNN System

Timing: after D2L Chapters 7-8.

Role: first serious vision training project and first reference-guided architecture project.

Core idea:

Build a production-shaped image classification project guided by a CNN reference paper or publication. The provisional direction is an advanced AlexNet-style classifier. The likely primary reference is the AlexNet paper, but the exact project target should be revisited after Chapter 8 so the selected architecture matches what has actually been learned.

Main learning targets:

- load a real image dataset
- inspect image tensors and label tensors
- build an MLP baseline
- build a LeNet-style baseline
- build a reference-inspired CNN model
- write a reusable training and evaluation loop
- track train/validation/test metrics
- diagnose overfitting
- use initialization, normalization, dropout, weight decay, and augmentation intentionally where appropriate
- save and reload model weights
- compare the implemented model against the reference architecture at the level of mechanisms, not just names
- document what was scaled down, omitted, or changed

Suggested dataset options:

- Fashion-MNIST for easiest continuity with D2L
- CIFAR-10 for a more meaningful color-image version
- Tiny ImageNet or a curated small image dataset only if runtime and data handling are under control

Reference-guided scope:

- imitate architecture motifs such as stacked convolution blocks, larger early receptive fields where justified, nonlinearities, pooling, dropout, and classifier heads
- keep input size, dataset, and training budget scaled to local hardware
- compare against MLP and LeNet-style baselines
- include at least one ablation, such as removing dropout, changing kernel sizes, weakening augmentation, or reducing depth
- use standard PyTorch optimizers unless a custom optimizer becomes a deliberate add-on later

Required implementation skeleton:

Project 1 should be written as a small production-shaped Python project, not as one large notebook. This structure is the source of truth for the future Project 1 tutorial unless deliberately revised:

```text
project_1_cnn_reference/
  pyproject.toml
  src/cnn_reference/
    data.py
    models.py
    train.py
    evaluate.py
    checkpoint.py
    cli.py
    reference_notes.py
  tests/
    test_data.py
    test_shapes.py
    test_checkpoint.py
    test_train_step.py
  notebooks/
    experiments.ipynb
  configs/
    default.toml
```

File responsibilities:

- `data.py`: dataset loading, train/validation/test split logic, transforms, and DataLoader construction
- `models.py`: MLP baseline, LeNet-style baseline, and reference-inspired CNN definitions with explicit shape notes
- `train.py`: reusable training loop, epoch loop, metric collection, and device placement
- `evaluate.py`: validation/test loops, accuracy/loss metrics, and per-class or confusion-matrix helpers
- `checkpoint.py`: save and reload model weights, optimizer state, config, and metric history
- `cli.py`: command-line entry points for training, evaluation, and checkpoint inspection
- `reference_notes.py`: structured notes mapping reference mechanisms to implementation choices
- `tests/`: focused tests for data shapes, model forward shapes, checkpoint round trips, and one train-step update
- `notebooks/experiments.ipynb`: small experiment notebook that calls project modules instead of storing repeated helper code
- `configs/default.toml`: reproducible default settings for the selected dataset and model

Done criteria:

- MLP baseline works
- LeNet-style baseline works
- reference-inspired CNN beats, meaningfully differs from, or fails against the baseline for explainable reasons
- training curves are saved
- overfitting is demonstrated and reduced
- at least one architecture or regularization ablation is run
- final notes explain convolution, pooling, channels, architecture motifs, and reference deviations in plain language

## Project 2: Paper-Guided RNN Sequence System

Timing: after D2L Chapters 9-10.

Role: sequence-modeling checkpoint before attention and transformers.

Core idea:

Build a production-shaped RNN sequence project guided by a recurrent-model reference. The exact task can be language modeling, sequence classification, time-series forecasting, or another sequence problem, but it should force the mechanics of hidden state, recurrence, truncation, batching, and evaluation to become concrete.

Main learning targets:

- prepare sequential data with explicit time and batch dimensions
- build simple RNN, GRU, or LSTM baselines depending on the selected reference
- handle hidden-state initialization, detachment, and sequence boundaries
- compare teacher-forced training against autoregressive generation or rollout where relevant
- measure loss, accuracy, perplexity, forecasting error, or task-specific sequence metrics
- inspect generated or predicted sequences manually
- run at least one ablation involving sequence length, hidden size, recurrence type, or state handling
- write a reference-to-implementation map

Suggested dataset options:

- character-level or word-level language modeling corpus
- sentiment or topic sequences if classification is preferred
- small time-series dataset if forecasting better exposes recurrence
- curated personal text only if privacy and cleaning rules are explicit

Done criteria:

- sequential data pipeline is reproducible
- at least one non-recurrent baseline is included where possible
- RNN/GRU/LSTM model trains and evaluates cleanly
- hidden-state mechanics are explained with tensor shapes
- at least one sequence-specific failure mode is analyzed
- final notes explain recurrence and gating in plain language

## Project 3: Transformer / Small LLM-Style Training System

Timing: after D2L Chapters 11-13.

Role: modern attention/transformer project, pushed as close to production-quality training practice as is realistic at small scale.

Core idea:

Build a small transformer training system guided by attention and transformer references. The default direction is a small decoder-only language model or another transformer variant that makes masking, attention, batching, optimization, checkpoints, evaluation, and performance visible. The scale can be small; the engineering should not be sloppy.

Main learning targets:

- implement or assemble tokenization and dataset preparation with clear sequence/block shapes
- build attention masks and explain their shapes
- implement or use transformer blocks with embeddings, positional information, multi-head attention, MLP blocks, residual paths, normalization, and output heads
- train with a reproducible config
- use Chapter 12 optimization ideas intentionally
- use Chapter 13 performance ideas intentionally: device placement, batching, profiling, throughput, memory, and checkpointing
- evaluate with loss/perplexity and at least one task-appropriate qualitative or behavioral check
- save checkpoints, resume training, and log runs
- keep tests for model shapes, masks, one training step, and checkpoint round trips
- document reference deviations and scaling choices

Candidate directions:

- decoder-only small language model
- encoder-based classifier or representation learner
- sequence-to-sequence transformer if the selected reference and dataset justify the extra complexity

Done criteria:

- tokenizer or text preprocessing path is reproducible
- transformer forward pass has explicit shape checks
- training can resume from checkpoint
- at least one baseline or simplified transformer variant is included
- at least one optimization or performance comparison is logged
- final notes explain attention, masking, residual paths, optimizer choices, and scaling limits

## Optional Project 3B: RAG/API/Agent System

Timing: after Project 3 has a stable baseline, or as a sibling project if the goal shifts toward AI systems.

Role: transition from model learner to AI systems builder without bloating the core transformer project.

Core idea:

Build a retrieval-augmented question-answering system, wrap it in an API, add evaluation and logging, then optionally extend it into a small agent with tool use and memory/state. This may use the Project 3 model only if that is technically sensible. It is also acceptable to use an external embedding or generation API if the goal is systems behavior rather than model pretraining.

Main learning targets:

- ingest documents
- chunk text
- create or call embeddings
- store vectors
- retrieve relevant chunks
- generate answers with citations or evidence
- evaluate retrieval and answer quality
- serve the system with FastAPI or an equivalent API layer
- track latency and failures
- add basic logging and monitoring
- build a small agent loop with tools and state only after RAG works

Minimum system components:

- ingestion pipeline
- vector index
- retriever
- answer generator
- evaluation set
- API service
- logs for query, retrieved chunks, answer, latency, and errors
- simple UI or CLI client

Done criteria:

- RAG pipeline answers questions against a known corpus
- retrieval can be evaluated separately from generation
- API can serve requests
- logs expose latency and failure cases
- agent loop, if included, can use at least one real tool with clear stop conditions
- final notes distinguish model behavior, retrieval behavior, and system behavior

## Project 4: Paper-Guided Computer Vision System

Timing: after D2L Chapter 14.

Role: practical computer vision project after the CNN architecture project.

Core idea:

Build a computer vision system guided by a vision reference that is closer to applied CV than generic classification. The exact task should be chosen after Chapter 14. Possible directions include transfer learning, detection, segmentation, fine-grained classification, augmentation-heavy training, or a small deployment-oriented vision pipeline.

Main learning targets:

- use a real vision dataset with clear train/validation/test handling
- apply augmentation and preprocessing deliberately
- use pretrained weights or transfer learning when appropriate
- evaluate with task-specific metrics, not only top-1 accuracy when the task demands more
- inspect errors visually
- compare against a simpler baseline
- document reference mechanisms and practical deviations

Done criteria:

- data pipeline handles images and labels cleanly
- model or pipeline trains/evaluates reproducibly
- visual failure analysis is included
- at least one task-specific metric is reported
- at least one ablation or preprocessing comparison is run
- final notes connect Chapter 14 concepts to the selected reference

## Project 5: NLP Representation, Search, and Application System

Timing: after D2L Chapters 15-16.

Role: main NLP project, relocating the earlier NLP embeddings/search idea into the correct NLP block.

Core idea:

Build a small but serious text system that turns text into useful representations, uses those representations for classification, similarity search, retrieval, or another NLP application, and evaluates whether the predictions or retrieved neighbors make sense. This project should be reference-guided, with the reference chosen after Chapters 15-16 clarify the available NLP mechanisms.

Main learning targets:

- tokenize text
- build simple lexical baselines such as bag-of-words or TF-IDF when useful
- train or fine-tune a text model
- compute or use embeddings
- build a small similarity or retrieval index
- compare keyword search against vector similarity where relevant
- evaluate retrieval, classification, or task quality
- inspect failure cases manually
- connect NLP pretraining ideas to downstream behavior

Done criteria:

- text classifier baseline works
- similarity search returns inspectable neighbors
- retrieval quality is measured with simple metrics
- at least one failure mode is analyzed
- final notes explain embeddings and representations as geometry, training signal, and task behavior rather than magic

## Production ML Systems Capstone

Timing: after Project 5, or earlier after optional Project 3B if the career goal shifts strongly toward ML software engineering roles.

Role: job-readiness bridge from "I can train and evaluate models" to "I can own an ML service end to end." This is the capstone that directly strengthens the production gaps in ML software engineer postings such as API design, persistence, deployment, CI/CD, logging, observability, evaluation, and operating a model-backed system.

Core idea:

Take one completed ML project, preferably Project 3B or Project 5, and turn it into a small deployed ML product/service. The model does not need to be large. The engineering surface should be real: a user or client sends a request, the service retrieves or computes model features, produces a response, records useful logs/metrics, and can be tested, rebuilt, and redeployed reproducibly.

Recommended project directions:

- RAG or semantic search service over a known document corpus
- text classification or similarity API with stored examples and evaluation cases
- vision inference API with uploaded images, saved predictions, and error review
- experiment/evaluation dashboard for a previously trained model

Minimum system components:

- `api/`: FastAPI or equivalent service with request/response schemas
- `model/`: loading path for trained weights, tokenizer/vectorizer, preprocessing, and inference
- `storage/`: relational database, document store, vector index, or object storage depending on the project
- `evals/`: fixed evaluation set, regression tests, quality metrics, and failure cases
- `tests/`: unit tests plus at least one API/integration test
- `configs/`: environment-aware config for local and deployed runs
- `docker/` or `Dockerfile`: reproducible runtime container
- `ci/`: GitHub Actions or equivalent checks for tests, formatting, and basic build validation
- `logs/`: structured logs for request ID, latency, model/version, retrieved context or features, prediction, and errors
- `docs/`: design doc, runbook, model card or service card, and known limitations

Main learning targets:

- separate offline training code from online inference code
- design stable input/output schemas
- persist data, predictions, embeddings, or evaluation records
- write tests for preprocessing, inference, API behavior, and saved artifacts
- containerize the service and run it from a clean environment
- add CI checks that would catch broken imports, broken tests, and broken API contracts
- log latency, failures, model version, and important input/output metadata
- evaluate model quality separately from service reliability
- write operational notes: how to run, how to deploy, how to debug, how to roll back

Done criteria:

- service runs locally from a clean command
- API has at least one real inference endpoint and one health/status endpoint
- model or retrieval artifacts load without notebook state
- persistence layer stores the key application records
- tests cover preprocessing, inference, API behavior, and one failure case
- CI passes from a fresh checkout
- Docker image builds and runs
- deployment target is documented, and deployed if practical
- logs expose latency and failure cases
- fixed eval set reports quality metrics before and after at least one code or model change
- final writeup maps the project to ML software engineer responsibilities: data path, model path, API path, storage path, tests, deployment, observability, limitations, and future work

## Later Project Track

After Projects 0-5, the Production ML Systems Capstone, and optional Project 3B if it is selected, choose based on goal.

Career/product track:

- deepen the capstone into a full AI product with RAG, agent, API, UI, evals, and deployment
- startup-style personalized feed reader or research assistant
- evaluation dashboard for retrieval quality, hallucination rate, latency, and regression tests

Research track:

- extend Project 0 into a stronger sparse expert simulator
- test local update rules against ordinary backprop baselines
- add prioritized replay
- simulate async local mechanisms such as homeostatic scaling, replay, and retrieval-triggered writable subsets
- simulate hibernation, activation budgets, and region frequency caches only when they clarify an algorithmic question
- compare catastrophic forgetting across global, sparse, replay, and gated-update systems

Hardware track:

- keep this deferred until the software and algorithmic simulators have real results
- treat the current hardware document as a north star, not a binding execution model
- revisit the hardware design after meaningful mechanical comfort with models, routing, updates, and evaluation
- start with fake latency and active-region accounting if it helps interpret software results
- only then consider NUMA, gem5, C++, or memory-subsystem experiments

## Side Reading Map

Side readings should support the current project or D2L block. They should not replace the remaining D2L chapters.

Immediate:

- D2L Chapters 2-4 notes
- your own Project 0 experiment logs

After Project 0:

- D2L Chapters 5-8

During Project 1:

- D2L Chapter 7 convolution mechanics
- D2L Chapter 8 modern CNN architectures
- primary CNN reference, likely AlexNet if that remains the selected direction
- PyTorch image-data, model, checkpoint, and evaluation documentation as needed

During Project 2:

- D2L Chapters 9-10 for recurrent sequence models
- one RNN/GRU/LSTM reference selected after Chapter 10
- Neuromatch computational neuroscience tutorials, selectively

During Project 3:

- D2L Chapter 11 for attention and transformers
- D2L Chapter 12 for optimization algorithms
- D2L Chapter 13 for computational performance
- primary transformer or small-LLM reference selected after Chapter 11
- selected LLM-from-scratch or training-systems material only when implementation creates demand

During optional Project 3B:

- Chip Huyen, AI Engineering
- production RAG material only after a minimal RAG baseline works

During Project 4:

- D2L Chapter 14 for practical computer vision
- one computer vision reference selected after Chapter 14
- OpenCV, detection, segmentation, transfer-learning, or deployment resources only if they support the chosen task

During Project 5:

- D2L Chapters 15-16 for NLP pretraining and NLP applications
- one NLP representation, pretraining, retrieval, or application reference selected after Chapter 16
- Chip Huyen, Designing Machine Learning Systems if the NLP system needs production evaluation framing

During Production ML Systems Capstone:

- FastAPI, Docker, GitHub Actions, database/vector-store, and observability documentation as needed
- Chip Huyen, Designing Machine Learning Systems for production evaluation, data/model monitoring, and failure analysis
- deployment-platform documentation only after the local service, tests, and Docker path work

Later:

- D2L Chapters 17-21 depending on interest and project need
- Hands-On Machine Learning with Scikit-Learn and PyTorch for breadth and repetition
- Elements of Deep Learning for math comfort
- selected papers only when a project creates a concrete reason to read them

## Current Next Step

Project 0 and D2L Chapters 5-8 are complete.

The next main step is Project 1. Do not pause D2L for a full DevOps or MLOps course. Use small production-adjacent slices only when they directly support the current ML project.

Active cadence:

```text
D2L concept block
-> one D2L project
-> one small production wrapper around that exact project
-> return to D2L
```

### Near-Term Step Guide

All resources in this section are free or have a free self-paced path as of 2026-08-24. Use local or notebook/cloud-free paths first. Do not use a paid cloud service unless a later project explicitly calls for it.

#### 1. Before Project 1: Gritty Preflight

Do this:

- get comfortable with shell paths, project folders, Git status/diff/commit, Python environments, Docker vocabulary, and MLflow vocabulary
- run beginner examples only
- stop before this turns into a separate DevOps course

Why:

- shell/Git/package basics make project folders, scripts, paths, and reproducibility less mysterious
- Docker gives the deployment vocabulary: image, container, build, run, port, mounted files
- MLflow gives the ML workflow vocabulary: run, parameter, metric, artifact, checkpoint, model reload

Resources and stopping points:

- [MIT Missing Semester](https://missing.csail.mit.edu/)
  - Read/run: [Course Overview + Introduction to the Shell](https://missing.csail.mit.edu/2026/course-shell/), [Command-line Environment](https://missing.csail.mit.edu/2026/command-line-environment/), [Version Control and Git](https://missing.csail.mit.edu/2026/version-control/), and [Packaging and Shipping Code](https://missing.csail.mit.edu/2026/shipping-code/).
  - Focus on: `pwd`, `cd`, `ls`, `mkdir`, `cp`, `mv`, `rm`, `cat`, `head`, `tail`, `rg`/`grep`, running `python train.py`, `$PATH`, virtual environments, `git status`, `git diff`, `git add`, `git commit`, `git log`, and what an artifact is.
  - Stop when: you can navigate a project folder, run a Python script from the terminal, inspect files, make a small Git commit, and explain why a virtual environment changes dependency resolution.
  - Skip for now: heavy shell customization, dotfiles, terminal multiplexers, remote machines, TestPyPI publishing, GitHub Pages, Redis, and advanced packaging.

- ✅ [Docker official workshop](https://docs.docker.com/get-started/workshop/)
  - Read/run: the beginner path through containerizing a small app, building an image, and running a container.
  - Focus on: image vs container, `Dockerfile`, build context, `docker build`, `docker run`, port mapping, logs, stopping/removing containers.
  - Stop when: you can say "this image packages the app and dependencies" and run one container from an image you built.
  - Skip for now: Docker Hub publishing, Compose databases, Kubernetes integration, security scanning, production hardening.

- [MLflow Tracking quickstart](https://mlflow.org/docs/latest/ml/getting-started/quickstart/) and [MLflow Tracking concepts](https://mlflow.org/docs/latest/tracking)
  - Read/run: quickstart Steps 1-6: set experiment, log params/metrics/model, inspect the run if the UI is available, and load the logged model for inference.
  - Focus on: experiment, run, parameter, metric, artifact, model, local `mlruns` folder.
  - Stop when: you understand that `python train.py` can create a run record with the model config, validation metrics, and saved files.
  - Skip for now: remote tracking servers, database backend stores, model registry workflows, team permissions.

- [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/)
  - Read/watch selectively: Lecture 2 "Development Infrastructure & Tooling" only as vocabulary before Project 1.
  - Focus on: what tools surround model training in real projects.
  - Stop when: you can name the basic system pieces around training: code, config, data, experiment tracking, artifact storage, testing, deployment.
  - Skip for now: labs, W&B-specific implementation, annotation systems, monitoring, continual learning, team/project-management lectures.

Do not do yet:

- Kubernetes
- Terraform
- Prometheus/Grafana
- full DataTalksClub MLOps Zoomcamp
- cloud billing or managed deployment platforms

#### 2. Project 1: Modern CNN System

Do this:

- build the Project 1 modern CNN tutorial
- keep AlexNet as historical context or a weak baseline, not the main target by default
- use a modern CNN design target that reflects Chapter 8: convolution blocks, ReLU, normalization, residual-style thinking, global/adaptive pooling, and a compact classification head
- train on a real image dataset, not synthetic logits
- write clean project modules, not one giant notebook
- log metrics and save checkpoints
- reload the best checkpoint for evaluation

Why:

- Chapter 8 makes CNN architecture choices design decisions, not memorized layer lists
- real data forces data import, transforms, batching, labels, train/validation/test splits, and artifact handling
- checkpoints and reloads turn the model into a reusable object instead of a temporary notebook state

Required production-shaped habits:

- `experiments.ipynb` stays readable
- reusable logic goes into project files
- configs control data/model/training choices
- every major tensor handoff has a shape contract
- every result has a baseline or ablation
- metrics are written somewhere inspectable
- the final model can be loaded without rerunning training

Resources and stopping points:

- [Project 1 tutorial](project_1_modern_cnn/TUTORIAL.md)
  - Read/run: the project tutorial in order.
  - Focus on: data import, image transforms, DataLoader batches, CNN blocks, training/eval loops, ablations, checkpoints, and reloadable inference.
  - Stop when: the tutorial's final checkpoint and notes are complete.

- [PyTorch Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
  - Read/run: load/normalize CIFAR-10, inspect classes, create DataLoaders, define a CNN, define loss, train, test.
  - Focus on: how real images become `[batch, channels, height, width]` tensors and how the training loop consumes DataLoader batches.
  - Stop when: CIFAR-10 data loading, `DataLoader`, logits, loss, optimizer step, and test accuracy feel mechanically connected.
  - Skip for now: adapting the tutorial to many architectures or chasing accuracy.

- [Torchvision transforms v2 getting started](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html)
  - Read/run: "The basics" and "I just want to do image classification."
  - Focus on: `RandomResizedCrop`, `RandomHorizontalFlip`, `ToDtype`, `Normalize`, and why train transforms differ from validation/test transforms.
  - Stop when: you can explain the exact tensor shape/range before and after transforms.
  - Skip for now: object detection, segmentation, bounding boxes, masks, videos, keypoints.

- [Torchvision model docs](https://docs.pytorch.org/vision/master/models.html) and [Torchvision ConvNeXt docs](https://docs.pytorch.org/vision/main/models/convnext.html)
  - Read: classification model setup, pretrained weights idea, and the ConvNeXt Tiny builder.
  - Focus on: model constructors, classifier head replacement, weight transforms, and the difference between architecture choice and training loop choice.
  - Stop when: you can instantiate a model, inspect its final classifier, and explain whether Project 1 is training from scratch or fine-tuning.
  - Skip for now: benchmarking the full model zoo, detection/segmentation models, quantization, export.

- [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/)
  - Read/watch selectively during the project: Lecture 2 "Development Infrastructure & Tooling", Lecture 3 "Troubleshooting & Testing", and Lecture 4 "Data Management."
  - Focus on: why serious ML work needs experiment records, failure analysis, data handling, and tests.
  - Stop when: you have copied the relevant ideas into Project 1's project habits: config, logs, artifacts, sanity checks, and final notes.
  - Skip for now: FSDL labs, deployment lecture, monitoring lecture, project showcase.

#### 3. Immediately After Project 1: First Serving Wrapper

Do this:

- create a small inference script that loads the saved CNN checkpoint
- preprocess one input image the same way training did
- return class scores and predicted label
- expose that inference path with FastAPI or BentoML
- containerize the inference service with Docker

Why:

- inference separates "trained model" from "training loop"
- FastAPI or BentoML creates a real service boundary: request in, prediction out
- Docker makes the service portable enough to run outside the original notebook/project environment

Stop when:

- the service can start
- one request returns a prediction
- the model reload path is tested
- the Docker image can run the service

Resources and stopping points:

- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
  - Read/run: [First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/), [Request Body](https://fastapi.tiangolo.com/tutorial/body/), and [Request Files](https://fastapi.tiangolo.com/tutorial/request-files/).
  - Focus on: `FastAPI()`, route functions, `POST`, JSON input/output, `UploadFile`, and returning a prediction dictionary.
  - Stop when: you have `/health` and `/predict` endpoints and one request returns a class prediction from the saved Project 1 model.
  - Skip for now: authentication, databases, background tasks, dependency injection, async architecture, deployment providers.

- [BentoML Hello World](https://docs.bentoml.org/en/latest/get-started/hello-world.html) and [BentoML Services](https://docs.bentoml.org/en/latest/build-with-bentoml/services.html)
  - Use only if BentoML feels clearer than raw FastAPI after Project 1.
  - Read/run: define a `service.py`, expose one API method, run `bentoml serve`, call the endpoint locally.
  - Focus on: model-serving object, API method, local server, request/response shape.
  - Stop when: the local BentoML service returns one prediction.
  - Skip for now: BentoCloud, scaling configuration, multiple services, deployment automation.

- [Docker official workshop](https://docs.docker.com/get-started/workshop/)
  - Revisit after the FastAPI or BentoML service exists.
  - Read/run: the parts needed to write a `Dockerfile`, build an image, and run the service with a port mapping.
  - Focus on: copying code into the image, installing dependencies, exposing a port, and starting the server process.
  - Stop when: `docker run -p ...` starts the service and the prediction request still works.
  - Skip for now: Compose, registries, Kubernetes, cloud deployment, security hardening.

#### 4. Just Before Project 2: Light MLOps Preview

Do this:

- run the Kubernetes official basics tutorial: deploy, expose, scale, update
- read DataTalksClub MLOps Zoomcamp Module 1 and Module 2 selectively
- skim the deployment module only enough to recognize batch vs real-time serving

Why:

- Kubernetes only becomes meaningful after there is a containerized service to deploy
- MLOps experiment tracking matters more once Project 1 has real runs to compare
- deployment patterns are easier to understand after one model has already been served locally

Resources and stopping points:

- [Kubernetes official basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
  - Read/run: Create a cluster, Deploy an app, Explore your app, Expose your app publicly, Scale up your app, Update your app.
  - Focus on: cluster, node, pod, deployment, service, scale, rollout, rollback, logs.
  - Stop when: you can explain how a containerized app becomes a running service with multiple replicas.
  - Skip for now: Helm, ingress controllers, persistent volumes, secrets, service meshes, autoscaling policy, managed cloud clusters.

- [DataTalksClub MLOps Zoomcamp](https://datatalks.club/docs/courses/mlops-zoomcamp/) and [curriculum](https://datatalks.club/docs/courses/mlops-zoomcamp/curriculum/)
  - Read selectively: Module 1 "Introduction" and Module 2 "Experiment Tracking & Model Management."
  - Skim selectively: Module 4 "Model Deployment" only for batch vs online/web-service vs streaming vocabulary.
  - Focus on: MLOps maturity, experiment tracking, model saving/loading, model deployment categories.
  - Stop when: you can map Project 1 onto these words: training script, experiment run, model artifact, inference service, batch vs online prediction.
  - Skip for now: Module 3 orchestration, Module 5 monitoring, Module 6 best practices, final project, AWS Kinesis/Lambda, Terraform, Prometheus/Grafana.

- [DataTalksClub MLOps Zoomcamp GitHub repo](https://github.com/DataTalksClub/mlops-zoomcamp)
  - Use it as the navigation source if the docs page points to code or videos.
  - Focus on: `01-intro`, `02-experiment-tracking`, and only the deployment overview in `04-deployment`.
  - Stop when: the repo layout makes sense.
  - Skip for now: homework completion, certificate logistics, cohort deadlines, final project rubric.

Do not do yet:

- full Kubernetes operations
- Terraform infrastructure
- production monitoring stacks
- cloud streaming systems
- final MLOps capstone-style architecture

### Carry-Forward Rules

- Learn the production skill at the moment the ML project creates demand for it.
- Prefer one short external module over one full course detour.
- Keep D2L as the main algorithm spine.
- Keep ops/devops as a parallel support lane, not a replacement path.
- Do not productionize every D2L chapter.
- Production-shape the projects, especially Project 1 and later.

## Appendix: Reviewed Resource Dump

This appendix records the July 2026 resource dump review.

The dump included books, courses, GitHub repos, Chinese-language RAG/LLM materials, computer vision resources, theory texts, comp-neuro material, AI engineering resources, and LLM-from-scratch systems material.

Conclusion:

Do not expand the active roadmap right now. The mainline remains D2L plus the project checkpoints. Most resources in the dump are useful only after a project creates demand for them.

Active rule:

- do not add a resource just because it is good
- add a resource only when it answers a current project question
- prefer one good resource per project block over five parallel resources
- select the primary project reference close to the project start
- use GitHub awesome lists as idea mines, not curricula

### Resource Dump Items Worth Parking

Classic ML / job breadth:

- ISLP Python labs
- Hands-On Machine Learning with Scikit-Learn and PyTorch
- Stanford CS229 notes, selectively

Use case:

These are useful if classic ML, tabular modeling, feature engineering, cross-validation, regularization, trees, SVMs, or baseline evaluation feel weak. They support job preparation, but should not interrupt Project 0.

When to consider:

- after Project 0
- before or alongside a classic ML/tabular side project
- when interview prep exposes weak classical ML fundamentals

AI engineering / hiring signal:

- AI Engineering Field Guide by Alexey Grigorev
- Chip Huyen, AI Engineering
- Chip Huyen, Designing Machine Learning Systems

Use case:

These support the career/product track: what companies hire for, how AI systems are evaluated, how RAG/agent systems are built and debugged, what makes a portfolio project legible.

When to consider:

- during the RAG/API/agent phase
- when shaping portfolio projects
- when preparing for AI engineering interviews

Computational neuroscience / research bridge:

- Neuromatch computational neuroscience tutorials

Use case:

This is the best near-term bridge to the algorithm topology. Use only selected modules: model fitting, sparse/local learning themes, dynamical systems, biological neuron models, STDP, reinforcement learning, and causality where relevant.

When to consider:

- after Project 5, if research energy is high
- after Project 3, if attention or memory experiments create a concrete question
- when extending Project 0 into local learning, replay, or catastrophic-forgetting experiments

LLM from scratch / systems capstone:

- Stanford CS336
- minimind
- nano-vLLM
- selected "build a reasoning model" or LLM-from-scratch materials

Use case:

These are later-stage implementation-heavy resources for tokenizers, transformers, pretraining, evaluation, systems, inference, and scaling.

When to consider:

- during or after Project 3, once the transformer baseline is stable
- after optional Project 3B, if systems and inference become the focus
- after Project 5, if NLP pretraining/application work creates demand
- not during Projects 0-2

Theory shelf:

- Bishop, Pattern Recognition and Machine Learning
- Foundations of Machine Learning
- Elements of Deep Learning
- Optimization for Data Analysis
- Linear Algebra and Optimization for Machine Learning
- AIMA

Use case:

These are not active curriculum. Use them as references when a concrete project or D2L chapter creates a theory gap.

When to consider:

- when reading papers becomes a real blocker
- when optimization/generalization questions become concrete
- after project experience makes the abstractions less floating

Vision shelf:

- OpenCV resources
- YOLOv5
- image processing books
- large vision-language model materials
- deep image recognition books

Use case:

Useful only after CNN basics and modern vision chapters. These should support a vision project, not replace D2L vision foundations.

When to consider:

- after D2L Chapters 8 and 14
- during Project 1 for CNN architecture support
- during Project 4 for augmentation, transfer learning, detection, segmentation, or deployment

RAG / LangChain / app-development shelf:

- production RAG books
- LangChain books
- RAG fusion repos
- Chinese-language RAG full-stack guides
- DAIR-style prompt/LLM resources

Use case:

Useful only once the minimal RAG pipeline exists. Do not begin with LangChain abstractions before understanding retrieval, chunking, embeddings, evaluation, and API behavior.

When to consider:

- during optional Project 3B
- after a minimal RAG pipeline answers questions against a known corpus
- when adding production evaluation, observability, or agent tooling

Hardware / edge / systems shelf:

- Harvard edge/AI systems resources
- hardware-adjacent LLM deployment and inference resources
- C++/runtime/system materials

Use case:

Deferred. The hardware topology stays a north star, not active implementation work.

When to consider:

- after D2L Chapter 13 computational performance
- after software routing/local-learning experiments produce meaningful questions
- before any serious NUMA, gem5, C++, or hardware simulation work

### Resources Intentionally Not Added To Active Roadmap

The rest of the dump is intentionally parked.

Examples:

- broad GitHub resource lists
- generic "50 ML projects" lists
- multiple overlapping RAG/LangChain books
- multiple overlapping computer-vision books
- advanced theory texts without an immediate project need
- LLM deployment repos before transformer/system foundations

Reason:

They are not bad resources. They are premature or duplicative for the current stage.

Current priority remains:

```text
Project 0
-> D2L Chapters 5-6
-> D2L Chapters 7-8
-> Project 1: paper-guided CNN system
-> D2L Chapters 9-10
-> Project 2: paper-guided RNN sequence system
-> D2L Chapters 11-13
-> Project 3: transformer / small LLM-style training system
-> optional Project 3B: RAG/API/agent system
-> D2L Chapter 14
-> Project 4: paper-guided computer vision system
-> D2L Chapters 15-16
-> Project 5: NLP representation, search, and application system
```
