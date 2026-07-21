# D2L Project Roadmap

Last updated: 2026-07-20

## Purpose

This document is the high-level topology for project work around the D2L study path.

The D2L notebooks teach concepts in small controlled pieces. The projects below are where those pieces get forced into one working system: data, tensors, model code, loss, gradients, optimization, evaluation, failure analysis, and iteration.

This is not a full walkthrough yet. Each project should later receive its own tutorial or notebook sequence.

## Learning Intensity Rule

Each project should be dense but bounded.

Every project must include:

- a from-scratch version before the concise framework version
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
- using library magic before the manual version exists
- building product polish before the learning mechanism works

## Roadmap Shape

This roadmap is not saying D2L ends at Chapter 7. D2L continues through Chapter 21. The projects are checkpoints inserted between D2L blocks so the material becomes mechanical rather than only readable.

```text
D2L Chapters 2-4
-> Project 0: Sparse Linear Experts Under Distribution Shift
-> D2L Chapters 5-7
-> Project 1: CNN Image Classifier + Custom Optimizer
-> D2L Chapters 8, 12-14 as the deeper vision/performance pass
-> D2L Chapters 9-11 and 15-16 as the sequence/attention/NLP pass
-> Project 2: NLP Embeddings and Similarity Search
-> Project 3: RAG/API/Agent System
-> D2L Chapters 17-21 as advanced/specialized methods
-> Later: full AI product / startup-style system / research-track experiments
```

## Project 0: Sparse Linear Experts Under Distribution Shift

Timing: after D2L Chapters 2-4, before D2L Chapters 5-7.

Role: integration project for tensors, preprocessing, linear algebra, calculus, autodiff, linear regression, softmax regression, generalization, weight decay, and distribution shift.

Core idea:

Build a small simulator where a router chooses among several simple linear experts. Each expert is only a Chapter 3 or Chapter 4 model. The system should feel like a toy version of the PFC/router idea, without pretending to be real hardware or real MoE training.

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

## D2L Chapters 5-7 Bridge

After Project 0, continue with:

- Chapter 5: multilayer perceptrons, nonlinear activations, initialization, dropout, generalization
- Chapter 6: modules, parameters, custom layers, initialization, file I/O, devices
- Chapter 7: convolution, padding, stride, channels, pooling, LeNet

The point of this stretch is to upgrade from linear models to compositional models.

The mental shift:

- Chapter 3-4: one linear map can learn simple decision boundaries
- Chapter 5: stacked nonlinear layers can learn richer functions
- Chapter 6: models become reusable software objects
- Chapter 7: spatial structure matters, especially for images

Important scope note:

Chapters 5-7 are not the end of D2L. They are the first point where a CNN project becomes possible. Chapter 7 gives the basic mechanics of convolution, padding, stride, channels, pooling, and LeNet. Later D2L chapters make that practical and modern.

## Full D2L Continuation Map

Use the projects as reinforcement checkpoints while still continuing through D2L.

Foundation block:

- Chapters 2-4: tensors, data, linear algebra, calculus, autodiff, regression, classification, generalization, distribution shift
- Project 0 belongs here

Basic deep learning block:

- Chapters 5-7: MLPs, modules, parameters, initialization, dropout, CNN basics
- Project 1 can start here as a small CNN training project

Modern vision/performance block:

- Chapter 8: modern CNN architectures
- Chapter 12: optimization algorithms
- Chapter 13: computational performance
- Chapter 14: computer vision
- Project 1 can be revisited here with stronger models, augmentation, transfer learning, and better training discipline

Sequence and NLP block:

- Chapter 9: recurrent neural networks
- Chapter 10: modern recurrent neural networks
- Chapter 11: attention mechanisms and transformers
- Chapter 15: NLP pretraining
- Chapter 16: NLP applications
- Project 2 belongs here, starting simple and becoming more meaningful after attention/transformers

Advanced/specialized block:

- Chapter 17: reinforcement learning
- Chapter 18: Gaussian processes
- Chapter 19: hyperparameter optimization
- Chapter 20: generative adversarial networks
- Chapter 21: recommender systems
- These chapters become optional branches depending on whether the next goal is career, product, research, or theory

## Project 1: CNN Image Classifier + Custom Optimizer

Timing: after D2L Chapters 5-7.

Role: first real deep learning training project.

Core idea:

Train an image classifier twice: first as a plain MLP baseline, then as a CNN. Use the project to understand why convolution, pooling, channels, initialization, dropout, and weight decay exist.

Main learning targets:

- load a real image dataset
- inspect image tensors and label tensors
- build an MLP baseline
- build a CNN model
- write a clean training loop
- track train/validation/test metrics
- diagnose overfitting
- use dropout and weight decay intentionally
- save and reload model weights
- hand-write a basic `torch.optim.Optimizer` subclass

Suggested dataset options:

- Fashion-MNIST for easiest continuity with D2L
- MNIST if the goal is pure debugging simplicity
- CIFAR-10 if ready for harder images

Custom optimizer add-on:

Start with simple SGD as a custom optimizer. Then optionally add momentum or a gated-update variant inspired by Stage 3/4 ideas.

Done criteria:

- MLP baseline works
- CNN beats or meaningfully differs from MLP
- training curves are saved
- overfitting is demonstrated and reduced
- custom optimizer can train at least one model without using Adam
- final notes explain convolution, pooling, channels, and optimizer state in plain language

## Project 2: NLP Embeddings and Similarity Search

Timing: after Project 1 and after enough D2L sequence/attention/NLP material to make text representations feel grounded. A shallow TF-IDF warmup can happen earlier, but the real version belongs after Chapters 9-11 and 15-16.

Role: first language representation project before RAG.

Core idea:

Build a small text system that turns text into vectors, uses those vectors for classification and similarity search, and evaluates whether the retrieved neighbors or predicted labels make sense.

Main learning targets:

- tokenize text
- build simple bag-of-words or TF-IDF features
- train a text classifier
- compute cosine similarity
- build a small embedding search index
- compare keyword search against vector similarity
- evaluate retrieval quality
- inspect failure cases manually

Suggested dataset options:

- movie review sentiment
- news topic classification
- support-ticket category classification
- personal notes or curated article snippets

Connection to Project 0:

- query vectors return
- cosine similarity becomes a real retrieval tool
- region/expert routing becomes document or embedding retrieval
- distribution shift becomes topic shift, style shift, or vocabulary shift

Optional research hook:

After the baseline works, skim selected computational neuroscience material from Neuromatch only where it clarifies sparse coding, local learning, or representational similarity.

Done criteria:

- text classifier baseline works
- similarity search returns inspectable neighbors
- retrieval quality is measured with simple metrics
- at least one failure mode is analyzed
- final notes explain embeddings as geometry, not magic

## Project 3: RAG/API/Agent System

Timing: after Project 2.

Role: transition from model learner to AI systems builder.

Core idea:

Build a retrieval-augmented question-answering system, wrap it in an API, add evaluation and logging, then extend it into a small agent with tool use and memory/state.

Main learning targets:

- ingest documents
- chunk text
- create embeddings
- store vectors
- retrieve relevant chunks
- generate answers with citations or evidence
- evaluate retrieval and answer quality
- serve the system with FastAPI
- track latency and failures
- add basic logging and monitoring
- build a small agent loop with tools and state

Minimum system components:

- ingestion pipeline
- vector index
- retriever
- answer generator
- evaluation set
- FastAPI service
- logs for query, retrieved chunks, answer, latency, and errors
- simple UI or CLI client

Agent extension:

The agent should use tools deliberately, not as decoration. Tool use should have observable state, failure handling, and clear stop conditions.

Optional Stage 6 research hook:

Once the ordinary agent memory/state version works, experiment with a toy session-local adapter or writable memory module. It should be instantiated per session and either merged, saved, or discarded based on evaluation.

Done criteria:

- RAG pipeline answers questions against a known corpus
- retrieval can be evaluated separately from generation
- API can serve requests
- logs expose latency and failure cases
- agent loop can use at least one real tool
- final notes distinguish model behavior, retrieval behavior, and system behavior

## Later Project Track

After these four projects, choose based on goal.

Career/product track:

- full AI product with RAG, agent, API, UI, evals, and deployment
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

- D2L Chapters 5-7

During Project 1:

- PyTorch optimizer documentation
- selected D2L material on initialization, dropout, and convolution
- continue into D2L Chapter 8 and Chapter 14 when the basic CNN project needs stronger vision practice

After the first CNN project:

- D2L Chapter 8 for modern CNN architectures
- D2L Chapter 12 for optimizers
- D2L Chapter 13 for computational performance
- D2L Chapter 14 for practical computer vision

Before and during Project 2:

- D2L Chapters 9-11 for sequence models, attention, and transformers
- D2L Chapters 15-16 for NLP pretraining and NLP applications
- Neuromatch computational neuroscience tutorials, selectively
- Chip Huyen, Designing Machine Learning Systems

During Project 3:

- Chip Huyen, AI Engineering
- production RAG material only after a minimal RAG baseline works

Later:

- D2L Chapters 17-21 depending on interest and project need
- Hands-On Machine Learning with Scikit-Learn and PyTorch for breadth and repetition
- Elements of Deep Learning for math comfort
- selected papers only when a project creates a concrete reason to read them

## Current Next Step

The Project 0 tutorial now lives at:

```text
Project 0 - Sparse Linear Experts Under Distribution Shift/TUTORIAL.md
```

Work through it before continuing to D2L Chapters 5-7. It defines:

- exact dataset design
- exact expert/region design
- exact routing function
- exact training stages
- exact experiments
- exact metrics
- expected bugs and debugging checks
- a hand-typing-friendly tutorial sequence

## Appendix: Reviewed Resource Dump

This appendix records the July 2026 resource dump review.

The dump included books, courses, GitHub repos, Chinese-language RAG/LLM materials, computer vision resources, theory texts, comp-neuro material, AI engineering resources, and LLM-from-scratch systems material.

Conclusion:

Do not expand the active roadmap right now. The mainline remains D2L plus the project checkpoints. Most resources in the dump are useful only after a project creates demand for them.

Active rule:

- do not add a resource just because it is good
- add a resource only when it answers a current project question
- prefer one good resource per project block over five parallel resources
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

- after the NLP/search project, if research energy is high
- between Project 2 and Project 3, selectively
- when extending Project 0 into local learning, replay, or catastrophic-forgetting experiments

LLM from scratch / systems capstone:

- Stanford CS336
- minimind
- nano-vLLM
- selected "build a reasoning model" or LLM-from-scratch materials

Use case:

These are later-stage implementation-heavy resources for tokenizers, transformers, pretraining, evaluation, systems, inference, and scaling.

When to consider:

- after D2L attention/transformer chapters
- after the NLP embeddings/search project
- after basic RAG/API work is already functional
- not during Project 0

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
- when revisiting the CNN project with real augmentation, transfer learning, detection, segmentation, or deployment

RAG / LangChain / app-development shelf:

- production RAG books
- LangChain books
- RAG fusion repos
- Chinese-language RAG full-stack guides
- DAIR-style prompt/LLM resources

Use case:

Useful only once the minimal RAG pipeline exists. Do not begin with LangChain abstractions before understanding retrieval, chunking, embeddings, evaluation, and API behavior.

When to consider:

- during Project 3
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
-> D2L Chapters 5-7
-> CNN/custom optimizer project
-> D2L modern vision/optimization/performance
-> sequence/attention/NLP block
-> NLP/search project
-> RAG/API/agent project
```
