"""Generate Chapter 9 notebooks for the D2L rewrite project.

Chapter 9 introduces sequence prediction, text data, language models, recurrent
state, a from-scratch RNN, PyTorch's RNN layer, and backpropagation through
time. All examples use inline or synthetic data and require no downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Chapter 9 - Recurrent Neural Networks"
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
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:03d}")
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
    path = OUT_DIR / filename
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")


def setup_cell(extra: str = "") -> dict:
    body = """import math
import random
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
random.seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

def shape(x):
    return tuple(x.shape)
"""
    if extra.strip():
        body += "\n" + extra.strip() + "\n"
    return code(body)


def title_cell(title: str, purpose: str, outcomes: list[str]) -> dict:
    items = "\n".join(f"- {item}" for item in outcomes)
    return md(
        f"""# {title}

{purpose.strip()}

## How to use this notebook

Run the notebook from top to bottom in a clean kernel. Everything is generated from small tensors or inline text, so there are no downloads. Before important cells, predict the time, batch, feature, vocabulary, and hidden-state shapes. Treat every assertion as an executable contract rather than decoration.

## You are done when you can

{items}
"""
    )


def checkpoint(section: str, questions: list[str]) -> dict:
    items = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return md(
        f"""## {section} Checkpoint

Answer these without rerunning the notebook. Short markdown answers are enough.

{items}
"""
    )


def build_91() -> None:
    cells = [
        title_cell(
            "Chapter 9.1 - Working with Sequences",
            "A sequence is an ordered collection whose position matters. This notebook turns a time series into supervised examples, trains a small autoregressive predictor, and separates one-step prediction from recursive forecasting.",
            [
                "explain why ordinary shuffled rows do not fully describe sequence prediction",
                "build lagged features and correctly aligned labels",
                "split time series without leaking the future into training",
                "distinguish one-step predictions from recursive multi-step forecasts",
                "diagnose an off-by-one target bug",
            ],
        ),
        setup_cell(),
        md(
            """## 9.1.0 The Problem This Notebook Solves

In ordinary tabular regression, each row can often be interpreted independently. In a sequence, the observation at time `t` is related to earlier observations. A temperature, price, sensor reading, or token arrives in an order.

An **autoregressive model** predicts a future value using previous values from the same sequence. If the lag length is `tau`, its input at time `t` is

```text
[x[t - tau], ..., x[t - 2], x[t - 1]]
```

and its target is `x[t]`. The word *lag* means a backward time offset. The immediate engineering problem is therefore not merely choosing a network. It is constructing examples without breaking time alignment.
"""
        ),
        md(
            """## 9.1.1 Autoregressive Models: Turn History Into Features

We first create a smooth synthetic signal with noise. Then `make_lagged` slides a window across it. One input row contains `tau` consecutive past values; the corresponding label is the very next value.

Before running the cell, predict the shapes when a length-12 sequence uses `tau=4`. There are only `12 - 4 = 8` complete input-target pairs.
"""
        ),
        code(
            """def make_lagged(series, tau):
    if series.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if not 1 <= tau < len(series):
        raise ValueError("tau must be between 1 and len(series) - 1")
    features = torch.stack([series[i : i + tau] for i in range(len(series) - tau)])
    labels = series[tau:].reshape(-1, 1)
    return features, labels

tiny = torch.arange(12, dtype=torch.float32)
X_tiny, y_tiny = make_lagged(tiny, tau=4)
print(X_tiny[:3])
print(y_tiny[:3])
assert shape(X_tiny) == (8, 4)
assert shape(y_tiny) == (8, 1)
assert torch.equal(X_tiny[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
assert y_tiny[0].item() == 4.0"""
        ),
        md(
            """## 9.1.2 Sequence Models and Chronological Splits

A **time series** is a sequence indexed by time. A **univariate** time series has one measured value at each time; a multivariate series has several. The lagged table above is a fixed-window sequence model: it assumes the last `tau` values contain the useful context.

Validation must imitate deployment. If training randomly samples windows from the whole timeline, a window from late in the series can influence the model before we evaluate on an earlier window. A **chronological split** trains on the past and evaluates on a later contiguous region.

The model below is an MLP, not yet an RNN. That is intentional: it isolates the data contract shared by sequence predictors.
"""
        ),
        code(
            """time = torch.arange(0, 120, dtype=torch.float32)
series = torch.sin(time * 0.12) + 0.15 * torch.randn(120)
tau = 8
features, labels = make_lagged(series, tau)

split = 84
X_train, y_train = features[:split], labels[:split]
X_valid, y_valid = features[split:], labels[split:]
model = nn.Sequential(nn.Linear(tau, 16), nn.ReLU(), nn.Linear(16, 1))

assert X_train[-1, -1].item() == series[split + tau - 2].item()
assert y_train[-1].item() == series[split + tau - 1].item()
assert X_valid[0, 0].item() == series[split].item()
assert shape(model(X_train[:3])) == (3, 1)"""
        ),
        md(
            """## 9.1.3 Training: What Repeats and What Changes

One **epoch** is one complete optimization pass over the chosen training examples. In each epoch:

1. the current parameters produce predictions;
2. mean squared error measures prediction error;
3. `backward()` stores gradients in parameter `.grad` fields;
4. `optimizer.step()` changes parameters;
5. `zero_grad()` clears old gradients before the next pass.

The data stay fixed here. Model parameters and optimizer state change. Full-batch training keeps the control flow visible.
"""
        ),
        code(
            """optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
for epoch in range(80):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = F.mse_loss(predictions, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    train_mse = F.mse_loss(model(X_train), y_train).item()
    valid_mse = F.mse_loss(model(X_valid), y_valid).item()

print({"train_mse": train_mse, "valid_mse": valid_mse})
assert math.isfinite(train_mse)
assert math.isfinite(valid_mse)"""
        ),
        md(
            """## 9.1.4 Prediction: One Step Versus Many Steps

A **one-step prediction** gets the real previous values at every time. A **recursive forecast** predicts one value, appends that prediction to its own history, and uses it to predict again. Recursive errors can compound because later inputs contain earlier mistakes.

The cell starts both methods at the validation boundary. The one-step path uses `X_valid`, which was made from observed values. The recursive path receives only the last real training window and then feeds itself.
"""
        ),
        code(
            """with torch.no_grad():
    one_step = model(X_valid).squeeze(1)
    history = X_valid[0].clone()
    recursive_values = []
    for _ in range(len(X_valid)):
        next_value = model(history.reshape(1, -1)).squeeze()
        recursive_values.append(next_value)
        history = torch.cat((history[1:], next_value.reshape(1)))
    recursive = torch.stack(recursive_values)

print("first five one-step:", one_step[:5])
print("first five recursive:", recursive[:5])
assert shape(one_step) == shape(recursive) == (len(X_valid),)
assert torch.allclose(one_step[0], recursive[0])"""
        ),
        md(
            """## 9.1.5 Break It Deliberately: An Off-by-One Label

If labels use the final value *inside* each input window, the model is rewarded for copying a value it already received. The shapes still look correct, so shape checks alone cannot catch this bug. Alignment needs a value-level contract.
"""
        ),
        code(
            """wrong_labels = series[tau - 1 : -1].reshape(-1, 1)
print("input window:", features[0])
print("wrong label:", wrong_labels[0].item())
print("correct label:", labels[0].item())

assert shape(wrong_labels) == shape(labels)
assert wrong_labels[0].item() == features[0, -1].item()
assert labels[0].item() == series[tau].item()
assert wrong_labels[0].item() != labels[0].item()"""
        ),
        checkpoint(
            "9.1",
            [
                "For a sequence of length N and lag tau, how many complete examples exist?",
                "Why can a random train-validation split leak temporal information?",
                "What information differs between one-step and recursive prediction?",
                "Why does correct shape not prove correct label alignment?",
                "Which objects change during the training loop?",
            ],
        ),
    ]
    write_nb("Chapter 9.1 - Working with Sequences.ipynb", cells)


def build_92() -> None:
    cells = [
        title_cell(
            "Chapter 9.2 - Converting Raw Text into Sequence Data",
            "Models consume numbers, while language arrives as characters and words. This notebook builds the complete text pipeline: normalize text, tokenize it, construct a vocabulary, map tokens to integer IDs, and inspect language statistics.",
            [
                "distinguish a text line, token, token ID, vocabulary, and corpus",
                "tokenize the same text at word and character granularity",
                "read and extend a small vocabulary class",
                "round-trip known tokens through IDs",
                "handle unknown tokens deliberately instead of silently failing",
            ],
        ),
        setup_cell(),
        md(
            """## 9.2.0 The Problem This Notebook Solves

Raw text is not yet training data. A sequence model needs a deterministic mapping:

```text
raw text -> cleaned lines -> tokens -> vocabulary -> integer token IDs
```

A **token** is the unit the model sees, such as a word or character. A **vocabulary** is the finite lookup table between tokens and integer IDs. A **corpus** is the complete token-ID sequence used as data.

Integer IDs are names, not measurements. Token ID 8 is not twice token ID 4. Later code will convert IDs to one-hot vectors or learned embeddings before treating them as numeric features.
"""
        ),
        md(
            """## 9.2.1 Reading and Normalizing an Inline Dataset

Real D2L code can download a book and read it line by line. This notebook uses an original inline corpus so it is reproducible offline. Normalization lowercases letters, replaces punctuation with spaces, collapses repeated whitespace, and discards blank lines.
"""
        ),
        code(
            """import re

RAW_TEXT = (
    "Signals arrive in order; models remember useful context.\\n"
    "Context changes a prediction, and prediction changes the next context!\\n"
    "Small sequences make hidden mechanics visible."
)

def read_inline_text(text):
    cleaned = []
    for raw_line in text.strip().splitlines():
        line = re.sub("[^A-Za-z]+", " ", raw_line).strip().lower()
        if line:
            cleaned.append(line)
    return cleaned

lines = read_inline_text(RAW_TEXT)
print(lines)
assert len(lines) == 3
assert all(line == line.lower() for line in lines)
assert all(";" not in line and "!" not in line for line in lines)"""
        ),
        md(
            """## 9.2.2 Tokenization: Choose the Unit of Sequence

**Word tokenization** splits a line into word units. **Character tokenization** splits it into individual characters, including spaces if we keep them. The choice changes sequence length and vocabulary size.

Word tokens make sequences shorter but need an unknown-token policy for unseen words. Character tokens use a small vocabulary but require many more time steps to express the same sentence.
"""
        ),
        code(
            """def tokenize(lines, mode="word"):
    if mode == "word":
        return [line.split() for line in lines]
    if mode == "char":
        return [list(line) for line in lines]
    raise ValueError("mode must be 'word' or 'char'")

word_tokens = tokenize(lines, "word")
char_tokens = tokenize(lines, "char")
print("words:", word_tokens[0])
print("characters:", char_tokens[0][:12])
assert len(char_tokens[0]) > len(word_tokens[0])
assert "signals" in word_tokens[0]"""
        ),
        md(
            """## 9.2.3 Vocabulary: A Small Inspectable Python Class

The class below stores two inverse mappings:

- `idx_to_token[id]` returns a token;
- `token_to_idx[token]` returns an ID.

`self` is the instance being operated on. `__init__` runs when `Vocab(...)` constructs that instance. `__len__` lets Python call `len(vocab)`. `__getitem__` lets square brackets call `vocab[tokens]`. These are ordinary methods connected to familiar Python syntax.

`<unk>` is the unknown token. Reserved tokens are inserted before corpus tokens so their IDs are stable.
"""
        ),
        code(
            """class Vocab:
    def __init__(self, token_lines, min_freq=1, reserved_tokens=None):
        reserved_tokens = reserved_tokens or []
        counts = Counter(token for line in token_lines for token in line)
        self.token_freqs = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        self.idx_to_token = ["<unk>"]
        for token in reserved_tokens:
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
        for token, frequency in self.token_freqs:
            if frequency >= min_freq and token not in self.idx_to_token:
                self.idx_to_token.append(token)
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, str):
            return self.token_to_idx.get(tokens, self.unk)
        return [self[token] for token in tokens]

    @property
    def unk(self):
        return 0

    def to_tokens(self, indices):
        if isinstance(indices, int):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]"""
        ),
        md(
            """The `@property` decorator makes the zero-argument method `unk` readable as `vocab.unk` instead of `vocab.unk()`. It does not precompute a new value; Python calls the method when the attribute is accessed.

Before the next cell, predict the ID of `<unk>` and what ID the unseen word `memory` receives.
"""
        ),
        code(
            """vocab = Vocab(word_tokens, min_freq=1, reserved_tokens=["<pad>"])
known = ["context", "prediction"]
known_ids = vocab[known]
round_trip = vocab.to_tokens(known_ids)

print(list(enumerate(vocab.idx_to_token)))
print("known round trip:", known, known_ids, round_trip)
print("unseen token ID:", vocab["memory"])
assert vocab["<unk>"] == vocab.unk == 0
assert vocab["<pad>"] == 1
assert round_trip == known
assert vocab["memory"] == vocab.unk"""
        ),
        md(
            """## 9.2.4 Putting It All Together: Build a Corpus

Flattening removes line boundaries and creates one long stream. Keeping a parallel token list makes the alignment inspectable. Every corpus element is an integer in `[0, len(vocab))`.
"""
        ),
        code(
            """flat_tokens = [token for line in word_tokens for token in line]
corpus = torch.tensor(vocab[flat_tokens], dtype=torch.long)
decoded = vocab.to_tokens(corpus.tolist())

print("corpus IDs:", corpus)
print("decoded prefix:", decoded[:8])
assert corpus.ndim == 1
assert len(corpus) == len(flat_tokens)
assert decoded == flat_tokens
assert 0 <= corpus.min() and corpus.max() < len(vocab)"""
        ),
        md(
            """## 9.2.5 Exploratory Language Statistics

A **unigram** is one token. A **bigram** is an ordered pair of adjacent tokens. A **trigram** contains three. Their frequency tables reveal both common units and local order.

Natural-language frequencies are often highly uneven: a few tokens occur frequently and many are rare. Small corpora are too noisy for broad conclusions, but the counting mechanism is the same at scale.
"""
        ),
        code(
            """unigrams = Counter(flat_tokens)
bigrams = Counter(zip(flat_tokens[:-1], flat_tokens[1:]))
trigrams = Counter(zip(flat_tokens[:-2], flat_tokens[1:-1], flat_tokens[2:]))

print("top unigrams:", unigrams.most_common(5))
print("top bigrams:", bigrams.most_common(5))
print("top trigrams:", trigrams.most_common(3))
assert sum(unigrams.values()) == len(flat_tokens)
assert sum(bigrams.values()) == len(flat_tokens) - 1
assert sum(trigrams.values()) == len(flat_tokens) - 2"""
        ),
        md(
            """## 9.2.6 Break It Deliberately: No Unknown-Token Policy

A raw dictionary lookup raises `KeyError` for an unseen token. That failure is safer than silently assigning an arbitrary ID, but production text pipelines normally reserve an explicit unknown ID so inference can continue predictably.
"""
        ),
        code(
            """unsafe_mapping = {"known": 0}
try:
    unsafe_mapping["never_seen"]
except KeyError as error:
    print(type(error).__name__, error)
else:
    raise AssertionError("Expected an unseen token to raise KeyError")

assert vocab["never_seen"] == vocab.unk"""
        ),
        checkpoint(
            "9.2",
            [
                "Why are integer token IDs names rather than numeric measurements?",
                "How does character tokenization trade sequence length for vocabulary size?",
                "What state is stored on each Vocab instance through self?",
                "Why reserve an unknown token?",
                "How many bigrams exist in a flattened sequence of N tokens?",
            ],
        ),
    ]
    write_nb("Chapter 9.2 - Converting Raw Text into Sequence Data.ipynb", cells)


def build_93() -> None:
    cells = [
        title_cell(
            "Chapter 9.3 - Language Models",
            "A language model assigns probabilities to token sequences and predicts the next token from context. This notebook connects the probability chain rule to cross-entropy, perplexity, and two correct ways to partition one corpus into minibatches.",
            [
                "factor a sequence probability into next-token conditional probabilities",
                "interpret cross-entropy and perplexity for language modeling",
                "construct random and sequential subsequence minibatches",
                "trace batch and time axes without guessing",
                "catch a target-shift bug that creates a meaningless copy task",
            ],
        ),
        setup_cell(),
        md(
            """## 9.3.0 The Problem This Notebook Solves

For tokens `x1, x2, ..., xT`, the probability chain rule gives

```text
P(x1, ..., xT) = P(x1) P(x2 | x1) ... P(xT | x1, ..., x[T-1])
```

A **language model** estimates these probabilities. During next-token training, each input position is paired with the token one step to its right. The statistical definition and the array alignment describe the same task.

An **n-gram model** shortens the context to the previous `n - 1` tokens. It is easy to count but cannot flexibly share knowledge across similar contexts. Neural language models learn distributed internal representations instead.
"""
        ),
        md(
            """## 9.3.1 Learning a Count-Based Bigram Model

A bigram model estimates `P(next | current)`. Add-one smoothing gives every possible next token one artificial count, preventing a zero probability for unseen pairs.
"""
        ),
        code(
            """tokens = "time moves and time changes and time moves".split()
vocabulary = sorted(set(tokens))
token_to_id = {token: i for i, token in enumerate(vocabulary)}
V = len(vocabulary)
counts = torch.ones(V, V)

for current, following in zip(tokens[:-1], tokens[1:]):
    counts[token_to_id[current], token_to_id[following]] += 1

probabilities = counts / counts.sum(dim=1, keepdim=True)
time_row = probabilities[token_to_id["time"]]
print(dict(zip(vocabulary, time_row.tolist())))
assert torch.allclose(probabilities.sum(dim=1), torch.ones(V))
assert time_row[token_to_id["moves"]] > time_row[token_to_id["and"]]"""
        ),
        md(
            """## 9.3.2 Cross-Entropy and Perplexity

For the correct next token, negative log-likelihood is `-log(p_correct)`. Averaging it across tokens gives cross-entropy loss. **Perplexity** is

```text
perplexity = exp(average cross-entropy)
```

It can be read as an effective number of equally plausible choices. Perplexity 1 is perfect confidence on every correct token. A uniform guess over `V` tokens has perplexity `V`. Lower is better, but comparisons are meaningful only when tokenization and evaluation data match.
"""
        ),
        code(
            """def perplexity(correct_token_probabilities):
    probabilities = torch.as_tensor(correct_token_probabilities)
    if torch.any(probabilities <= 0) or torch.any(probabilities > 1):
        raise ValueError("probabilities must be in (0, 1]")
    cross_entropy = -torch.log(probabilities).mean()
    return torch.exp(cross_entropy)

perfect = perplexity([1.0, 1.0, 1.0])
uniform_four = perplexity([0.25, 0.25, 0.25])
mixed = perplexity([0.8, 0.4, 0.2])
print(perfect, uniform_four, mixed)
assert torch.allclose(perfect, torch.tensor(1.0))
assert torch.allclose(uniform_four, torch.tensor(4.0))
assert 1 < mixed < 4"""
        ),
        md(
            """## 9.3.3 Partitioning Sequences: The Universal Shift Contract

Suppose `X` contains token IDs with shape `(batch, time)`. Its label `Y` must have the same shape, and every position must satisfy

```text
Y[:, t] is the token immediately after X[:, t]
```

Random iteration samples subsequences from shuffled starting positions. Sequential iteration reshapes a contiguous stream into rows and moves forward in time. Random iteration weakens continuity between minibatches; sequential iteration makes it possible to carry recurrent state between neighboring minibatches.
"""
        ),
        code(
            """def random_sequence_batches(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps - 1)
    usable = corpus[offset:]
    starts = list(range(0, len(usable) - num_steps, num_steps))
    random.shuffle(starts)
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        if len(batch_starts) < batch_size:
            continue
        X = torch.stack([usable[j : j + num_steps] for j in batch_starts])
        Y = torch.stack([usable[j + 1 : j + num_steps + 1] for j in batch_starts])
        yield X, Y

corpus = torch.arange(30)
X_random, Y_random = next(random_sequence_batches(corpus, batch_size=2, num_steps=5))
print(X_random)
print(Y_random)
assert shape(X_random) == shape(Y_random) == (2, 5)
assert torch.equal(Y_random[:, :-1], X_random[:, 1:])"""
        ),
        code(
            """def sequential_sequence_batches(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    usable_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = corpus[offset : offset + usable_tokens].reshape(batch_size, -1)
    Ys = corpus[offset + 1 : offset + 1 + usable_tokens].reshape(batch_size, -1)
    for start in range(0, Xs.shape[1] - num_steps + 1, num_steps):
        yield Xs[:, start : start + num_steps], Ys[:, start : start + num_steps]

random.seed(0)
batches = list(sequential_sequence_batches(torch.arange(40), batch_size=2, num_steps=4))
X0, Y0 = batches[0]
X1, Y1 = batches[1]
print(X0, Y0, X1, sep="\\n")
assert shape(X0) == shape(Y0) == (2, 4)
assert torch.equal(Y0, X0 + 1)
assert torch.equal(X1[:, 0], X0[:, -1] + 1)"""
        ),
        md(
            """## 9.3.4 Break It Deliberately: Labels Equal Inputs

If `Y = X`, the network learns to reproduce the visible current token rather than predict the next token. Accuracy can look excellent while the language-model objective is wrong. We require both a shape contract and a temporal shift contract.
"""
        ),
        code(
            """X = torch.tensor([[2, 4, 1, 3]])
wrong_Y = X.clone()
correct_Y = torch.tensor([[4, 1, 3, 0]])

assert shape(wrong_Y) == shape(correct_Y) == shape(X)
assert torch.equal(wrong_Y, X)
assert torch.equal(correct_Y[:, :-1], X[:, 1:])
try:
    assert torch.equal(wrong_Y[:, :-1], X[:, 1:])
except AssertionError:
    print("Caught: equal-shaped labels violate the next-token shift contract")"""
        ),
        md(
            """## 9.3.5 What This Notebook Does Not Do

It does not estimate serious perplexity from this tiny invented corpus. Reliable language-model evaluation needs a held-out corpus, consistent tokenization, sufficient data, and careful treatment of sequence boundaries. Here the goal is to make probability, loss, and minibatch alignment mechanically exact.
"""
        ),
        checkpoint(
            "9.3",
            [
                "What does the probability chain rule contribute to language modeling?",
                "Why is a zero-probability n-gram dangerous?",
                "What perplexity does a uniform model over V tokens obtain?",
                "What are the axes of a language-model minibatch in this notebook?",
                "Why might sequential partitioning be useful for recurrent state?",
            ],
        ),
    ]
    write_nb("Chapter 9.3 - Language Models.ipynb", cells)


def build_94() -> None:
    cells = [
        title_cell(
            "Chapter 9.4 - Recurrent Neural Networks",
            "A recurrent neural network keeps a hidden state that is updated at every time step. This notebook derives the recurrence, executes it by hand, turns hidden states into character logits, and makes the state-shape contract explicit.",
            [
                "contrast a next-token network with and without hidden state",
                "calculate a recurrent update one time step at a time",
                "identify input, hidden, and output parameter shapes",
                "explain how one set of parameters is reused across time",
                "diagnose a malformed initial hidden state",
            ],
        ),
        setup_cell(),
        md(
            """## 9.4.0 The Problem This Notebook Solves

A fixed-window model can use only the values explicitly placed in its input window. An RNN adds a **hidden state**: a learned numeric summary passed from one time step to the next.

For input `X_t` and previous state `H_(t-1)`, a simple RNN computes

```text
H_t = tanh(X_t W_xh + H_(t-1) W_hh + b_h)
O_t = H_t W_hq + b_q
```

`tanh` bounds each hidden value between -1 and 1. The logits `O_t` are unnormalized scores over possible next tokens. The same weight matrices are reused at every time step; recurrence does not create new parameters for every position.
"""
        ),
        md(
            """## 9.4.1 Neural Networks Without Hidden States

A context-free network maps the current token to next-token logits. Two identical current tokens always produce identical logits, even if their earlier contexts differ. One-hot vectors make the lookup mechanism visible: a vector has length `vocab_size`, with one 1 at the token ID.
"""
        ),
        code(
            """vocab_size = 5
current_ids = torch.tensor([2, 2])
one_hot = F.one_hot(current_ids, num_classes=vocab_size).float()
W_xq = torch.arange(vocab_size * vocab_size, dtype=torch.float32).reshape(vocab_size, vocab_size)
context_free_logits = one_hot @ W_xq

print(one_hot)
print(context_free_logits)
assert shape(one_hot) == (2, vocab_size)
assert torch.equal(context_free_logits[0], context_free_logits[1])"""
        ),
        md(
            """## 9.4.2 Recurrent Networks With Hidden States

The batch below has shape `(batch=2, time=3)`. We transpose its one-hot representation to `(time, batch, vocab)` because the manual loop consumes one complete batch at each time step.

At each iteration, `X_t` changes and `H` is replaced by the new state. Parameters remain the same. The list `outputs` stores one logit tensor per time step.
"""
        ),
        code(
            """batch_ids = torch.tensor([[0, 1, 2], [3, 2, 1]])
inputs = F.one_hot(batch_ids.T, num_classes=vocab_size).float()
hidden_size = 4

W_xh = torch.randn(vocab_size, hidden_size) * 0.1
W_hh = torch.randn(hidden_size, hidden_size) * 0.1
b_h = torch.zeros(hidden_size)
W_hq = torch.randn(hidden_size, vocab_size) * 0.1
b_q = torch.zeros(vocab_size)
H = torch.zeros(batch_ids.shape[0], hidden_size)
outputs = []

for X_t in inputs:
    H = torch.tanh(X_t @ W_xh + H @ W_hh + b_h)
    outputs.append(H @ W_hq + b_q)

logits = torch.stack(outputs)
print("inputs:", shape(inputs), "state:", shape(H), "logits:", shape(logits))
assert shape(inputs) == (3, 2, 5)
assert shape(H) == (2, 4)
assert shape(logits) == (3, 2, 5)"""
        ),
        md(
            """## 9.4.3 Why Order Changes the State

The final hidden state is not a bag-of-tokens count. Swapping token order changes the sequence of nonlinear updates, even when the same tokens are present. This makes state useful for language, where `dog bites person` and `person bites dog` do not mean the same thing.
"""
        ),
        code(
            """def final_state(token_ids):
    state = torch.zeros(1, hidden_size)
    sequence = F.one_hot(torch.tensor(token_ids), num_classes=vocab_size).float()
    for X_t in sequence:
        state = torch.tanh(X_t.reshape(1, -1) @ W_xh + state @ W_hh + b_h)
    return state

state_forward = final_state([0, 1, 2])
state_reversed = final_state([2, 1, 0])
print(state_forward)
print(state_reversed)
assert not torch.allclose(state_forward, state_reversed)"""
        ),
        md(
            """## 9.4.4 RNN-Based Character-Level Language Models

A **character-level language model** receives a character ID at each time and predicts the following character. If logits have shape `(time, batch, vocab)`, flattening the first two axes gives `(time * batch, vocab)`. Labels must be flattened in the same time-major order.

Cross-entropy expects one logit row per target and a target integer ID per row.
"""
        ),
        code(
            """targets_batch_first = torch.tensor([[1, 2, 3], [2, 1, 0]])
targets_time_first = targets_batch_first.T
flat_logits = logits.reshape(-1, vocab_size)
flat_targets = targets_time_first.reshape(-1)
loss = F.cross_entropy(flat_logits, flat_targets)

print("flat logits:", shape(flat_logits), "flat targets:", shape(flat_targets))
print("loss:", loss.item())
assert shape(flat_logits) == (6, vocab_size)
assert shape(flat_targets) == (6,)
assert loss.ndim == 0"""
        ),
        md(
            """## 9.4.5 Break It Deliberately: Hidden-State Shape Mismatch

The hidden state needs one row per sequence in the batch and one column per hidden feature: `(batch, hidden_size)`. A state with three rows cannot be combined with a two-example input batch.
"""
        ),
        code(
            """X_t = inputs[0]
wrong_H = torch.zeros(3, hidden_size)
try:
    torch.tanh(X_t @ W_xh + wrong_H @ W_hh + b_h)
except RuntimeError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected incompatible batch dimensions to fail")"""
        ),
        checkpoint(
            "9.4",
            [
                "What information enters the hidden-state update at time t?",
                "Which RNN quantities change each time step and which are reused?",
                "Why can reversed token order produce a different final state?",
                "What does each axis of `(time, batch, vocab)` mean?",
                "Why must logits and labels use the same flattening order?",
            ],
        ),
    ]
    write_nb("Chapter 9.4 - Recurrent Neural Networks.ipynb", cells)


def build_95() -> None:
    cells = [
        title_cell(
            "Chapter 9.5 - Recurrent Neural Network Implementation from Scratch",
            "This notebook implements a character-level RNN language model from explicit parameters. It covers state initialization, forward recurrence, gradient clipping, a transparent training epoch, and prefix-conditioned decoding.",
            [
                "identify every trainable parameter in a scratch RNN",
                "trace token IDs through one-hot inputs, hidden states, and logits",
                "explain why carried state is detached between minibatches",
                "clip a global gradient norm without changing its direction",
                "decode a prefix and generate additional tokens",
            ],
        ),
        setup_cell(),
        md(
            """## 9.5.0 The Problem This Notebook Solves

Framework RNN layers hide a time loop and several matrix operations. Here we expose them. The implementation is still a PyTorch `nn.Module`, so parameters register correctly with optimizers, but the recurrent computation itself is handwritten.

`nn.Parameter` wraps a tensor that should be trainable. Assigning it as an attribute of a module registers it. `model.parameters()` can then find it. Calling `model(X, state)` invokes the module's `forward` method through `nn.Module.__call__`, which also manages framework hooks.
"""
        ),
        md(
            """## 9.5.1 RNN Model: Explicit Parameters and State

Parameter shapes are contracts:

```text
W_xh: (vocab, hidden)
W_hh: (hidden, hidden)
W_hq: (hidden, vocab)
```

The input arrives as token IDs shaped `(batch, time)`. The model one-hot encodes and transposes them to `(time, batch, vocab)`. It returns flattened logits `(batch * time, vocab)` plus final state `(batch, hidden)`.
"""
        ),
        code(
            """class ScratchRNNLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, sigma=0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.randn(vocab_size, hidden_size) * sigma)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * sigma)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hq = nn.Parameter(torch.randn(hidden_size, vocab_size) * sigma)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def begin_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, token_ids, state):
        inputs = F.one_hot(token_ids.T, self.vocab_size).float()
        outputs = []
        for X_t in inputs:
            state = torch.tanh(X_t @ self.W_xh + state @ self.W_hh + self.b_h)
            outputs.append(state @ self.W_hq + self.b_q)
        return torch.cat(outputs, dim=0), state"""
        ),
        code(
            """model = ScratchRNNLM(vocab_size=7, hidden_size=8)
X = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
state = model.begin_state(batch_size=2, device=X.device)
logits, next_state = model(X, state)

print("registered parameters:", [name for name, _ in model.named_parameters()])
print("logits:", shape(logits), "state:", shape(next_state))
assert shape(logits) == (8, 7)
assert shape(next_state) == (2, 8)
assert len(list(model.parameters())) == 5"""
        ),
        md(
            """## 9.5.2 RNN-Based Language Model Loss and Axis Alignment

The model concatenates outputs in time-major order: all batch examples at time 0, then all at time 1, and so on. Therefore targets must transpose from `(batch, time)` to `(time, batch)` before flattening.
"""
        ),
        code(
            """Y = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 6]])
targets = Y.T.reshape(-1)
loss = F.cross_entropy(logits, targets)

print("targets in model output order:", targets)
print("average token loss:", loss.item())
assert shape(targets) == (8,)
assert targets[:2].tolist() == [1, 2]
assert loss.ndim == 0"""
        ),
        md(
            """## 9.5.3 Gradient Clipping

The **global gradient norm** treats all parameter gradients as pieces of one long vector. If its length exceeds threshold `theta`, clipping multiplies every gradient by the same factor `theta / norm`. The direction is preserved while the step magnitude is capped.

Clipping handles unusually large gradients; it does not repair bad data, guarantee convergence, or recover a gradient that has already vanished.
"""
        ),
        code(
            """def clip_gradients(module, theta):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if not gradients:
        return 0.0
    norm = torch.sqrt(sum(torch.sum(gradient ** 2) for gradient in gradients))
    if norm > theta:
        scale = theta / (norm + 1e-12)
        for gradient in gradients:
            gradient.mul_(scale)
    return norm.item()

model.zero_grad()
loss.backward()
norm_before = clip_gradients(model, theta=0.25)
norm_after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())).item()
print({"before": norm_before, "after": norm_after})
assert norm_after <= 0.25001"""
        ),
        md(
            """## 9.5.4 Training With Consecutive Minibatches

This inline corpus repeats a short pattern. `sequence_batches` makes contiguous `(batch, time)` minibatches. At the start of an epoch, state is zeros. Between adjacent minibatches, the numeric state is carried forward but detached from its old computation graph.

Detaching says: use these state values as the next starting point, but do not backpropagate through all earlier minibatches. This is **truncated backpropagation through time** at minibatch boundaries.
"""
        ),
        code(
            """def sequence_batches(corpus, batch_size, num_steps):
    usable = ((len(corpus) - 1) // batch_size) * batch_size
    Xs = corpus[:usable].reshape(batch_size, -1)
    Ys = corpus[1 : usable + 1].reshape(batch_size, -1)
    for start in range(0, Xs.shape[1] - num_steps + 1, num_steps):
        yield Xs[:, start : start + num_steps], Ys[:, start : start + num_steps]

corpus = torch.tensor(([0, 1, 2, 3, 4, 5, 6] * 30), dtype=torch.long)
model = ScratchRNNLM(vocab_size=7, hidden_size=16)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

def train_epoch(model, corpus, batch_size=2, num_steps=7):
    state = model.begin_state(batch_size, corpus.device)
    total_loss = 0.0
    token_count = 0
    for X, Y in sequence_batches(corpus, batch_size, num_steps):
        state = state.detach()
        optimizer.zero_grad()
        logits, state = model(X, state)
        targets = Y.T.reshape(-1)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        clip_gradients(model, theta=1.0)
        optimizer.step()
        total_loss += loss.item() * targets.numel()
        token_count += targets.numel()
    return math.exp(total_loss / token_count)

perplexities = [train_epoch(model, corpus) for _ in range(8)]
print(perplexities)
assert all(math.isfinite(value) for value in perplexities)"""
        ),
        md(
            """## 9.5.5 Decoding: Warm Up With a Prefix, Then Generate

**Decoding** converts model scores into output tokens. Greedy decoding selects the largest logit at each step. The prefix first warms the hidden state using known tokens. After the prefix ends, each predicted ID becomes the next input.

Greedy generation is deterministic and easy to inspect, but it is not the only strategy. Sampling and beam search make different quality-diversity tradeoffs.
"""
        ),
        code(
            """def predict(prefix, num_predictions, model):
    state = model.begin_state(batch_size=1)
    outputs = [prefix[0]]
    for token in prefix[1:]:
        X = torch.tensor([[outputs[-1]]])
        _, state = model(X, state)
        outputs.append(token)
    for _ in range(num_predictions):
        X = torch.tensor([[outputs[-1]]])
        logits, state = model(X, state)
        outputs.append(int(logits[-1].argmax()))
    return outputs

generated = predict(prefix=[0, 1], num_predictions=10, model=model)
print(generated)
assert generated[:2] == [0, 1]
assert len(generated) == 12
assert all(0 <= token < model.vocab_size for token in generated)"""
        ),
        md(
            """## 9.5.6 Break It Deliberately: Reuse Attached State

After `backward()`, PyTorch normally frees intermediate graph buffers. Reusing a state still attached to that graph and calling `backward()` again tries to traverse freed history. Detach the carried state at the minibatch boundary.
"""
        ),
        code(
            """probe = ScratchRNNLM(vocab_size=7, hidden_size=4)
state = probe.begin_state(batch_size=1)
X1 = torch.tensor([[0, 1]])
Y1 = torch.tensor([1, 2])
logits, state = probe(X1, state)
F.cross_entropy(logits, Y1).backward()

try:
    logits2, state = probe(torch.tensor([[2, 3]]), state)
    F.cross_entropy(logits2, torch.tensor([3, 4])).backward()
except RuntimeError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected backward through freed history to fail")"""
        ),
        checkpoint(
            "9.5",
            [
                "Why does assigning nn.Parameter attributes register trainable tensors?",
                "Why are labels transposed before flattening?",
                "How does global-norm clipping change gradient magnitude and direction?",
                "What information is kept and cut by state.detach()?",
                "What happens during prefix warm-up in decoding?",
            ],
        ),
    ]
    write_nb("Chapter 9.5 - Recurrent Neural Network Implementation from Scratch.ipynb", cells)


def build_96() -> None:
    cells = [
        title_cell(
            "Chapter 9.6 - Concise Implementation of Recurrent Neural Networks",
            "PyTorch's `nn.RNN` packages the recurrent loop into a tested layer. This notebook maps its exact tensor contracts to a character-level language model, trains it on a tiny offline pattern, and decodes predictions.",
            [
                "read the input, output, and hidden-state contracts of nn.RNN",
                "explain the roles of an embedding, recurrent layer, and output projection",
                "handle `batch_first=True` without transposing axes accidentally",
                "train and decode with a concise RNN language model",
                "diagnose hidden-state errors involving layers and batch size",
            ],
        ),
        setup_cell(),
        md(
            """## 9.6.0 The Problem This Notebook Solves

The scratch model made every equation visible. A framework layer reduces repeated code and offers optimized kernels, but it introduces a strict interface.

With `batch_first=True`, an `nn.RNN` receives `(batch, time, input_size)` and returns:

```text
outputs:    (batch, time, hidden_size)
final_state: (num_layers, batch, hidden_size)
```

The leading state axis exists because a stacked RNN stores one final state per recurrent layer. It remains present even when `num_layers=1`.
"""
        ),
        md(
            """## 9.6.1 Defining the Model

An **embedding** is a trainable lookup table shaped `(vocab_size, embedding_size)`. It converts integer token IDs to dense feature vectors. An RNN then processes those vectors through time. A linear projection turns every hidden vector into `vocab_size` next-token logits.

`super().__init__()` initializes the `nn.Module` part of the object before child modules are assigned. Without it, PyTorch cannot reliably register the embedding, RNN, and projection.
"""
        ),
        code(
            """class ConciseRNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def begin_state(self, batch_size, device=None):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, token_ids, state=None):
        embedded = self.embedding(token_ids)
        recurrent_outputs, final_state = self.rnn(embedded, state)
        logits = self.output(recurrent_outputs)
        return logits, final_state"""
        ),
        code(
            """model = ConciseRNNLM(vocab_size=7, embedding_size=5, hidden_size=9, num_layers=2)
X = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
state = model.begin_state(batch_size=X.shape[0], device=X.device)
embedded = model.embedding(X)
logits, final_state = model(X, state)

print("IDs:", shape(X))
print("embedded:", shape(embedded))
print("logits:", shape(logits))
print("final state:", shape(final_state))
assert shape(embedded) == (2, 4, 5)
assert shape(logits) == (2, 4, 7)
assert shape(final_state) == (2, 2, 9)"""
        ),
        md(
            """The scratch notebook flattened logits in time-major order. This model preserves batch-first axes, so `logits.reshape(-1, vocab)` aligns directly with `Y.reshape(-1)`. Neither convention is inherently better. Mixing conventions is the bug.
"""
        ),
        md(
            """## 9.6.2 Training and Predicting

We train on the same repeating seven-token pattern. Each training call starts from zeros, so it does not carry state between epochs. The single full sequence is small enough that backpropagation through all its time steps is safe for this drill.
"""
        ),
        code(
            """corpus = torch.tensor(([0, 1, 2, 3, 4, 5, 6] * 12), dtype=torch.long)
X_train = corpus[:-1].reshape(1, -1)
Y_train = corpus[1:].reshape(1, -1)
model = ConciseRNNLM(vocab_size=7, embedding_size=8, hidden_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

losses = []
for epoch in range(60):
    optimizer.zero_grad()
    logits, _ = model(X_train)
    loss = F.cross_entropy(logits.reshape(-1, 7), Y_train.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    losses.append(loss.item())

print("first and last loss:", losses[0], losses[-1])
assert math.isfinite(losses[-1])
assert losses[-1] < losses[0]"""
        ),
        code(
            """def generate(prefix, length, model):
    model.eval()
    state = None
    output = list(prefix)
    with torch.no_grad():
        for token in prefix[:-1]:
            _, state = model(torch.tensor([[token]]), state)
        for _ in range(length):
            logits, state = model(torch.tensor([[output[-1]]]), state)
            output.append(int(logits[0, -1].argmax()))
    return output

generated = generate([0, 1], length=12, model=model)
print(generated)
assert generated[:2] == [0, 1]
assert len(generated) == 14"""
        ),
        md(
            """## 9.6.3 Inspect What the Concise Layer Owns

For each RNN layer, PyTorch registers an input-to-hidden weight, hidden-to-hidden weight, and usually two biases. Layer 0 receives embedding features. Higher layers receive hidden outputs from the layer below.
"""
        ),
        code(
            """two_layer = ConciseRNNLM(vocab_size=7, embedding_size=5, hidden_size=9, num_layers=2)
for name, parameter in two_layer.rnn.named_parameters():
    print(name, shape(parameter))

names = dict(two_layer.rnn.named_parameters())
assert shape(names["weight_ih_l0"]) == (9, 5)
assert shape(names["weight_hh_l0"]) == (9, 9)
assert shape(names["weight_ih_l1"]) == (9, 9)"""
        ),
        md(
            """## 9.6.4 Break It Deliberately: Missing the Layer Axis

For a two-layer RNN and batch size two, state must be `(2, 2, hidden_size)`. A tensor shaped only `(batch, hidden_size)` looks like the scratch state but violates the concise layer contract.
"""
        ),
        code(
            """wrong_state = torch.zeros(2, two_layer.hidden_size)
try:
    two_layer(torch.tensor([[0, 1], [2, 3]]), wrong_state)
except RuntimeError as error:
    print(type(error).__name__)
    print(str(error).splitlines()[0])
else:
    raise AssertionError("Expected a missing recurrent-layer axis to fail")"""
        ),
        checkpoint(
            "9.6",
            [
                "What does an embedding do that a raw integer token ID does not?",
                "Why does final RNN state have a num_layers axis?",
                "When can batch-first labels be flattened without transposing?",
                "What work does nn.RNN replace from the scratch implementation?",
                "Why is a framework layer still not shape magic?",
            ],
        ),
    ]
    write_nb("Chapter 9.6 - Concise Implementation of Recurrent Neural Networks.ipynb", cells)


def build_97() -> None:
    cells = [
        title_cell(
            "Chapter 9.7 - Backpropagation Through Time",
            "Backpropagation through time applies the chain rule to an unrolled recurrent computation. This notebook makes gradient paths visible, demonstrates vanishing and exploding products, and shows exactly what state detachment truncates.",
            [
                "draw an RNN as repeated uses of shared parameters",
                "explain direct and recurrent gradient paths",
                "verify a recurrent gradient with autograd",
                "predict when repeated Jacobian factors vanish or explode",
                "distinguish gradient clipping from state detachment",
            ],
        ),
        setup_cell(),
        md(
            """## 9.7.0 The Problem This Notebook Solves

The forward recurrence reuses the same parameters:

```text
h0 --[w, x1]--> h1 --[w, x2]--> h2 --[w, x3]--> h3
```

**Unrolling** draws each time step as a separate computation even though all steps share `w`. **Backpropagation through time (BPTT)** applies ordinary reverse-mode automatic differentiation to this unrolled graph.

A parameter such as `w` affects the loss through many paths: directly at the current step and indirectly through every later hidden state. The product of many derivatives can become tiny (vanishing gradient) or huge (exploding gradient).
"""
        ),
        md(
            """## 9.7.1 A Scalar Recurrence You Can Differentiate

To isolate time effects, use the linear recurrence `h_t = w * h_(t-1)` with `h_0 = 1`. After `T` steps, `h_T = w^T`, so

```text
d h_T / d w = T * w^(T - 1)
```

The factor `T` appears because the shared parameter participates at every step. Autograd should agree with the analytic derivative.
"""
        ),
        code(
            """def final_linear_state(weight, steps):
    state = torch.tensor(1.0)
    for _ in range(steps):
        state = weight * state
    return state

w = torch.tensor(1.2, requires_grad=True)
steps = 5
final_state = final_linear_state(w, steps)
final_state.backward()
analytic = steps * (w.detach() ** (steps - 1))

print("final state:", final_state.item())
print("autograd:", w.grad.item(), "analytic:", analytic.item())
assert torch.allclose(w.grad, analytic)"""
        ),
        md(
            """## 9.7.2 Vanishing and Exploding Through Repeated Products

The sensitivity of a late state to an early state contains repeated local derivatives. In the linear example, `d h_T / d h_0 = w^T`.

- If `|w| < 1`, the product shrinks exponentially.
- If `|w| > 1`, the product grows exponentially.
- Nonlinear RNNs replace the scalar factor with Jacobian matrices, but repeated multiplication creates the same core risk.
"""
        ),
        code(
            """steps = torch.arange(1, 21)
vanishing = 0.5 ** steps
stable = 1.0 ** steps
exploding = 1.5 ** steps

print("at step 20:", {
    "0.5^T": vanishing[-1].item(),
    "1.0^T": stable[-1].item(),
    "1.5^T": exploding[-1].item(),
})
assert vanishing[-1] < 1e-5
assert stable[-1] == 1
assert exploding[-1] > 1000"""
        ),
        md(
            """`tanh` also contributes derivatives. Its derivative is `1 - tanh(z)^2`, at most 1 and near 0 when `z` has large magnitude. Saturated hidden units can therefore weaken long-range gradients even if recurrent weights alone are not small.
"""
        ),
        code(
            """z = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
tanh_derivative = 1 - torch.tanh(z) ** 2
print(tanh_derivative)
assert torch.allclose(tanh_derivative[2], torch.tensor(1.0))
assert tanh_derivative[0] < 0.001
assert tanh_derivative[-1] < 0.001"""
        ),
        md(
            """## 9.7.3 Full BPTT Versus Truncated BPTT

Detaching a hidden state preserves its numeric value but removes its connection to earlier operations. The first computation below allows the final loss to send gradient through all four recurrent steps. The second detaches after two steps, so the final loss cannot update the earlier part of the graph.

Truncation reduces memory and limits gradient path length. It is an approximation: long-range credit assignment across the cut is removed.
"""
        ),
        code(
            """def two_segment_gradient(detach_between):
    weight = torch.tensor(1.2, requires_grad=True)
    state = torch.tensor(1.0)
    for _ in range(2):
        state = weight * state
    if detach_between:
        state = state.detach()
    for _ in range(2):
        state = weight * state
    state.backward()
    return state.detach(), weight.grad.detach()

full_value, full_gradient = two_segment_gradient(detach_between=False)
truncated_value, truncated_gradient = two_segment_gradient(detach_between=True)
print({"full": full_gradient.item(), "truncated": truncated_gradient.item()})
assert torch.allclose(full_value, truncated_value)
assert full_gradient > truncated_gradient
assert torch.allclose(full_gradient, torch.tensor(4 * 1.2 ** 3))
assert torch.allclose(truncated_gradient, torch.tensor(2 * 1.2 ** 3))"""
        ),
        md(
            """## 9.7.4 Shared Parameters Accumulate Gradient Contributions

When one parameter is used several times, autograd adds the gradient contribution from every path into the same `.grad` tensor. It does not create a separate parameter per time step. Calling `zero_grad()` is necessary before a new optimization step because PyTorch also accumulates gradients across separate backward calls.
"""
        ),
        code(
            """weight = torch.tensor(2.0, requires_grad=True)
state = torch.tensor(1.0)
states = []
for _ in range(3):
    state = weight * state
    states.append(state)
loss = sum(states)
loss.backward()

expected = 1 + 2 * 2.0 + 3 * (2.0 ** 2)
print("gradient from all loss paths:", weight.grad.item())
assert torch.allclose(weight.grad, torch.tensor(expected))"""
        ),
        md(
            """## 9.7.5 Break It Deliberately: Exploding Gradient

This is a controlled failure demonstration. A recurrent multiplier of 2 over 30 steps creates a derivative larger than a billion. Gradient clipping can cap the stored gradient before an optimizer step, but it does not change the unstable forward dynamics or restore information lost to vanishing gradients.
"""
        ),
        code(
            """weight = torch.tensor(2.0, requires_grad=True)
state = final_linear_state(weight, steps=30)
state.backward()
raw_gradient = weight.grad.item()

torch.nn.utils.clip_grad_norm_([weight], max_norm=1.0)
clipped_gradient = weight.grad.item()
print({"raw": raw_gradient, "clipped": clipped_gradient})
assert raw_gradient > 1e9
assert abs(clipped_gradient) <= 1.00001"""
        ),
        md(
            """## 9.7.6 Practical Interpretation

Three mechanisms are easy to confuse:

- **BPTT** computes gradients through the unrolled recurrence.
- **State detachment** chooses where that graph is cut, limiting history and memory use.
- **Gradient clipping** rescales gradients after backward computation, before the optimizer step.

None is a substitute for the others. Modern gated recurrent cells in Chapter 10 add learnable paths designed to preserve useful information more effectively, but they still use backpropagation through time.
"""
        ),
        checkpoint(
            "9.7",
            [
                "Why does unrolling not create new parameters at every time step?",
                "Where does the factor T come from in d(w^T)/dw?",
                "Why can repeated Jacobian multiplication harm long-range learning?",
                "What changes and what stays identical when state is detached?",
                "Why can clipping contain an update without fixing the recurrent dynamics?",
            ],
        ),
    ]
    write_nb("Chapter 9.7 - Backpropagation Through Time.ipynb", cells)


def main() -> None:
    build_91()
    build_92()
    build_93()
    build_94()
    build_95()
    build_96()
    build_97()
    print(f"Wrote Chapter 9 notebooks to {OUT_DIR}")


if __name__ == "__main__":
    main()
