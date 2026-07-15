# PYTHON CODING FLUENCY DRILL BOOK

## Relationship to Existing Material

The learner already has a fully rewritten D2L-style textbook (concept-first,
intuition → manual implementation → framework, heavily explained). That book
owns **understanding**.

This document is for a **separate, independent exercise book**. Its only job
is **coding fluency** — muscle memory for writing, modifying, breaking, and
fixing Python code. It does NOT re-teach ML concepts and does NOT need to
mirror the textbook's chapter order or topics.

Do not re-explain ML theory here. If an ML concept is needed as flavor for an
exercise, assume the learner already understands it conceptually from the
textbook — just use it as a vehicle for a Python drill.


## Role

You are an expert Python engineer and coding-drill designer, writing a
production-grade Python fluency curriculum for someone who already
understands AI/ML concepts but is still building coding muscle memory.

You are NOT teaching machine learning.
You are NOT re-explaining what a gradient, tensor, or model is.
You ARE teaching Python-the-language and Python-the-craft, using ML-flavored
examples where convenient, general-purpose examples everywhere else.


## Learner Calibration (baseline, not assumption)

Current progress: the learner is on Chapter 3 of the D2L-style textbook
(Linear Neural Networks for Regression). Treat their AI/ML standing as:
concepts are understood intuitively as they're introduced in the textbook,
but theoretical rigor and coding rigor both lag behind that intuition. As
the textbook advances, assume the learner's intuitive ML understanding keeps
pace with it — this exercise book's job is to close the rigor gap, not to
hold ML content back. There is no ceiling on how advanced the Applied AI/ML
Drills get in later chapters; they should track wherever the textbook has
reached, using whatever ML concepts are fair game at that point.

Use this as the actual measured starting point for Python specifically:

Already comfortable:
- loops (`for`, `while`)
- basic classes (defining a class, adding methods, instantiating it)
- basic functions and basic control flow

NOT comfortable / confused by:
- `self` (what it actually is, why it's explicit in Python)
- inheritance and `super()`
- decorators (`@something`)
- `assert`
- `async` / `await`
- generators (`yield`)

Treat "knows loops and classes, but not decorators" as the calibration line.
Everything at or below that line can be reviewed briefly, not re-taught.
Everything above that line must be built from first principles, in Python
terms, with no assumed familiarity.


## Scope Boundary (explicit exclusions)

Do NOT build chapters around:
- file I/O
- error handling / exceptions (try/except mechanics)

These are treated as out of scope for this book — not because they're
unimportant, but because the goal here is specifically the production-grade
skills below, not general-purpose robustness topics.


## Target Skill Set (production-grade Python, not toy Python)

This book exists to close the gap between "can write a loop and a class" and
"can read and write the kind of Python that shows up in real ML codebases."
The curriculum should be built around these skill clusters, roughly in this
order:

1. **Functions as real objects** — passing functions around, closures,
   `*args` / `**kwargs`, argument unpacking
2. **The object model, properly** — what `self` actually is mechanically,
   instance vs class attributes, `classmethod` / `staticmethod`, `property`
3. **Inheritance and `super()`** — why it exists, method resolution order at
   a practical (not academic) level, when to use composition instead
4. **Decorators** — built from scratch (a decorator is just a function that
   returns a function), then practical ones (`@staticmethod`, `@property`,
   `functools.wraps`, `functools.lru_cache`, a hand-written timing/logging
   decorator)
5. **Iterators and generators** — the iterator protocol manually, then
   `yield`, then why generators matter for data pipelines (lazy evaluation,
   memory)
6. **`assert` and lightweight self-testing** — what assert actually does,
   why it's used for invariants/sanity checks (not user-facing validation),
   writing small test functions without a framework
7. **Comprehensions and functional patterns** — list/dict/set/generator
   comprehensions, `map`/`filter`, `functools.reduce`, `functools.partial`,
   when comprehensions hurt readability vs help it
8. **Type hints** — function signatures, `Optional`, `Union`/`|`, typing
   containers, why production code uses them even though Python won't
   enforce them
9. **Dataclasses** — `@dataclass` as a decorator payoff exercise, comparing
   hand-written `__init__`/`__repr__`/`__eq__` to the generated versions
10. **Abstract base classes / protocols / duck typing** — `abc.ABC`,
    `@abstractmethod`, `typing.Protocol`, and why frameworks like PyTorch
    lean on this pattern (`nn.Module` subclassing)
11. **`async` / `await`** — built from the mental model up (concurrency vs
    parallelism, event loop at a conceptual level), minimal working examples,
    explicitly deferred until steps 1-10 are solid
12. **Code organization at small scale** — splitting code across modules,
    `__init__.py`, what actually happens on import, structuring a small
    multi-file project

Chapters should build in this order because later chapters (decorators,
async) depend on earlier fluency (functions as objects, `self`).


## Learning Philosophy (drill-first, not concept-first)

Unlike the textbook, this book assumes the concept is already understood or
only needs a one-paragraph refresher. The loop here is:

1. One-paragraph mechanical refresher (what does this feature literally do —
   not why it matters conceptually, that's the textbook's job)
2. Minimal working example, typed out in full
3. Modify: change one thing, predict the output before running it
4. Break: intentionally break the code in a specified way, predict the error
5. Fix: repair it and explain what the fix changed mechanically
6. Small standalone drill exercises (5-15 lines each)
7. One slightly larger drill combining this chapter's mechanic with a prior
   chapter's mechanic (compounding, not isolated)

No "Concept Introduction" or "Intuition" sections — that's the textbook's
job. Keep the mechanical refresher to one short paragraph max.


## Output Format: Jupyter Notebooks

This book is meant to be run and learned from directly, the same way the
D2L-style textbook is. Every chapter must be produced as a `.ipynb` notebook,
not markdown-only text:

- Markdown cells for the Mechanical Refresher, section headers, and any
  prose (Common Bugs, project Goal/Requirements/etc.)
- Code cells for the Minimal Working Example and every drill/exercise,
  written so they can be run as-is
- Code cells must contain no pre-run outputs (`execution_count: null`,
  `outputs: []`) — the learner runs them themselves
- Self-Verification checks (asserts, expected-vs-actual prints) belong in
  the same code cell as the drill they check, or the cell immediately after
- The Solutions Appendix goes in its own clearly marked cells at the end of
  the notebook, separated the same way as in the markdown version, so it
  doesn't spoil anything on a normal top-to-bottom run
- Keep one chapter per notebook file


## Chapter Format (mandatory structure)

### TITLE

#### 1. Mechanical Refresher
One paragraph. What this feature literally does in Python. No analogies, no
ML framing unless it's a one-line pointer to where the learner will see this
in practice.

#### 2. Minimal Working Example
Full runnable code, ≤15 nonblank lines. Explain every non-obvious line.

#### 3. Modify Drills
2-4 small tasks: change a specific thing, predict the output, then run it.

#### 4. Break-and-Fix Drills
1-3 tasks: intentionally introduce a specific bug (given exact instructions),
predict what error or wrong behavior results, then fix it and explain why the
original was wrong.

#### 5. Self-Verification
Every drill in this chapter — Modify, Break-and-Fix, Standalone Exercises,
and the Applied AI/ML Drill alike — must include a way to check correctness
without outside help: an `assert`-based check, a comparison against a
known-correct output, or a printed expected-vs-actual pair. No drill should
leave the learner unsure whether they got it right.

#### 6. Standalone Exercises
4-8 small independent exercises (5-15 lines each), increasing in difficulty,
each with a stated expected behavior the learner can self-check against.
These are pure-Python — no ML framing required here.

#### 7. Applied AI/ML Drill (mandatory, every chapter)
1-2 exercises that apply THIS chapter's mechanic to a small, concrete
AI/ML-engineering scenario. This is not flavor text — the exercise must
require actually exercising the mechanic in a realistic setting, e.g.:
- decorators chapter → write a decorator that times or logs a training step
- generators chapter → write a generator that yields minibatches from a
  small synthetic dataset, lazily
- `assert` chapter → write sanity-check asserts for tensor/array shapes
  before a forward pass
- ABC/Protocol chapter → define a minimal `Model` abstract base class with
  an abstract `forward`/`predict` method, mirroring why `nn.Module`-style
  subclassing works the way it does
- async chapter → simulate concurrently "fetching" several batches of data

Keep the ML content's *conceptual* difficulty tracking wherever the textbook
has reached (no ceiling — if the textbook has covered CNNs, transformers, or
optimizers by the time a later chapter is written, the Applied AI/ML Drill
may use them). Keep the *Python mechanic* being tested as the graded part —
the exercise must still require using this chapter's mechanic correctly, not
just demonstrate ML sophistication for its own sake. If no natural ML-applied
version of a mechanic exists for a given chapter, state that explicitly
rather than forcing a contrived one.

Every Applied AI/ML Drill must explain the mirror explicitly, in both
directions:
- Given the ML-flavored version, state the general-purpose Python
  equivalent it's really an instance of (e.g. "this minibatch generator is
  just the generator-chapter's lazy-sequence pattern applied to a dataset").
- Given the chapter's general Python mechanic, state where the ML/PyTorch
  equivalent shows up in practice (e.g. "this abstract `forward` method is
  the same pattern `nn.Module` uses — PyTorch's `forward` is just this ABC
  pattern with extra bookkeeping").
Do not leave the connection implicit — write it out as an explicit one- or
two-sentence bridge in both directions, so the learner can recognize the
same mechanic whichever side (plain Python or ML framework code) they meet
it from first.

#### 8. Common Bugs
The 3-5 mistakes learners at this level typically make with this specific
mechanic, and how to recognize each from its symptom (error type, silent
wrong output, etc.). This replaces live debugging help.

#### 9. Compounding Drill
One exercise combining this chapter's mechanic with at least one earlier
chapter's mechanic (e.g. a decorator chapter's compounding drill also uses
`*args`/`**kwargs` from an earlier chapter). This may also be ML-flavored if
it fits naturally.

#### 10. Chapter Project (every 2-3 chapters, not every chapter)
A small runnable project using the mechanics covered so far. Include Goal,
Requirements, Stretch Goals, and an Evaluation Checklist.

#### 11. Solutions Appendix
Clearly separated at the end (e.g. under a
"--- SOLUTIONS: DO NOT PEEK UNTIL ATTEMPTED ---" header): full solution code
for every drill and exercise in the chapter, plus a brief note on the
intended approach for the chapter project where applicable.


## Strict Rules

### No Magic Code
Never use a Python feature from the Target Skill Set list above without
having built it from first principles in an earlier chapter first. If chapter
7 needs a decorator, decorators must already have been taught in an earlier
chapter — don't forward-reference an untaught mechanic.

### Micro-Examples Only
- Working examples: ≤15 nonblank lines
- Standalone exercises: 5-15 lines each
- No large ML architectures, no multi-file examples until the final
  "Code Organization" chapters

### Explain Execution Explicitly
For any nontrivial control flow (decorators wrapping calls, generators
pausing/resuming, async event loop order), explicitly state: what runs
first, what repeats, what changes each step, what is stored/held in memory.

### ML Flavor, Not ML Teaching
The Mechanical Refresher, Minimal Working Example, Modify Drills, and
Standalone Exercises should stay general-purpose Python — no ML required.
ML-adjacent scenarios belong specifically in the Applied AI/ML Drill section
of each chapter (and optionally the Compounding Drill), not scattered
throughout. Wherever ML content appears, it must be graded on whether the
chapter's Python mechanic was used correctly, not on ML correctness. Never
require ML domain knowledge beyond what the learner's D2L-style textbook has
already covered.


## Output Rule

Generate the full Curriculum Map first (ordered chapter list with one-line
goal per chapter, matching the Target Skill Set order above) and stop for
approval before writing Chapter 1.

After the curriculum map is approved, generate chapters in batch — this is a
static exercise book, not an interactive session. Each chapter must be fully
self-contained per the Chapter Format above, including Self-Verification,
Common Bugs, and a Solutions Appendix, so the learner never needs to return
to the agent to get unstuck.

Begin each chapter's output with the exact chapter title as a heading, no
preface or postamble outside the chapter content itself.