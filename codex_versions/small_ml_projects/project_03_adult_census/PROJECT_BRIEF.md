# Project 3: Adult Census Messy-Data Classification

## Project status

- Environment: Google Colab
- Expected scale: 3-5 focused sessions
- Deliverable: one Colab notebook exported as `.ipynb`
- Difficulty: guided specification, independent implementation
- Primary task: cleaning and evaluating mixed tabular data

## Scenario

You receive census-derived records and are asked to predict whether a person's
reported annual income falls above `$50K`. Unlike earlier projects, the central
challenge is not selecting a sophisticated model. It is discovering and
documenting how raw representations, missing markers, redundant information,
duplicates, category inconsistencies, and demographic variables affect a
machine-learning workflow.

This is a historical benchmark dataset, not a suitable basis for making
high-stakes decisions about individuals. Treat the prediction task as a data
cleaning and evaluation exercise.

## Dataset

Use the UCI Adult dataset:

- Dataset page: <https://archive.ics.uci.edu/dataset/2/adult>
- Citation: Becker and Kohavi (1996), UCI Machine Learning Repository
- DOI: <https://doi.org/10.24432/C5XW20>
- License: CC BY 4.0

The combined dataset has approximately 48,842 observations and 14 input
features. It contains numerical and categorical data and documented missing
values.

Use UCI's `ucimlrepo` loader so that you begin from one dataframe while still
auditing the returned values yourself:

```python
!pip -q install ucimlrepo

from ucimlrepo import fetch_ucirepo

adult = fetch_ucirepo(id=2)
X_raw = adult.data.features.copy()
y_raw = adult.data.targets.copy()
```

Do not immediately clean or reshape the returned objects. Preserve `X_raw` and
`y_raw` unchanged so you can compare raw and cleaned representations later.

## Prediction-time and use contract

Assume the model receives one completed census-style record and estimates the
dataset's binary income label. All included fields are treated as available at
that time.

The model must not be presented as suitable for lending, employment,
immigration, policing, benefits, or other consequential decisions. Historical
patterns can reflect structural inequity and measurement choices. Predictive
performance does not establish that a feature is fair, causal, or appropriate
for a real decision system.

## Learning objectives

By the end, you should be able to:

- preserve raw data while constructing a documented clean representation;
- discover missing values encoded as strings;
- normalize whitespace and inconsistent label formatting;
- reason about duplicate rows rather than deleting them automatically;
- distinguish missingness, unknown categories, and legitimate categories;
- handle mixed data with train-fitted pipelines;
- compare a minimal cleaning policy with a more considered policy;
- evaluate an imbalanced classifier overall and across selected groups;
- communicate ethical and statistical limitations without overclaiming.

## Rules and scope

You must:

- keep immutable-style raw copies for audit comparisons;
- record every cleaning rule and the number of affected cells or rows;
- inspect raw string representations before replacing values;
- normalize the binary target explicitly and verify exactly two classes remain;
- split before fitting imputers, encoders, scalers, or models;
- use stratification and a fixed random seed;
- include a dummy baseline and logistic regression;
- use pipelines for learned preprocessing;
- compare at least two documented data-treatment variants;
- evaluate overall metrics and limited group slices;
- maintain a cleaning log, experiment log, and final reflection.

You may:

- use pandas, NumPy, Matplotlib, Seaborn, and scikit-learn;
- use simple deterministic string cleanup before splitting when it does not
  learn population statistics;
- treat documented missing markers with an imputer fit only on training data;
- drop a redundant feature if you explain and test the decision;
- add one optional model after the required work.

Do not:

- copy an existing Adult dataset solution notebook;
- silently call `dropna()`, `drop_duplicates()`, or blanket string replacements;
- fit preprocessing on the full dataset;
- remove demographic groups merely to avoid examining them;
- claim group metric parity establishes fairness;
- use this model to recommend decisions about people;
- use neural networks, broad tuning, deployment, Docker, or MLflow.

## Notebook structure and exact checkpoints

Use the numbered headings below.

### 0. Project header and reproducibility

Include:

- title, your name, and date;
- fixed random seed;
- dataset source and citation;
- prediction-time contract;
- prohibited-use statement;
- imports.

Checkpoint:

- What does one row represent?
- What does the target claim to measure?
- Why is this project primarily about data handling rather than deployment?

### 1. Load and preserve raw objects

Load `X_raw` and `y_raw` and inspect:

- object types and shapes;
- column names and dtypes;
- first and last rows;
- raw target shape and unique values;
- raw representations of selected categorical values using `repr()` where
  whitespace may be invisible.

Then create separate working copies. Never overwrite the raw variables.

Required checks:

```python
assert len(X_raw) == len(y_raw)
assert X_raw is not X_work
assert y_raw is not y_work
```

Checkpoint:

- What is the raw shape of the target?
- Are logically identical labels represented by different strings?
- Why is preserving raw data useful when a later cleaning rule looks wrong?

### 2. Build a raw data-quality report

Before cleaning, report:

- standard null counts;
- counts of exact and whitespace-padded `?`-style markers;
- leading/trailing whitespace in categorical values;
- unique target strings using visible representations;
- duplicate full rows and duplicate feature rows;
- category counts for every categorical column;
- numerical summaries and implausible or sentinel-like values;
- target counts and proportions.

Do not modify anything in this section.

Required visualizations:

1. raw target-class proportions after grouping logically equivalent labels only
   for display;
2. missing-marker counts by column;
3. distribution of one numerical variable;
4. target rate across one categorical variable.

Checkpoint:

- Which columns contain documented missing values?
- Are those values recognized by pandas as `NaN`, stored as strings, or both?
- What invisible formatting differences did you discover?
- Why might two identical feature rows legitimately represent different people?

### 3. Write a cleaning contract

Before applying transformations, create a markdown table:

| Issue | Detection rule | Planned action | Why | Learned from data? |
|---|---|---|---|---|
| Whitespace | String differs from stripped form | Strip categorical strings | Normalize representation | No |
| Missing marker | Value equals documented marker after stripping | Convert to missing | Enable explicit handling | No |
| Target punctuation | Logically identical label has trailing punctuation | Map to canonical class | Two semantic classes | No |

Add rows for every additional action. Distinguish deterministic normalization
from learned preprocessing. Learned preprocessing belongs inside a pipeline and
must be fit on training data only.

Checkpoint:

- Which rules are justified by documentation or representation alone?
- Which decisions require estimating a mode, median, scale, or category set?
- Why does that distinction matter for leakage?

### 4. Apply deterministic normalization

Create cleaned working objects by applying only your deterministic rules:

- trim categorical whitespace;
- turn documented missing-marker strings into real missing values;
- canonicalize the two target labels;
- reshape the target intentionally if necessary.

Create a cleaning log with counts before and after each rule. Confirm that raw
objects did not change.

Required checks:

```python
assert y_clean.ndim == 1
assert y_clean.nunique(dropna=False) == 2
assert len(X_clean) == len(y_clean)
assert X_raw.shape == X_clean.shape
```

The last assertion applies unless you have an unusually well-justified
deterministic row-removal rule. Row removal is not expected here.

Checkpoint:

- How many cells became missing values?
- How many target representations collapsed into each canonical class?
- Did stripping strings merge any categories?
- Can you reproduce every change from the cleaning contract?

### 5. Investigate duplicates and redundancy

Analyze, but do not automatically remove:

- exact duplicate feature-and-target records;
- identical feature rows with conflicting targets;
- the relationship between `education` and `education-num`;
- the meaning and limitations of `fnlwgt`;
- high-cardinality or rare categories.

Choose and document a duplicate policy. Keeping duplicates is acceptable if you
explain that the dataset lacks a person identifier and repeated feature values
need not represent accidental duplicate records. Removing exact duplicates is
also acceptable as an experiment, but not as an unquestioned cleaning ritual.

Checkpoint:

- Can duplicate rows be proven to be the same person?
- Does `education-num` duplicate all information in `education`, or merely encode
  a strong relationship?
- Could random splitting place identical records on both sides?
- What experiment could measure sensitivity to your duplicate policy?

### 6. Define two data-treatment variants

Create two documented variants while preserving a common target and split.

#### Variant A: minimal treatment

- deterministic string and label normalization;
- missing categorical values handled by a simple train-fitted policy;
- all otherwise usable features retained.

#### Variant B: considered treatment

Choose a small number of justified changes, such as:

- represent missing categorical values as an explicit category rather than mode
  imputation;
- remove one of the strongly redundant education features;
- alter the treatment of exact duplicates;
- treat a numerical code differently based on meaning.

Do not bundle many changes together. The comparison should remain interpretable.

Checkpoint:

- What exact difference separates A from B?
- What outcome do you predict before training?
- Does the variant alter rows, columns, values, or preprocessing behavior?

### 7. Make the train/test split

Create one 80/20 stratified split using cleaned row indices and the fixed seed.
Reuse those exact indices for both variants so data treatment—not sampling—is
the intended difference.

Display:

- all shapes;
- target proportions overall, in training, and in test;
- index non-overlap;
- missing counts on each side before imputation.

Checkpoint:

- Why must both variants share the same split?
- Why should imputation occur after the split?
- If duplicate rows remain across train and test, what limitation does that
  introduce?

### 8. Establish the dummy baseline

Train a majority-class `DummyClassifier` on the training target and evaluate it
on the held-out target.

Report:

- accuracy;
- positive-class precision;
- positive-class recall;
- positive-class F1;
- confusion matrix.

Checkpoint:

- How does class imbalance affect the baseline accuracy?
- Which metric most clearly shows that the model does not identify the positive
  class?
- Why must every later model be compared with this baseline?

### 9. Build train-fitted preprocessing pipelines

For each treatment variant, build a scikit-learn pipeline that:

- imputes missing numerical or categorical values according to the variant;
- encodes categorical columns while tolerating unseen categories;
- scales numerical data when appropriate;
- trains logistic regression;
- learns all statistical state from training rows only.

Keep model hyperparameters fixed between variants unless the data representation
requires a documented adjustment.

Checkpoint:

- Which earlier cleaning was deterministic, and which pipeline transformations
  learn state?
- Where are imputation values learned?
- What happens to a category appearing only in the test set?
- Why would calling `get_dummies()` on the full dataset contaminate the workflow?

### 10. Evaluate and compare variants

For both variants, report:

- accuracy;
- positive-class precision;
- positive-class recall;
- positive-class F1;
- ROC-AUC from probabilities;
- confusion matrix;
- predicted-positive count.

Create one comparison table and use the same metric definitions for every run.

Checkpoint:

- Did either variant materially outperform the other?
- Could a very small difference be sampling noise?
- Did the considered treatment improve data defensibility even if metrics barely
  changed?
- Which variant would you carry into further investigation, and why?

### 11. Limited group-slice evaluation

Using the chosen valid model's fixed test predictions, report metrics separately
for a small number of documented groups from `sex` and/or `race`. At minimum,
include:

- sample count;
- actual positive prevalence;
- predicted-positive rate;
- accuracy;
- precision and recall where mathematically defined;
- false-positive and false-negative counts.

Handle zero denominators transparently. Do not hide undefined metrics by
silently replacing them with a persuasive-looking number.

Checkpoint:

- Do base rates differ among the inspected groups?
- Do error rates differ?
- Are any groups too small for stable conclusions?
- Why does reporting group metrics not prove the model is fair or unfair by
  itself?
- What historical and measurement context is missing from this dataset?

### 12. Error analysis

Create false-positive and false-negative tables joined to cleaned test features.
Compare a few feature or category distributions among errors and correct
predictions. Include group sizes.

Write observations without causal language. For example, prefer:

```text
False negatives were more frequent in this observed subgroup.
```

Do not write:

```text
This characteristic causes the model to underestimate income.
```

Checkpoint:

- Which error type is more common?
- What patterns are visible?
- Could class prevalence, sample size, or correlated features explain them?
- What cannot be learned from this observational benchmark?

### 13. Sensitivity check

Choose exactly one cleaning decision and test its effect while keeping the split
and model fixed. Suitable choices include:

- keep versus remove exact duplicates;
- retain both education representations versus drop one;
- explicit missing category versus most-frequent imputation;
- include versus exclude `fnlwgt` as an ordinary predictor.

Predict the outcome before running the comparison. Report both metric changes
and row/feature-count changes.

Checkpoint:

- Was the model sensitive to this decision?
- Did the change affect validity, interpretability, or only predictive metrics?
- Would you need repeated splits or cross-validation before trusting a small
  difference?

### 14. Logs

Maintain a cleaning log:

| Step | Rule | Cells affected | Rows affected | Shape after | Reason |
|---|---|---:|---:|---|---|
| 0 | Raw copy | 0 | 0 | | Preserve source |
| 1 | Strip categorical strings | | | | Normalize representation |
| 2 | Convert documented marker to missing | | | | Explicit missingness |
| 3 | Canonicalize target | | | | Two semantic labels |

Maintain an experiment log:

| Run | Treatment | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | Clean target only | Dummy majority | | | | | N/A | Baseline |
| 1 | Variant A | Logistic regression | | | | | | Minimal treatment |
| 2 | Variant B | Logistic regression | | | | | | Considered treatment |
| 3 | Sensitivity variant | Logistic regression | | | | | | One isolated decision |

### 15. Final report

Write a concise report containing:

1. **Problem and use limits:** prediction target, available information, and
   prohibited high-stakes interpretation.
2. **Raw data findings:** shapes, hidden markers, whitespace, target formats,
   duplicates, and redundancy.
3. **Cleaning contract:** deterministic versus learned transformations.
4. **Method:** common split, baseline, pipelines, and treatment variants.
5. **Results:** comparison using suitable classification metrics.
6. **Sensitivity:** effect of one cleaning decision.
7. **Group slices:** what was observed and why it is not a complete fairness
   assessment.
8. **Error analysis:** one careful, non-causal observation.
9. **Limitations:** at least five statistical, historical, measurement, or use
   limitations.
10. **What I now understand:** five concrete statements about messy-data work.

## Minimum completion rubric

| Area | Completion standard |
|---|---|
| Raw preservation | Raw features and target remain available and unchanged for comparison. |
| Audit | Hidden markers, whitespace, labels, duplicates, dtypes, categories, and balance are inspected before modification. |
| Cleaning contract | Every rule has a detection method, action, count, and justification. |
| Target | Exactly two canonical labels remain and target shape is explicit. |
| Duplicate policy | Duplicates are reasoned about rather than reflexively deleted. |
| Variants | Two interpretable data-treatment policies share the same split. |
| Leakage control | Imputation, encoding, and scaling learn state from training data only. |
| Baseline | Majority dummy performance is measured and interpreted. |
| Modeling | Comparable logistic-regression pipelines evaluate both variants. |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, and confusion matrices are reported. |
| Slices | Selected group metrics include counts and cautious interpretation. |
| Sensitivity | One cleaning decision is isolated and measured. |
| Reproducibility | Seed, source, rules, split, cleaning log, and run log are recorded. |
| Communication | Final report states ethical and statistical limitations without causal claims. |

## Hint ladder

When blocked, state the section number and provide attempted code, the complete
error or surprising output, observed shapes/dtypes, and expected behavior.
Assistance should progress from:

1. conceptual hint;
2. relevant pandas or scikit-learn mechanism;
3. pseudocode or unrelated miniature example;
4. partial project code;
5. complete local solution only if earlier levels fail.

## Optional extension (choose at most one)

- use cross-validation on training data to assess whether variant differences
  are stable;
- compare logistic regression with one tree-based model;
- evaluate the effect of sample weighting using `fnlwgt`, with careful research
  into what that weight represents;
- create a model card-style summary focused on intended use, prohibited use,
  data limitations, and evaluation gaps.

Do not perform all four. The required project already contains substantial
analysis.
