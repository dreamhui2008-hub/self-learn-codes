# Project 1: Bank Marketing Classification

## Project status

- Environment: Google Colab
- Expected scale: 2-4 focused sessions
- Deliverable: one Colab notebook exported as `.ipynb`
- Difficulty: guided specification, independent implementation
- Primary task: binary tabular classification

## Scenario

A bank has run telephone marketing campaigns for term deposits. It wants a
model that estimates whether a contacted client will subscribe to a deposit.

Your task is to build and evaluate a defensible baseline classification
workflow. Your conclusion must explain what the model can and cannot claim. A
high score alone is not a completed project.

## Dataset

Use the UCI Bank Marketing dataset:

- Dataset page: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
- Citation: Moro, Rita, and Cortez (2014), UCI Machine Learning Repository
- DOI: <https://doi.org/10.24432/C5K306>
- License: CC BY 4.0

Use the `bank-full.csv` version: 45,211 observations, 16 input features, and the
binary target `y`. It is contained in the `bank.zip` download on the UCI page.

You may load it either by downloading `bank.zip` from UCI and uploading the CSV
to Colab, or through UCI's `ucimlrepo` package. Using `ucimlrepo` is convenient,
but you must still inspect the returned objects and confirm their shapes.

If using the package, the only setup code supplied by this brief is:

```python
!pip -q install ucimlrepo

from ucimlrepo import fetch_ucirepo

bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features.copy()
y = bank_marketing.data.targets.copy()
```

Do not assume that `y` has the one-dimensional shape expected by every
scikit-learn estimator. Inspect it and make an explicit decision if reshaping
is necessary.

## The prediction-time contract

Before modeling, write one paragraph answering this question:

> At what exact moment is this prediction supposed to be made?

For the main experiment, assume the bank wants a prediction **before the phone
call has finished**. Therefore, the feature `duration` is forbidden in the main
model: its value is only known after the call, and a zero-duration call cannot
produce a successful subscription. Including it would give the model
information unavailable at the stated prediction time.

Later, you will run one deliberately leaky comparison that includes `duration`.
Its purpose is to measure how misleading leakage can be, not to select a final
model.

## Learning objectives

By the end, you should be able to:

- inspect an unfamiliar tabular dataset systematically;
- distinguish features, target, identifiers, and prediction-time leakage;
- split data before fitting preprocessing;
- preprocess numerical and categorical features without contaminating the test
  set;
- establish a naive baseline before interpreting a trained model;
- evaluate an imbalanced binary classifier using more than accuracy;
- compare a valid workflow with a deliberately invalid one;
- describe errors and limitations in plain language.

## Rules and scope

You must:

- use a fixed random seed and record it near the top of the notebook;
- reserve a test set that is used only for final evaluation;
- use a stratified split;
- fit all learned preprocessing on training data only;
- include a naive baseline and logistic regression;
- exclude `duration` from the main model;
- report the required metrics and plots;
- keep a short experiment log;
- finish with the required reflection.

You may:

- use pandas, NumPy, Matplotlib, Seaborn, and scikit-learn;
- use `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, and standard imputers;
- add one optional model after the required baseline and logistic regression;
- consult API documentation and ask for hints.

Do not:

- copy an existing Bank Marketing solution notebook;
- perform broad hyperparameter searches;
- use neural networks;
- use MLflow, Docker, deployment, or cloud services beyond Colab;
- optimize repeatedly against the test set;
- claim that association implies causation;
- keep `duration` in the final deployable feature set.

## Notebook structure and exact checkpoints

Use the numbered headings below in your notebook. Do not move forward from a
checkpoint until you can answer its questions.

### 0. Project header and reproducibility

Include:

- project title;
- your name and date;
- prediction-time contract;
- fixed random seed;
- library imports;
- dataset citation and URL.

Checkpoint:

- What is one row intended to represent?
- What is the target?
- What event would happen after a positive prediction?

### 1. Load and establish data shape

Load `X` and `y`, then inspect:

- Python object types;
- row and column counts;
- column names;
- first few rows;
- target shape and unique values.

Required assertion:

```python
assert len(X) == len(y)
```

Checkpoint:

- Write the observed shape of `X`.
- Write the observed shape of `y` before and after any intentional reshaping.
- Confirm that the target contains exactly two semantic classes.

### 2. Data audit

Create a compact audit that examines:

- pandas dtypes;
- missing values, including whether strings such as `unknown` encode missing or
  unavailable information;
- duplicate rows;
- numerical summaries;
- unique category counts;
- suspicious sentinel values, especially `pdays == -1`;
- target count and target proportion.

Do not automatically delete every duplicate or replace every `unknown`. First
decide what each value might mean in this dataset.

Required visualizations:

1. target-class counts or proportions;
2. subscription rate across one categorical feature;
3. distribution of one numerical feature, separated by target where useful.

Checkpoint:

- Is the target balanced enough for accuracy to be meaningful by itself?
- Which columns are numerical in storage but might represent categories or
  special codes?
- What does `pdays == -1` appear to mean?
- Are duplicate records necessarily erroneous here? Explain your decision.

### 3. Define the valid feature set

Create the main feature table without `duration`. Separate its columns into
numerical and categorical lists, and verify that every feature appears in
exactly one list.

Required checks:

```python
assert "duration" not in X_main.columns
assert set(numeric_features).isdisjoint(categorical_features)
assert set(numeric_features) | set(categorical_features) == set(X_main.columns)
```

Checkpoint:

- Why is `duration` leakage under this project's prediction-time contract?
- Which other columns might become unavailable under a different prediction
  moment?
- Why should column roles be chosen by meaning rather than dtype alone?

### 4. Make the train/test split

Create one stratified train/test split. A 20% test fraction is appropriate.
Use the fixed seed declared in Section 0.

After splitting, display:

- all four shapes;
- positive-class proportion in the full target, training target, and test
  target;
- confirmation that row indices do not overlap.

Checkpoint:

- Why stratify?
- Why must the test set remain untouched during model and threshold decisions?
- What information is allowed to be learned from the training set?

### 5. Establish a naive baseline

Train a `DummyClassifier` that always predicts the most frequent class. Evaluate
it on the held-out test set.

Record:

- accuracy;
- precision;
- recall;
- F1 score;
- confusion matrix.

Checkpoint:

- Why can this baseline achieve apparently respectable accuracy?
- What does it do for the positive class?
- Which metric exposes its main weakness most directly?

### 6. Build the preprocessing and logistic-regression pipeline

Build a scikit-learn workflow that:

- transforms numerical and categorical columns appropriately;
- handles any missing-value policy you chose;
- one-hot encodes categorical features while tolerating unseen categories;
- trains logistic regression;
- prevents learned preprocessing from being fit on test data.

Use standard library components rather than manually creating dummy columns
before the split. Keep the first model simple. If convergence warnings occur,
investigate scaling and/or the iteration limit instead of suppressing warnings.

Checkpoint:

- Which pipeline steps learn state during `.fit()`?
- On which rows is that state learned?
- What happens if a category exists in the test set but not the training set?
- Why is a pipeline safer than preprocessing the whole dataframe first?

### 7. Evaluate the main model

Evaluate logistic regression on the untouched test set.

Required outputs:

- accuracy;
- precision for the positive class;
- recall for the positive class;
- F1 score for the positive class;
- ROC-AUC using predicted probabilities, not hard labels;
- confusion matrix with readable class labels;
- one threshold-independent curve: ROC or precision-recall.

Also show the number of true negatives, false positives, false negatives, and
true positives, then explain each in the bank's context.

Checkpoint:

- Did the model improve over the dummy baseline in a meaningful way?
- Which error is more costly: calling someone unlikely to subscribe, or failing
  to call someone who would? State an assumption rather than claiming a universal
  answer.
- Would you choose a model using accuracy alone? Why or why not?
- What does ROC-AUC mean here without using the phrase "the model is X% accurate"?

### 8. Threshold experiment

The default decision threshold is commonly 0.5. Compare it with at least two
other thresholds using the same trained model. Do not retrain for each
threshold.

Create a small table containing, for every threshold:

- threshold;
- predicted-positive count;
- precision;
- recall;
- F1;
- false positives;
- false negatives.

Checkpoint:

- What changes mechanically when the threshold changes?
- Which threshold would you recommend under your stated business-cost
  assumption?
- Why is test-set threshold selection methodologically questionable?
- In a larger project, what additional split would you introduce?

This is a learning exercise, so you may display test-set threshold comparisons.
Explicitly acknowledge that a real threshold would be selected using validation
data and evaluated once on the test set.

### 9. Deliberately leaky variant

Build one comparison pipeline using the same split logic but including
`duration`. Keep other major choices as similar as possible.

Compare the valid and leaky models in one table. Include at least ROC-AUC, recall,
F1, and a short statement about deployability.

Checkpoint:

- Did the leaky model appear stronger?
- Why does better offline performance not make it the correct model?
- Give one example outside this dataset of a feature that could reveal the
  outcome because it is recorded too late.

### 10. Error analysis

Create dataframes for false positives and false negatives by joining predictions
back to the original test rows. Inspect a small sample and compare at least two
feature distributions or category rates between:

- correctly predicted positives and false negatives; or
- correctly predicted negatives and false positives.

Do not invent a causal story. Describe patterns as observations and propose what
additional information would be required to test an explanation.

Checkpoint:

- Is there an obvious subgroup where the model struggles?
- Could the pattern be caused by sample size?
- What new data might help distinguish competing explanations?

### 11. Experiment log

Maintain a compact table like this:

| Run | Feature policy | Model | Key change | Accuracy | Precision | Recall | F1 | ROC-AUC | Valid at prediction time? |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | Main | Dummy | Majority baseline | | | | | N/A | Yes |
| 1 | Main | Logistic regression | Initial valid pipeline | | | | | | Yes |
| 2 | Main + duration | Logistic regression | Deliberately leaky | | | | | | No |

If you add an optional model, add exactly one further row and justify why it was
worth testing.

### 12. Final report

Write a concise report in your own words containing:

1. **Problem:** prediction target and prediction moment.
2. **Data:** size, feature types, target balance, and important quality findings.
3. **Method:** split, preprocessing, baseline, and main model.
4. **Results:** comparison against baseline with the metrics that matter.
5. **Leakage finding:** what happened when `duration` was included and why that
   result is invalid for the main scenario.
6. **Error analysis:** one observed pattern, carefully stated.
7. **Limitations:** at least three limitations.
8. **Recommendation:** whether the current work is good enough for further
   investigation—not whether it should immediately be deployed.
9. **What I now understand:** five concrete statements about skills or concepts
   that became clearer.

## Minimum completion rubric

The project is complete only when all required rows below are satisfied.

| Area | Completion standard |
|---|---|
| Problem framing | Prediction moment is explicit and determines feature availability. |
| Data understanding | Shapes, dtypes, categories, sentinels, duplicates, and target balance are inspected. |
| Leakage control | `duration` is absent from the main model and its exclusion is explained. |
| Splitting | Test set is stratified, non-overlapping, and held out before learned preprocessing. |
| Baseline | Majority-class dummy metrics are reported and interpreted. |
| Pipeline | Categorical and numerical processing occur inside a fitted workflow. |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, and confusion matrix are correctly produced. |
| Thresholds | At least three thresholds are compared and validation-set caveat is stated. |
| Weak variant | The deliberately leaky model is measured but rejected as non-deployable. |
| Error analysis | False positives or false negatives are inspected without unsupported causal claims. |
| Reproducibility | Seed, imports, data source, decisions, and run table are recorded. |
| Communication | Final report explains results, limitations, and learning in plain language. |

## Hint ladder

When blocked, state the section number and show:

- the code you attempted;
- the complete error message or unexpected output;
- what you think should have happened.

Help should progress through these levels:

1. conceptual hint;
2. name of the relevant pandas or scikit-learn mechanism;
3. pseudocode or a tiny unrelated example;
4. partial project code with blanks;
5. complete local solution only when earlier levels are insufficient.

An error is not a failed project. Copying a completed Bank Marketing notebook
would defeat the purpose of the project.

## Optional extension (choose at most one)

Only after completing every required section, choose one:

- compare logistic regression with one tree-based model;
- use a validation split to select the decision threshold properly;
- inspect coefficient direction and magnitude carefully after preprocessing;
- compare a random split with a time-respecting split using the available
  campaign time fields, while documenting the limitations of that approximation.

Do not perform all four. The stopping skill is part of the exercise.
