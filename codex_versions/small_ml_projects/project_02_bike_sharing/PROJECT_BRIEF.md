# Project 2: Bike Sharing Demand Regression

## Project status

- Environment: Google Colab
- Expected scale: 2-4 focused sessions
- Deliverable: one Colab notebook exported as `.ipynb`
- Difficulty: guided specification, independent implementation
- Primary task: tabular regression with time-aware evaluation

## Scenario

A bike-sharing operator wants to estimate hourly rental demand so it can plan
bike availability and staffing. Your task is to build a regression workflow
that predicts the total number of rentals in an hour from information that
would be available before that hour begins.

This project adds three ideas to the classification workflow from Project 1:

- regression metrics and residuals;
- target leakage through arithmetic relationships;
- the difference between random and chronological evaluation.

## Dataset

Use the UCI Bike Sharing dataset:

- Dataset page: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>
- Citation: Fanaee-T (2013), UCI Machine Learning Repository
- DOI: <https://doi.org/10.24432/C5W894>
- License: CC BY 4.0

Download the archive from UCI and use `hour.csv`, which contains 17,389 hourly
observations. Keep a copy of the accompanying `Readme.txt` available while you
work. You may upload `hour.csv` to Colab manually or place it in Google Drive.

The CSV contains semicolon-separated fields. Confirm the delimiter rather than
assuming the pandas default.

The target is `cnt`, the total rental count. Two columns require immediate
attention:

```text
cnt = casual + registered
```

Therefore, `casual` and `registered` are forbidden predictors. They reveal the
target exactly and must be removed before any model is trained.

## Prediction-time contract

For the main experiment, assume a prediction is made immediately before an hour
begins. Calendar information and a weather forecast are considered available.
Actual rental counts during that hour are not available.

Write down the limitations of treating recorded weather observations as if they
were forecasts. This is an educational approximation, not proof that the same
performance would occur in a live forecasting system.

## Learning objectives

By the end, you should be able to:

- frame and evaluate a regression problem;
- identify exact target leakage from feature definitions;
- parse and derive useful information from a date column;
- build a naive regression baseline;
- compare random and chronological splits;
- interpret MAE, RMSE, R-squared, and residuals;
- distinguish interpolation from forecasting into a later period;
- explain why an impressive random-split score can be misleading.

## Rules and scope

You must:

- record a fixed random seed;
- use `hour.csv` and predict `cnt`;
- remove `casual`, `registered`, and the row identifier `instant`;
- verify the arithmetic leakage relationship before removing its columns;
- build a naive baseline and a linear regression model;
- run both a random-split experiment and a chronological-split experiment;
- perform preprocessing inside a scikit-learn pipeline;
- report MAE, RMSE, and R-squared;
- inspect residuals and time-related error patterns;
- maintain an experiment log and final reflection.

You may:

- use pandas, NumPy, Matplotlib, Seaborn, and scikit-learn;
- derive calendar features from `dteday`;
- add one optional nonlinear model after the required experiments;
- transform the target only as an optional extension with careful interpretation.

Do not:

- copy an existing Bike Sharing solution notebook;
- use `casual` or `registered` as predictors;
- shuffle data for the chronological experiment;
- use future target values as lag features in this introductory project;
- use neural networks or broad hyperparameter searches;
- use MLflow, Docker, deployment, or cloud infrastructure;
- repeatedly tune against the final chronological test period.

## Notebook structure and exact checkpoints

Use the numbered headings below in your notebook.

### 0. Project header and reproducibility

Include:

- project title, your name, and date;
- fixed seed;
- prediction-time contract;
- imports;
- dataset citation and URL.

Checkpoint:

- What does one row represent?
- What quantity is being predicted, and in what units?
- At what moment is the prediction assumed to be made?

### 1. Load and establish data shape

Load `hour.csv` and inspect:

- object type and shape;
- column names and first rows;
- dtypes;
- target summary;
- dataset date range;
- whether timestamps are already in chronological order.

Required assertions after parsing the date:

```python
assert "cnt" in df.columns
assert df["dteday"].notna().all()
assert df["cnt"].ge(0).all()
```

Checkpoint:

- How many hourly observations are present?
- Does every calendar day contain exactly 24 rows? If not, do not silently
  manufacture rows; describe what you observe.
- Which columns are coded as integers even though they represent categories?

### 2. Data audit and visualization

Audit:

- missing values;
- duplicate rows;
- unique values for coded categorical columns;
- numerical summaries and suspicious values;
- target distribution and skew;
- demand across hour, weekday, month, and weather situation;
- the relationship among `cnt`, `casual`, and `registered`.

Required visualizations:

1. distribution of hourly `cnt`;
2. average or median demand by hour of day;
3. demand over a sensible chronological slice;
4. demand grouped by one weather or calendar category.

Required leakage check: calculate whether `cnt == casual + registered` for every
row and report the result.

Checkpoint:

- Is the target symmetric or skewed?
- Are integer category codes quantities with meaningful numerical distance?
- Why would using `casual` and `registered` make the task meaningless?
- Which patterns suggest that time affects demand?

### 3. Define features and roles

Remove:

- `cnt` from the feature table because it is the target;
- `casual` and `registered` because they reveal the target;
- `instant` because it is a row identifier.

Parse `dteday`. Either derive documented calendar features and then remove the
raw date, or explain another pipeline-safe representation. Do not pass raw date
strings directly to a standard estimator.

Create explicit lists of:

- numerical continuous features;
- categorical or binary features;
- derived date features, if any;
- forbidden or removed columns.

Required checks:

```python
for forbidden in ["cnt", "casual", "registered", "instant"]:
    assert forbidden not in X.columns

assert set(numeric_features).isdisjoint(categorical_features)
assert set(numeric_features) | set(categorical_features) == set(X.columns)
```

The final equality assumes the raw date has already been removed or converted.

Checkpoint:

- Why is `hr` categorical or cyclical rather than an ordinary continuous
  quantity?
- Why might treating month `12` as far from month `1` be undesirable?
- What information does `yr` encode, and could a model misuse it as a shortcut?

### 4. Create two evaluation designs

Build and clearly name two splits.

#### Experiment A: random split

- Use an 80/20 split with the fixed seed.
- This is a conventional tabular comparison, not the final forecasting claim.

#### Experiment B: chronological split

- Sort by date and hour first.
- Use the earliest 80% for training and latest 20% for testing.
- Do not shuffle.
- Report the boundary timestamp and date ranges on both sides.

For both experiments, display all train/test shapes and confirm index
non-overlap.

Checkpoint:

- In the random split, can later observations help train a model evaluated on
  earlier observations?
- Which split better resembles predicting future demand?
- Why might the chronological experiment be harder?
- Is a single chronological holdout enough for a production forecasting claim?

### 5. Establish naive baselines

Use a `DummyRegressor` with a constant strategy such as the training-target
mean. Fit a separate baseline for each split.

Report:

- MAE;
- RMSE;
- R-squared;
- the training-target constant used for prediction.

Checkpoint:

- What does the MAE mean in approximate bikes per hour?
- Why is RMSE usually larger than MAE?
- What does an R-squared value near zero mean relative to this baseline?
- Can R-squared be negative on test data? Explain conceptually.

### 6. Build the linear-model pipeline

Build a scikit-learn pipeline that:

- processes numeric and categorical columns appropriately;
- fits learned preprocessing only on training rows;
- one-hot encodes category codes rather than assuming all codes are continuous;
- trains a regularized linear regression model such as `Ridge`.

Use the same feature logic and major model settings in both split experiments so
the evaluation design is the main difference.

Checkpoint:

- Which preprocessing steps learn state?
- Why can one-hot encoding be more appropriate for `season` or `weathersit`?
- Why is regularization useful after one-hot encoding?
- What behavior can a linear model not express easily for hour-of-day demand?

### 7. Evaluate both valid models

For the random and chronological models, report:

- MAE;
- RMSE;
- R-squared;
- mean residual;
- median absolute error.

Use a consistent sign convention such as:

```text
residual = actual - predicted
```

State your convention in the notebook.

Required plots for each split:

1. predicted versus actual values with a reference diagonal;
2. residual distribution;
3. residuals versus predictions.

For the chronological experiment, also plot actual and predicted demand over a
contiguous portion of the test period.

Checkpoint:

- Which split produces better metrics?
- Does the random split overstate likely future performance?
- Where does the model systematically underpredict or overpredict?
- Are errors larger during high-demand hours?

### 8. Segment the chronological errors

Using only the fixed chronological test predictions, calculate MAE for at least
three meaningful segments, such as:

- hour of day;
- working day versus non-working day;
- weather situation;
- season;
- month.

Show at least one table and one plot. Include sample counts so small groups are
not mistaken for reliable findings.

Checkpoint:

- Which segment has the largest absolute errors?
- Is that also the segment with the highest demand?
- Would normalized or percentage error tell a different story?
- What additional data could plausibly help, without claiming causation?

### 9. Deliberately leaky variant

For demonstration only, construct a model that includes `casual` and
`registered`. Keep the split and model family comparable to a valid experiment.

Before running it, predict what should happen. Then report its metrics and
explain why they are invalid for the stated task.

Do not call this the best model. Label it clearly as non-deployable.

Checkpoint:

- Why should this model approach a trivial arithmetic solution?
- If its score is not nearly perfect, what preprocessing or model limitation
  might explain the remaining error?
- How is exact target leakage different from an unusually predictive but valid
  feature?

### 10. Optional nonlinear comparison

This section is optional. Test at most one nonlinear tabular regressor using the
valid feature set. Do not launch a search over many model families.

Use unchanged holdout definitions. If you adjust model choices after seeing
chronological-test results, explicitly state that the test set has begun acting
like validation data.

Checkpoint:

- What interaction or nonlinear pattern can this model capture that Ridge
  cannot?
- Did both random and chronological performance improve?
- Is the improvement large enough to justify added complexity?

### 11. Experiment log

Maintain this table:

| Run | Split | Features | Model | MAE | RMSE | R-squared | Valid at prediction time? |
|---|---|---|---|---:|---:|---:|---|
| 0A | Random | Valid | Dummy mean | | | | Yes |
| 0B | Chronological | Valid | Dummy mean | | | | Yes |
| 1A | Random | Valid | Ridge | | | | Yes |
| 1B | Chronological | Valid | Ridge | | | | Yes |
| 2 | Chosen split | Includes component counts | Deliberately leaky | | | | No |

Add no more than one optional-model row.

### 12. Final report

Write a concise report containing:

1. **Problem:** target, unit, and prediction moment.
2. **Data:** size, date range, feature types, and target behavior.
3. **Leakage controls:** why `casual`, `registered`, and `instant` were removed.
4. **Evaluation design:** random versus chronological splitting.
5. **Results:** baseline and model metrics in units a stakeholder can understand.
6. **Residual analysis:** where errors are largest and what remains unexplained.
7. **Leaky comparison:** what happened and why it is invalid.
8. **Limitations:** at least three, including the observed-weather assumption.
9. **Recommendation:** what should be investigated next, not an immediate
   deployment claim.
10. **What I now understand:** five concrete learning statements.

## Minimum completion rubric

| Area | Completion standard |
|---|---|
| Framing | Target, units, prediction moment, and forecast-weather approximation are explicit. |
| Audit | Shapes, dtypes, categories, dates, missingness, duplicates, and target distribution are inspected. |
| Leakage | Arithmetic identity is verified and component counts are excluded from valid models. |
| Feature roles | Identifiers, continuous variables, coded categories, and dates are handled intentionally. |
| Evaluation | Both random and chronological holdouts are built and contrasted. |
| Baselines | Separate training-only dummy baselines are measured for both splits. |
| Pipeline | Learned preprocessing and Ridge are fit without test contamination. |
| Metrics | MAE, RMSE, R-squared, and residual behavior are correctly interpreted. |
| Error analysis | Chronological errors are analyzed by meaningful segments with sample counts. |
| Weak variant | Exact-leakage model is demonstrated and rejected. |
| Reproducibility | Seed, source, split boundaries, decisions, and run table are recorded. |
| Communication | Final report distinguishes interpolation-like evaluation from future forecasting. |

## Hint ladder

When blocked, state the section number and provide your attempted code, complete
error, observed shapes, and expected behavior. Assistance should progress from:

1. conceptual hint;
2. relevant pandas or scikit-learn mechanism;
3. pseudocode or unrelated miniature example;
4. partial project code;
5. complete local solution only if earlier levels fail.

## Optional extension (choose at most one)

- introduce a proper validation period between chronological train and test;
- encode hour and month with sine/cosine cyclical features;
- compare raw-target and `log1p`-target modeling with metrics returned to the
  original count scale;
- test one nonlinear model and analyze whether it improves peak-demand errors.

Do not perform all four. Stop after the core learning objectives are satisfied.
