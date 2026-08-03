## Annotations for functions

* torch.randperm(n): Creates a random permutation/rearrangement of the integers from 0 to n-1; e.g. [0, 3, 1, 2, 4] for n=5

## 7.9 Break It Deliberately

Try these one at a time:

- remove `p.grad.zero_()` in `train.py`
    * Euh, idk, I only saw my training AND test loss getting smaller which can only be a good thing...?
    * CORRECT ANSWER: Without grads/computational cleanups, gradients will accumulate across epochs, so you accidentally created a weird growing-step optimizer. On this tiny convex linear problem, that can look like “faster learning.” But it is unstable and not the intended SGD update.

- set `lr=5.0` in `experiments.ipynb` (was 0.05)
    * Wow, my training AND test loss got EVEN smaller than after getting rid of `p.grad.zero_()`!
    * I feel like this is because my layer is simple (just linear with simple convexity), and conversely, I feel like bigger lr may lead it to overfit.
    * CORRECT ANSWER: Large LR can speed convergence, but it can also overshoot or diverge. It is an optimization stability issue, not directly an overfitting issue. If lr=5.0 worked better, then it might be 1) original lr=0.05 being too slow, 2) the problem is simple/convex, 3) got lucky with initialization/data, 4) accidentally continued training from already-trained weights, 5) the run has not gone long enough to show instability

- set `noise_std=5.0` in `data.py` (was 0.1)
    * Hmm, my loss actually got larger, which means that this simple linear model learned part of the noise, when it was supposd to only focus on improving `w` and `b`.
    * CORRECT ANSWER: Higher label noise raises the minimum possible loss. With enough data, the model should learn the signal while noise averages out. With little data, the noise can distort the learned weights.
    
- use only 10 training examples (change `train_fraction` in `data.py` from 0.8 to 0.05 given `make_regression_data(200, 6)`)
    * Yup, training AND test loss got bigger because there is less data
    * Interestingly, test loss can be smaller than training loss, but this is just because the model is poorly underfit
    * CORRECT ANSWER: Less data increases estimation error and generalization variance. Test loss being smaller than train loss does not automatically mean underfitting. It can simply mean the tiny training set was noisier/harder than the test set. Given that

- evaluate on training data only (change `test_loss`' X and y dataset to training on `experiments.ipynb`)
    * Test loss just became the same to training loss whereas it should have been slightly bigger
    * CORRECT ANSWER: This destroys the generalization check. The number may look good, but it no longer measures unseen-data performance.

Write what happens in `notes.md`.

# 7.10 Checkpoint

You are ready to move on only when you can answer:

* What is the shape of X?
    * (200, 6) give X, y, true_w, true_b = make_regression_data(200, 6)

* What is the shape of w?
    * (6,) given w = torch.randn(6, requires_grad=True)

* Why must the loss be scalar before calling backward()?
    * Because we need to make sure the loss is fully derived before passing it back to the params
    * CORRECT ANSWER: A scalar loss gives PyTorch one objective to differentiate. If loss is a vector of per-example losses, PyTorch does not know what weighted combination of those losses I want gradients for unless I reduce it with mean/sum or explicitly provide gradient weights.

* Why do gradients need to be cleared?
    * Because PyTorch does not clear gradients on its own, and accumulating gradients create an evergrowing-step optimizer that could distort gradients.
    * CORRECT ANSWER: PyTorch accumulates gradients by default. If I do not clear them, each update uses old gradients plus new gradients, creating an unintended update rule.

* How do train loss and test loss differ?
    * Train loss focuses on loss versus the training dataset, and test loss compares against un-trained dataset (or how well the model generalizes vs memorizes).
    * CORRECT ANSWER: Train loss measures fit on data used for parameter updates. Test loss measures performance on data not used for updates, so it estimates generalization. Avoid “memorizes” for this tiny linear model unless you are specifically talking about overfitting. Linear regression can overfit, but here the main concept is generalization.

* What does it mean if w is close to true_w?
    * That the loss is very small and congratulations!
    * CORRECT ANSWER: It means the model recovered the hidden linear rule used to generate the synthetic labels. The learned parameters are close to the ground-truth parameters. Loss should be small too, but w close to true_w is specifically evidence that the learned rule matches the data-generating rule.

# 8.3 Region Prototpyes

* Suppose we have 3 regions and 2 features:

```
  region_table = torch.tensor([
      [1.0, 0.0],   # region 0 points right
      [0.0, 1.0],   # region 1 points up
      [-1.0, 0.0],  # region 2 points left
  ])
```


* Now suppose an input/query is:

```
  x = torch.tensor([0.9, 0.1])
```

* This points mostly right, so it should match region 0.

* The comparison is usually a dot product / cosine similarity:

```
  scores = x @ region_table.T
  print(scores)
```

* Result:

```
  tensor([ 0.9000,  0.1000, -0.9000])
```

* Interpretation:

```
  score vs region 0 =  0.9  high match
  score vs region 1 =  0.1  weak match
  score vs region 2 = -0.9  opposite direction
```

* Then route to the best region:

```
  chosen_region = scores.argmax()
  print(chosen_region)
```

* Result:

```
  tensor(0)
```

* For a batch:

```
  X = torch.tensor([
      [0.9, 0.1],    # near region 0
      [0.2, 0.8],    # near region 1
      [-0.7, 0.1],   # near region 2
  ])

  scores = X @ region_table.T
  top_regions = scores.argmax(dim=1)

  print(scores)
  print(top_regions)
```

* Output:

```
  tensor([[ 0.9000,  0.1000, -0.9000],
          [ 0.2000,  0.8000, -0.2000],
          [-0.7000,  0.1000,  0.7000]])

  tensor([0, 1, 2])
```

* That is the basic router idea:

```
  input vector @ region prototype vectors -> similarity scores
  highest score -> selected expert
```

* Because region_table rows are normalized to length 1, the dot product behaves like cosine similarity if x is also normalized.

# 8.7 Common Confusion Points

* `region_table` helps create inputs and route inputs.
* `true_W` creates labels.
* `region_ids` are known only because this is synthetic data.
* In real data, you usually do not get perfect region IDs because human-provided tags may not use the right ontology and may not map cleanly to your MoE experts.
* Do not train on region_ids as labels unless the experiment explicitly asks for oracle routing.

# 9.4 Experiment Prompt

* Experiment: global linear model on multi-region regression

* What I expected: Global model should perform worse than single rule linear models because dataset is generated from multiple region-specific linear rules.

* What happened: Both training and test loss got larger than the earlier data generation/training/run in the earlier section of experiemnts.ipynb

* Train loss: 0.7520253658294678 (vs 0.012925769202411175 1st run with single linear rule)

* Test loss: 0.7981556057929993 (vs 0.013824677094817162 1st run with single linear rule)

* Why one global model may struggle: Because the new data is generated by multiple region-specific linear rules, not by one global linear rule. The global linear model is therefore too simple because it must average across several different linear relationships.

* Interpretation: Since train and test loss are both high and relatively close, this looks more like underfitting/model mismatch than overfitting.

# 10.2 Cosine Similarity

* Cosine similarity compares direction rather than raw length.

* For two vectors:

```
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
```

* Where
> *  `dot(a, b)` measures how aligned the two vectors are. If they point in a similar direction, the dot product tends to be large.
> *  `norm(a)` and `norm(b)` calculate the lengths of the vectors.
> * Dividing by the two norms removes the effect of vector length. That means cosine similarity focuses on angle/direction, not magnitude.

* Example:

  ```python
  a = torch.tensor([1.0, 1.0])
  b = torch.tensor([2.0, 2.0])

  These point in the exact same direction, even though b is longer. Their cosine similarity is 1.0.

* But:

  a = torch.tensor([1.0, 0.0])
  b = torch.tensor([0.0, 1.0])

  These are perpendicular, so their cosine similarity is 0.0.

* Typical interpretation:

   1.0  = same direction
   0.0  = unrelated/perpendicular direction
  -1.0  = opposite direction

* In your MoE/router setup, cosine similarity can answer:

  > Which region prototype does this input vector point most similarly toward?


* If two vectors point in similar directions, cosine similarity is high.

# 10.6 Break It Deliberately
Try:
* remove normalization:
> * `make_region_table` already creates unit-length prototypes, so removing `normalize_rows(X)` should not change top-1 routing much, and often should give the same predicted ids

* increase feature_noise from 0.3 to 2.0: 
> * Router accuracy should drop because examples are no longer close to their assigned region prototypes.

* set all region prototypes to the same vector:
> * Router accuracy collapesed to random average (1/4) because every region has the same similarity score and the router cannot distinguish regions

* use k=2: Router accuracy improved by 5% (from 0.91 to 0.94)?
> * Top-2 routing accuracy improved because the correct region only needs to appear among the two selected regions. This is easier than top-1, so accuracy should be higher.

# 10.7 Common Confusion Points
* Routing accuracy is not model accuracy.
* The router can be right while the expert is untrained.
* The expert can learn if routing is noisy, but the job becomes harder.
* The router uses input geometry, not labels.

## Stopping Point

Date: 2026-08-03

Current phase:
Completed through 10.7 Common Confusion Points.

Working state:
- venv works
- torch imports
- safe_import cell works / should be used in experiments.ipynb
- Phase 1 linear regression completed through 7.10
- Phase 2 multiple-region data notes completed through 8.7
- Phase 3 global linear model experiment completed through 9.4
- Phase 4 similarity router completed through 10.7
- 10.6 break-it experiments done:
  - remove normalization
  - increase feature_noise from 0.3 to 2.0
  - set all region prototypes to the same vector
  - use k=2

Next step:
Start Phase 5: Routed Regression Experts.
Begin with 11.2 Expert Parameter Shapes in experiments.ipynb.

Prompt:
Read my notes.md and continue from the stopping point.
