## Annotations for functions

* `torch.randperm(n)`: Creates a random permutation/rearrangement of the integers from 0 to n-1; e.g. [0, 3, 1, 2, 4] for n=5

* `W_small.norm(dim=1, keepdim=True)`: Computes the L2 length of each row in W_small by reducing across columns/features;
> * `keepdim=True` keeps the result shaped like `[rows, 1]` instead of `[rows]`, so it can broadcast back across each row

* `torch.cat(..., dim=0)`: Concatenates tensors along dimension 0. The sizes along `dim=0` are added together, while all other dimensions must match exactly.
> * `dim=0` is the default if `dim` argument is not specified
> * For `torch.cat` to work, all dimensions except the chosen `dim` must match. If any non-concatenated dimension differs, PyTorch raises an error

* `X_demo[i].detach()` returns a tensor with the same values as `X_demo[i]`, but without its existing computation/edit history attached. This prevents stored tensors from keeping unnecessary history in memory.

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

# 11.8 Checkpoint
You are ready to move on when you can explain:

* why expert_W has shape [regions, features]
> * Because each row is one expert's weight vector. `expert_W[r]` has shape `[features]`, so examples routed to region `r` can compute `X[mask] @ expert_W[r]`.
> * Rule of thumb for weight shape in MoEs:
> * One selected expert row:

  ```
  X[mask] @ expert_W[r]
  ```

> * All experts at once:

  ```
  X @ expert_W.T
  ```

> * Rule of thumb for weight shape in linear regression:
> * PyTorch nn.Linear manually:

  ```
  X @ layer.weight.T + layer.bias
  ```

> * PyTorch nn.Linear normally:

  ```
  layer(X)
  ```

* why route_ids has shape [batch]
> * Because every example in `X` needs one expert ID. `route_ids[i]` tells us which expert should handle `X[i]`.
* why masks are needed
> * For expert `r`, `mask = route_ids == r` picks the rows of `X` and `y` that expert `r` should train on or predict for.
* why some experts may receive no update
> * An expert receives no update when no examples are routed to it.
> * Then its mask is all `False`, it does not participate in the loss computation, and no gradient is produced for that expert's weights or bias.
> * If it sees 2 examples, it still receives an update, just based on very little data.
* why this is not the same as training four separate global models manually
> * Because each expert only trains on examples routed to it, not the whole dataset.
> * The routed model uses one training loop, one combined loss, and one optimizer step, but the mask makes each expert specialize on its assigned subset.

# 12.4 What To Learn
* If oracle routing is much better than similarity routing:
> * The expert model can work, but the **router**is weak.

If similarity routing is much better than random routing:
> * The **region table** carries useful information.

If global model matches routed experts:
> * The **regions** may not actually have different enough rules, or the task is too easy.

If all models fail:
> * The learning rate, data, labels, or loss code may be broken.

# 13.4 What To Learn
* Weight decay usually increases training loss slightly.
* It can improve test loss if the model was overfitting.
* It can hurt if the task needs large weights or the model was not overfitting.

# 13.5 Common Confusion Points
* Weight decay is not dropout.
* Weight decay is not a learning rate.
* Weight decay changes the objective.
* A smaller weight norm is not automatically better.

# 15.7.7 Trace Checklist
Before moving on, trace the function with these questions:

1. What does one row of X represent?
> * Within the row represents various elements of features per example. One row simply represents 1 example.

2. What does one value in route_ids represent?
> * The respective attribution to MoE. e.g. [0, 1, 2, 2, 0] means that over the 5 examples from index 0 to 4, each is attributed to regions/MoEs 0, 1, 2, 2, and 0

```
  X[mask].shape        # [num_routed_examples, features]
  expert_W[r].shape    # [features, classes]
  result.shape         # [num_routed_examples, classes]
```

3. What is the shape of `expert_W[r]`?
> * `[features, classes]`

4. Why does `X[mask] @ expert_W[r]` produce class logits?
> * Because the shape of `[batch, classes]` from `X[mask] @ expert_W[r]` computes one score per class for each selected example.

5. Why does `logits[mask]` have the same row count as `X[mask]`?
> * Because both use the same boolean mask over the batch dimension. If 2 examples were routed to expert r, then `X[mask]` has 2 rows and `logits[mask]` also has 2 rows.

6. Why does this function return logits instead of predicted classes?
> * Because `CrossEntropyLoss` expects logits during training.
> * Hard predicted classes from `argmax` are not useful for training because it is not differentiable and loses the score information.
> * We can use `pred = logits.argmax(dim=1)` since `dim=1` is the class dimension. Each row contains 1 score/class, and `argmax` returns the class index with the highest score.

7. Why are route IDs not the same thing as class labels?
> * Route IDs choose the expert. Class labels are the target answers the model is trying to predict.
> * Example 3 might be routed to expert 2, but its correct class could still be class 0, 1, or 2.

Common failure modes:

* using class labels as `route_ids` (these are for MoEs whereas class labels are the target classes for `CrossEntropyLoss`)
* returning `argmax` predictions instead of logits (`argmax` is for evaluation/inference after the model produces logits)
* applying softmax inside the helper (`CrossEntropyLoss` expects raw logits and applies the log-softmax part internally)
* forgetting `if mask.any()` (helps to skip experts with no routed examples safely, especially if computing per-expert loss)
* making `expert_W` shape `[regions, classes, features]` (**WRONG**) instead of `[regions, features, classes]` (**CORRECT**)

# 15.11 Checkpoint

You are ready to move on when you can explain:

* why classification labels have shape `[batch]`
> * Each example has one correct class ID. So `y[i]` is the target class for `X[i]`, and `y` has one value per example.

* why logits have shape `[batch, classes]`
> * Each example needs one raw score per possible class. So if there are 500 examples and 3 classes, logits have shape `[500, 3]`.

* why `F.cross_entropy(logits, y)` receives raw logits
> * Because PyTorch cross_entropy already includes softmax calculation.
> * If hand-written (which this example is however it mimics PyTorch), you are welcomed to outsource softmax separately outside of cross_entropy

* why `argmax(dim=1)` converts logits into predicted class IDs
> * Because argmax picks the largest element inside the dimension, so dimension=classes has 80% cat and 20% dog, argmax will retain the label of cat

* how routed classification differs from routed regression
> * Classification tends to work with metadata (e.g. text) while regression works with numbers. A good example is cat vs dogs and housing prices predictions
> * Routed regression returns one number per example, usually shape `[batch]`. Routed classification returns one vector of class scores per example, shape `[batch, classes]`.

* why classification `expert_W` has shape `[regions, features, classes]`
> * The full routed model stores one classifier expert per region. For a single region `r`, `expert_W[r]` has shape `[features, classes]`, mapping input features to class logits.
> * Stacking all region experts together gives `[regions, features, classes]`.

# 16.4.4 File Edit: Full Function Trace checklist for `top2_routed_predict_regression()`

1. Why does `top_ids` have 2 columns?
> top_ids have the shape of (batch, k) for k = top selection. For (5, 2), for example, each example (over 5 total) will receive 2 route_ids for MoE inference (prediction)

2. Why is `top_ids[:, j]` a normal route_ids vector?
> `top_ids` has shape `[batch, 2]`. Selecting 1 column with `top_ids[:, j]` gives shape `[batch]`, or 1 expert ID per example, akin to a normal top-1 `route_ids`.

3. What does `preds[i, 0]` store?
> The regression prediction for example `i` made by its 1st choice expert. The expert ID itself is stored in `top_ids[i, 0]`.

4. What does `preds[i, 1]` store?
> The regression prediction for example `i` made by its 2nd choice expert. The expert ID itself is stored in `top_ids[i, 0]`.

5. Why does `preds.mean(dim=1)` return shape `[batch]`?
> Because predictions/inferences are now averaged together to 1 answer per example, so shape = `[batch]` or # of examples

6. Why does this function return top_ids as well as predictions?
> The averaged predictions are used for loss/accuracy, while `top_ids` lets us inspect which experts were selected for each example.
> This is useful for debugging routing, measuring region usage, or comparing top-1 vs top-2 routing.

# 16.6 Checkpoint

You are ready to move on when you can explain:
* why top_ids has shape `[batch, 2]`
> Because each example requires 2 experts where k=2 for top-k MoE routing

* why `top_ids[:, 0]` and `top_ids[:, 1]` each have shape `[batch]`
> Because you are selecting 1 column of top_ids, out of the 2 available experts to choose from

* why top-2 router accuracy can improve while prediction loss gets worse
> Because router accuracy only calculates when EITHER expert is correct, while prediction is averaged from BOTH experts
> So if the second expert performs terribly, it doesn't count into router accuracy, but it will be factored into prediction loss

* why "more active experts" is not automatically better
> Per above, if the 2nd expert is terrible it could significantly degrade the inference quality, which is an average 

# 17.5 Checkpoint

You are ready to move on when you can explain:

* why this gate is batch-level rather than per-example
> This gate is batch-level because `routed_regression_loss(...)` returns one average loss over the routed training batch.
> The threshold compares against that 1 scalar, so the optimizer step is either allowed or skipped for the whole batch.
> A per-example gate would require individual example losses and a different update rule

* why skipped epochs do not update parameters
> Because skipped epochs have acceptable (lower) loss comparing to the threshold
> From a mechanical standpoint, skipped epochs do not call `loss.backward()` or `sgd(...)` -> no backward pass -> no gradient update -> no sgd -> `expert_W` and `expert_b` UNCH

* why a low threshold behaves like ordinary training
> A low threshold behaves like ordinary training because the batch loss is usually greater than the threshold, so the update branch runs on most or all epochs
> That makes it close to threshold=None, where every epoch updates

* why a high threshold can cause underfitting
> Because many epochs are skipped -> fewer optimizer updates -> expert weights and biases may not adapt enough to the training data

* why this is only a toy analog of local gated learning
> Because the gate is a simple hand-written threshold on global batch loss.
> Real local gated learning would use more local signals, such as per-expert/example/synapse activity, possibly with separate eligibility or reward signals
> Here one scalar loss controls whether the whole model updates, so it is useful mechanically but not a realistic local learning rule

# 18.6 Checkpoint

You are ready to move on when you can explain:

* why `expert_W.norm(dim=1)` gives one norm per expert
> Because `expert_W` has shape `[regions, features]`, where each row is one expert's weight vector
> `dim=1` reduces across the feature columns, so the result has 1 L2 norm per expert row

* why `keepdim=True` matters for broadcasting
> `keepdim=True` keeps the norms shaped like `[regions, 1]` instead of `[regions]`
> That lets each expert's single scale value broadcast across that expert's feature columns when multiplying `expert_W * scale`

* why homeostatic scaling is not the same as weight decay
> Weight decay is added to the loss, so it changes the gradients during `loss.backward()`
> Homeostatic scaling happens after the optimizer step and directly rescales `expert_W` outside the loss. It makes each expert row move toward a target norm, not each individual weight value

* why smaller weight norm is not automatically better
> Smaller norms can reduce overfitting or instability, but if the weights become too small, the model may underfit
> Some data-generating rules require larger weight magnitudes, so forcing weights too small can hurt train and test loss
> Fraud detection
  - Large weight: “new device + foreign IP + unusual purchase size”
  - Smaller weight: “purchase made on weekend”
  - Reason: some signals sharply change fraud probability; others are weak context.
> Large weights: strong, reliable, high-sensitivity signals.
> Small weights: weak, noisy, redundant, or low-impact signals.
> Too-small weights: underfitting.
> Too-large weights: overfitting or instability.

# 19.7 Checkpoint

You are ready to move on when you can explain:

* why replay uses `torch.cat(..., dim=0)`
> Because you need to append the older examples into the new ones to create the training batch, so `torch.cat` creates 1 larger training batch without changing the # of features

* why replay can help old-region performance
> Because old examples keep appearing during later training. Without replay, stage 2 mostly optimizes for the new distribution, so old-heavy regions get little or no training signal
> Replay gives old regions more examples in the loss, which can reduce forgetting or keep old experts better calibrated
> The broader issue is that the later training objective is dominated by the new distribution, so old-region performance may stop improving, drift, or become underrepresented

* why replay can sometimes hurt new-region performance
> If old and new examples push parameters in different directions, or if model capacity/training time is limited, preserving old performance can slightly trade off against new performance

* why replay and sparse routing solve different problems
> Sparse routing decides which expert handles each example, helping experts specialize locally.
> Replay decides what examples are included during training, helping older data remain represented after the training distribution shifts.

# 21.1 Regression Metrics

Use these for regression:

* train MSE
* test MSE
* per-region MSE
* weight norm
* router accuracy
* region usage count
* number of parameter updates

# 21.2 Classification Metrics

Use these for classification:

* train accuracy
* test accuracy
* train cross-entropy
* test cross-entropy
* per-region accuracy
* confusion matrix
* router accuracy
* region usage count

# 25. Final Writeup Prompts

At the end, write a short technical note answering:

* What did the global regression model learn?
> It learned one single linear rule, one `w` and one `b`, to predict numeric targets from input features.
> On multi-region data, this global rule had to average across different region-specific rules, so it could underfit.

* What did the global classification model learn?
> It learned one global classifier that maps input features to class logits. The logits are raw class scores; `argmax` turns them into predicted class IDs.

* When did routed experts outperform the global model?
> When there is a big variety in the data (e.g. types of data are not just cats or dogs but billions of species), or different types of hidden rules that would be difficult for a single model to handle
> In that case, each expert could specialize instead of one global model averaging all regions together

* When did routed experts fail?
> They failed or helped less when routing was bad, when experts received too little data, when regions were not actually different enough, or when the task was simple enough for one global model.

* How much did router quality matter?
> Matters massively. Oracle routing (router knows the answer) was the upper bound, random routing was weak, and similarity routing worked well when input geometry matched the true regions (same tensor direction).

* What did distribution shift do?
> Distribution shift changed which regions appeared often during training or testing. For example, old data mostly used regions 0 and 1, while new data mostly used regions 2 and 3.
> This can make the model perform well on the new distribution while forgetting or undertraining the old one.

* Which parameters received gradients during sparse training?
> Only the experts selected by `route_ids` received gradients from their routed examples
> For expert `r`, `expert_W[r]` and `expert_b[r]` update only if at least one example is routed to expert `r`

* What did weight decay change?
> Weight decay added a penalty for large weights, pushing weights toward smaller norms. This can reduce overfitting or instability, but it can hurt if the task needs larger weights/stronger signals

* What did per-region metrics reveal that average metrics hid?
> Per-region metrics showed whether some regions performed much better or worse than others. A good average loss can hide one rare region failing badly.

* Which part connects most clearly to D2L Chapters 2-4?
> Tensor shapes, linear models, MSELoss, softmax -> CrossEntropy, sgd, backward() or chain-rules

* Which part feels like a real research question rather than a solved exercise?
> Whether Replay Buffer (mixing older examples in newer training set) meaningfully improves model performance (older experts receive updates alongside newer experts)?
> How to choose routing, replay, local update gates, and expert regularization so the model learns new distributions without forgetting old ones.

## Stopping Point

Date: 2026-08-08

Current phase:
Project 0 has been completed through Chapter 25: Final Writeup Prompts.
The main tutorial phases are complete through Phase 13: Optional Replay Buffer.
Chapter 21 metrics notes have been reviewed, including how to read a confusion matrix.
Chapter 25 final writeup answers have been written in `notes.md`.

Working state:
- venv works
- torch imports
- safe_import cell works / should be used in experiments.ipynb
- Phase 1 linear regression completed through 7.10
- Phase 2 multiple-region data notes completed through 8.7
- Phase 3 global linear model experiment completed through 9.4
- Phase 4 similarity router completed through 10.7
- Phase 9 classification version completed through 15.6
- Phase 9 classification model drills completed through 15.11:
  - 15.7.1 Tiny Routed Classification Shapes
  - 15.7.2 Boolean Mask Drill
  - 15.7.3 One-Expert Logit Drill
  - 15.7.4 Fill-Back Drill
  - 15.7.5 Full Loop Drill
  - 15.7.6 Function Version Drill
  - 15.7.7 Trace Checklist
- Phase 10 top-2 routing completed through 16.4.4:
  - top-k route shape drills
  - `top_ids[:, 0]` as first-choice route IDs
  - `top_ids[:, 1]` as second-choice route IDs
  - `preds[:, 0]` as first-choice expert predictions
  - `preds[:, 1]` as second-choice expert predictions
  - `preds.mean(dim=1)` as the averaged top-2 prediction
  - 16.4.4 trace checklist written
- 16.5 completed:
  - training still uses top-1 similarity routing
  - evaluation compares top-1 prediction vs top-2 averaged prediction
  - use tutorial variable names `train_top2_ids` and `test_top2_ids`
  - `train_top2_ids` has shape `[train_examples, 2]`
  - `test_top2_ids` has shape `[test_examples, 2]`
  - `top2_usage = torch.bincount(train_top2_ids.reshape(-1), minlength=num_regions).tolist()`
  - `top2_router_acc = (test_top2_ids == test_region_ids[:, None]).any(dim=1).float().mean()`
  - verified `models.py` exposes `top2_routed_predict_regression`
  - experiment output from the venv run:
    - `top-1 similarity`: train loss `0.733542263507843`, test loss `0.735161304473877`, router accuracy `0.9399999976158142`, usage `[108, 89, 105, 98]`
    - `top-2 similarity average`: train loss `0.8284111022949219`, test loss `0.8642458915710449`, router accuracy `0.9900000095367432`, usage `[282, 271, 139, 108]`
  - interpretation written in `experiments.ipynb`: top-2 router accuracy improved, but prediction loss worsened because averaging with a second expert can pull predictions away from the target.
- 16.6 checkpoint annotations completed in `experiments.ipynb`:
  - `top_ids` has shape `[batch, 2]`
  - `top_ids[:, 0]` and `top_ids[:, 1]` each have shape `[batch]`
  - top-2 router accuracy can improve while prediction loss gets worse
  - more active experts is not automatically better
- Phase 11 local update gate completed through 17.5:
  - 17.4.1 gated helper grammar drill completed
  - 17.4.2 threshold condition table completed
  - 17.4 Local Update Gate Experiment completed in `experiments.ipynb`
  - `run_gated_similarity_regression(threshold)` trains one fresh similarity-routed model per threshold
  - thresholds compared: `[None, 0.2, 0.5, 1.0]`
  - `threshold=None` means ordinary training: always update
  - numeric thresholds update only when `loss.item() > threshold`
  - `update_count` counts optimizer steps, not loop iterations
  - evaluation records train/test prediction loss, test router accuracy, and test per-region loss
  - 17.5 checkpoint answers written:
    - the gate is batch-level because `routed_regression_loss(...)` returns one scalar batch loss
    - skipped epochs do not call `loss.backward()` or `sgd(...)`, so parameters do not update
    - low thresholds behave like ordinary training because most epochs still update
    - high thresholds can underfit by skipping too many optimizer updates
    - this is a toy analog because one global batch-loss threshold controls the whole model update
- Phase 12 optional homeostatic scaling completed through 18.6:
  - 18.3 norms/rescaling drill completed
  - 18.4 in-place rescale grammar drill completed
  - `rescale_expert_weights(expert_W, target_norm=1.0)` added to `train.py`
  - 18.5 Weight Decay vs Homeostatic Scaling Experiment completed in `experiments.ipynb`
  - compared `none`, `weight decay`, `homeostatic scaling`, and `weight decay + scaling`
  - evaluation records train/test prediction loss, whole-`expert_W` norm, and per-expert row norms
  - 18.6 checkpoint answers written:
    - `expert_W.norm(dim=1)` gives one norm per expert because it reduces across feature columns for each expert row
    - `keepdim=True` keeps norms shaped `[regions, 1]` so scale values broadcast across feature columns
    - homeostatic scaling happens outside the loss and directly rescales `expert_W`; weight decay changes the loss and gradients
    - smaller weight norm is not automatically better because too-small weights can underfit or fail to match high-sensitivity data rules
- Phase 13 optional replay buffer completed through 19.7:
  - 19.3 `torch.cat(..., dim=0)` reviewed:
    - concatenates along the batch/example dimension
    - selected dimension sizes are added
    - all other dimensions must match exactly
  - 19.4 `ReplayBuffer` implemented in `train.py`:
    - stores detached cloned examples in `self.X` and `self.y`
    - keeps only the most recent `max_size` examples with `self.X[-self.max_size:]`
    - samples random stored examples using `torch.randint`
    - stacks sampled examples back into batch tensors with `torch.stack`
  - 19.5 replay add/sample drills completed:
    - `detach()` cuts tensor history without changing values
    - `clone()` makes an independent copy
    - `torch.stack([...])` turns separate stored examples into a batch
  - 19.6 curriculum replay experiment completed:
    - old mixture: `[0.45, 0.45, 0.05, 0.05]`
    - new mixture: `[0.05, 0.05, 0.45, 0.45]`
    - first trained on old distribution, then trained on new distribution with or without replay
    - replay branch used `torch.cat([new_X_train, replay_X], dim=0)` and matching `batch_y`
    - after resetting seed inside `run_curriculum`, observed:
      - no replay: old test loss `0.39968687295913696`, new test loss `0.5190250277519226`
      - replay: old test loss `0.3869458734989166`, new test loss `0.5516014695167542`
    - interpretation: replay slightly improved old-distribution performance and slightly worsened new-distribution performance, so the effect was real but modest in this toy setup
  - 19.7 checkpoint answers written:
    - replay uses `torch.cat(..., dim=0)` to append old examples as more batch rows
    - replay can help old-region performance by keeping old examples represented during later training
    - replay can hurt new-region performance when old examples compete with new examples under limited capacity or training time
    - sparse routing chooses which expert handles each example, while replay chooses which examples remain represented during training
- Chapter 21 metrics reviewed:
  - regression metrics: train/test MSE, per-region MSE, weight norm, router accuracy, region usage count, parameter update count
  - classification metrics: train/test accuracy, train/test cross-entropy, per-region accuracy, confusion matrix, router accuracy, region usage count
  - confusion matrix interpretation clarified:
    - rows are true labels
    - columns are predicted labels
    - `matrix[true_class, predicted_class]` is the occurrence count for that true/predicted pair
    - diagonal entries are correct predictions; off-diagonal entries are mistakes
- Chapter 25 final writeup answers completed:
  - global regression learns one linear rule and can underfit multi-region data
  - global classification learns one global logits-producing classifier
  - routed experts help when regions have different hidden rules and routing is good
  - routed experts fail or help less when routing is bad, data per expert is low, regions are not meaningfully different, or the global model is sufficient
  - router quality matters strongly: oracle routing is the upper bound, random routing is weak, similarity routing works when geometry matches regions
  - distribution shift changes which regions dominate training/testing and can cause forgetting or undertraining
  - only selected experts receive gradients during sparse training
  - weight decay pushes weights toward smaller norms but can hurt if strong weights are needed
  - per-region metrics reveal failures hidden by average metrics
  - D2L connections include tensor shapes, linear models, MSE, softmax, cross-entropy, `backward()`, and SGD
  - replay/routing/local gates/regularization under distribution shift feel like the real research questions
- Clarified that `torch.where(mask)[0]` returns original batch row indices, not expert IDs.
  - Example: `route_ids_small = [0, 2, 2, 1, 3]`, `r = 2`, so the true mask positions are rows `1` and `2`.
- Clarified routed classification logits:
  - `logits_small` has shape `[batch, classes]`.
  - Each row stores class scores for one example, not probabilities.
  - `pred_small = logits_small.argmax(dim=1)` returns the predicted class index for each example.
  - Columns/positions correspond to class IDs; values inside logits are class scores.
- Clarified why the routed classification function returns logits:
  - `CrossEntropyLoss` expects logits during training.
  - Hard predictions from `argmax` are not useful for training because `argmax` is not differentiable and loses score information.
- 10.6 break-it experiments done:
  - remove normalization
  - increase feature_noise from 0.3 to 2.0
  - set all region prototypes to the same vector
  - use k=2

Next step:
Project 0 is ready to close off.
If continuing later, use Chapter 27 Final Done Criteria in `TUTORIAL.md` only as a final checklist.
Optional cleanup: verify whether `confusion_matrix` was added to `metrics.py` if full optional completion is desired.

Prompt:
Read my notes.md and continue only if doing final cleanup/checklist work after Chapter 25.

# Key Takeaways
  You built the right foundation: tensor shapes, scalar losses, manual backward(), gradient clearing,
  SGD, train/test separation, and reading outputs mechanically instead of just asking whether a number
  “went down.”

  The strongest technical learning was sparse routing. You learned that route_ids choose experts,

  You also learned that regularization is contextual. Weight decay and homeostatic scaling reduced
  norms and helped in some runs, but the real lesson is not “smaller weights are better.” It is:
  weight size, generalization, underfitting, and task signal strength interact.
  loss, 0.3997 -> 0.3869, while slightly worsening new-distribution loss, 0.5190 -> 0.5516. That is a
  good toy example of stability/plasticity tradeoff.

  Progress
  My read of your progress: you moved from “what does this PyTorch function do?” to “what variable is
  allowed to differ, what dimension is being reduced, what parameter receives gradients, and is this
  comparison controlled?” That is a meaningful shift.

  The best habit you showed was not letting terms stay fuzzy. You drilled into cat, stack, detach,
  slicing, logits, argmax, masks, and confusion matrices until they became mechanical. That is exactly
  the kind of foundation future projects need.

# Bedrocks For Future Projects
Carry these forward:

- Always write shape contracts.
- Always keep global/random/oracle-style baselines when possible.
- Treat average metrics with suspicion; inspect per-region/per-class behavior.
- Separate routing quality from model quality.
- Separate training performance from generalization.
- Make experiments controlled when comparing conditions: same seed, same init, same data split.
- Keep notebooks clean-restartable.