# Active Learning for Efficient Data Labeling

Active learning is a training strategy in which a model helps decide which unlabeled examples should be labeled next. It is useful when obtaining labels requires expert time, manual review, experiments, or expensive annotation.

Instead of labeling a random sample from the entire pool, the learner selects examples expected to improve the model or reduce uncertainty.

## The active-learning loop

1. Start with a small, representative labeled seed set and a larger unlabeled pool.
2. Train a baseline model on the labeled set.
3. Score the unlabeled examples with a query strategy.
4. Select a batch for human annotation.
5. Add verified labels to the training set.
6. Retrain or incrementally update the model.
7. Evaluate on a fixed, untouched test set and repeat.

The fixed test set must not participate in query selection or training. Otherwise, the reported score no longer estimates performance on unseen data.

## Common query strategies

### Uncertainty sampling

Select examples for which the model is least confident.

- **Least confidence:** choose the smallest maximum class probability.
- **Margin sampling:** choose the smallest difference between the two largest class probabilities.
- **Entropy:** choose the largest predictive entropy.

For class probabilities p_1, ..., p_C:

\[
H(p) = -\sum_{c=1}^{C} p_c \log p_c
\]

High entropy indicates a diffuse probability distribution. It does not necessarily mean the example is informative: a badly calibrated model can be confidently wrong or uncertain for unhelpful reasons.

### Query by committee

Train several models and select examples where their predictions disagree. Disagreement can reveal decision-boundary cases that a single model misses.

### Diversity sampling

Select examples that cover different regions of the input space. This helps avoid spending an entire annotation batch on near-duplicates.

### Expected model change or expected error reduction

Estimate how much labeling an example might change the model or improve future performance. These methods can be expensive because they require additional model computations.

## Simple uncertainty-sampling example

This example assumes a fitted classifier with predict_proba and an unlabeled feature matrix X_pool. It selects a batch by predictive entropy.

~~~python
import numpy as np

probabilities = model.predict_proba(X_pool)
eps = 1e-12
entropy = -(probabilities * np.log(probabilities + eps)).sum(axis=1)

batch_size = 50
selected_indices = np.argsort(entropy)[-batch_size:]
X_to_annotate = X_pool[selected_indices]
~~~

The selected indices refer to the current pool. After annotation, remove those rows from the pool and preserve a stable record ID so labels cannot become detached from examples.

## Practical design choices

- **Seed set:** Start with a small but diverse sample, not only easy examples.
- **Batch size:** Small batches allow frequent feedback but increase operational overhead. Large batches are efficient but may contain redundant examples.
- **Annotator agreement:** Track disagreement, adjudication, and ambiguous cases.
- **Class coverage:** Add safeguards if rare classes are important but seldom selected.
- **Stopping rule:** Stop when the evaluation gain per annotation cost becomes too small or a budget is reached.
- **Label delay:** If annotations arrive asynchronously, track pending items so the same example is not repeatedly selected.

## How to evaluate the strategy

Compare active learning with random sampling under the same labeling budget. Plot test performance against the number of labels or annotation hours. Use multiple seeds when possible because the initial seed set can strongly affect results.

Useful measures include macro-F1 for imbalanced classification, task-specific error, annotation cost, and the number of examples requiring adjudication. Do not compare only the final score; the goal is to achieve useful quality with fewer labeling resources.

## Failure modes

- **Sampling bias:** Querying only uncertain cases can produce a labeled set unlike the deployment population.
- **Model blind spots:** Uncertainty methods cannot find examples the model is confidently wrong about.
- **Noisy annotations:** More labels do not help if annotation quality is poor.
- **Uncalibrated probabilities:** Entropy and margin rankings can be misleading.
- **Test leakage:** Using the test set to choose queries makes evaluation optimistic.

A strong workflow often combines uncertainty with diversity, random exploration, and class-coverage constraints.

## Mini-project

Train a text classifier using a small labeled seed set. Compare random sampling, entropy sampling, and a mixed strategy that selects half by entropy and half by embedding diversity. Keep one untouched test set and report macro-F1 after each annotation round.

## References

- Burr Settles, *Active Learning Literature Survey* (2009): https://minds.wisconsin.edu/handle/1793/60660
- modAL active learning library: https://modal-python.readthedocs.io/
