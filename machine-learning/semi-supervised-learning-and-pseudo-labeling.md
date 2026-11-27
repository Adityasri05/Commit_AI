# Semi-Supervised Learning and Pseudo-Labeling

Semi-supervised learning uses a small labeled dataset together with a larger unlabeled dataset. It is useful when collecting raw examples is inexpensive but obtaining trustworthy labels requires expert time, manual review, or costly experiments.

The challenge is that unlabeled examples do not reveal their correct targets. A method must exploit structure in the data without repeatedly reinforcing its own mistakes.

## Problem setup

- Labeled data: D_L = {(x_i, y_i)}.
- Unlabeled data: D_U = {x_j}.
- Predictor: f_theta(x), which returns a class or probability distribution.

A semi-supervised method combines supervised learning on D_L with an additional objective or training procedure that uses D_U.

## Common approaches

### 1. Self-training and pseudo-labeling

1. Train an initial model on labeled examples.
2. Predict labels for unlabeled examples.
3. Keep predictions that pass a confidence rule.
4. Add selected pseudo-labeled examples to training.
5. Retrain or continue training, then repeat cautiously.

A confidence threshold is not a guarantee of correctness. A model can be confidently wrong, particularly on out-of-distribution examples.

### 2. Consistency regularization

A model should often make similar predictions for two meaning-preserving perturbations of the same input. For an image, these might be a crop and a color transform; for text, the transformations must preserve the intended label.

A simplified objective is:

L = L_supervised + lambda * D(f_theta(a1(x)), f_theta(a2(x)))

Here a1 and a2 are augmentations and D measures prediction disagreement. Mean Teacher and FixMatch develop this idea in different ways.

### 3. Graph-based methods

Construct a graph in which similar examples are connected. Labels can be propagated through the graph under the assumption that nearby points tend to share labels. This is useful only when the similarity graph reflects the task; a poor graph can spread errors.

## scikit-learn example

The built-in self-training wrapper uses -1 to mark unlabeled training examples.

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_moons(n_samples=1200, noise=0.25, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)

rng = np.random.default_rng(7)
keep_labeled = rng.random(len(y_train)) < 0.15
y_partial = np.where(keep_labeled, y_train, -1)

model = SelfTrainingClassifier(
    estimator=LogisticRegression(max_iter=1000),
    threshold=0.80,
    criterion="threshold",
    max_iter=10,
)
model.fit(X_train, y_partial)

print("Initially labeled rows:", int(keep_labeled.sum()))
print("Pseudo-labeled rows:", int((model.labeled_iter_ > 0).sum()))
print("Total labeled rows:", int((model.labeled_iter_ >= 0).sum()))
print(classification_report(y_test, model.predict(X_test)))
```

Keep the test set fully labeled and untouched. Compare this approach with a supervised baseline trained on exactly the same labeled subset.

## Evaluation protocol

- Split data before pseudo-labeling. Test examples must never enter the unlabeled pool used during training.
- Keep a labeled validation set for selecting thresholds and stopping rules.
- Compare with a labeled-only baseline and, if possible, a fully supervised reference.
- Inspect pseudo-label precision, not only the number of examples selected.
- Report per-class results and important data slices; confidence can be poorly calibrated.

## Common failure modes

- **Confirmation bias:** early mistakes become training labels and are reinforced.
- **Class imbalance amplification:** the model may select many majority-class examples and few minority-class examples.
- **Distribution mismatch:** unlabeled data may not represent the deployment population.
- **Leakage:** test data or labels influence pseudo-label generation or threshold selection.
- **Overconfidence:** raw softmax scores are not necessarily calibrated probabilities.

## Practice

Use a dataset with 10%, 25%, and 50% labeled training examples. Compare logistic regression trained only on labeled rows with self-training. Track test macro-F1, per-class precision, and the number of pseudo-labels accepted at several thresholds.

## References

- [scikit-learn: SelfTrainingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html)
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
- [A Survey on Semi-Supervised Learning](https://doi.org/10.1109/TKDE.2008.239)

**Key takeaway:** Unlabeled data can improve learning, but only when assumptions behind pseudo-labels or consistency are reasonable and evaluation prevents leakage.
