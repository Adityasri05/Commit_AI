# Hyperparameter Tuning Without Overfitting the Validation Set

Model parameters are learned from data; **hyperparameters** are choices made before or around training, such as tree depth, regularization strength, or learning rate. Tuning searches for settings that perform well on data not used to fit those parameters.

## A disciplined workflow

1. Define a baseline and a task-appropriate metric.
2. Choose a search space based on plausible values.
3. Use cross-validation on the training portion.
4. Compare candidates using the same folds and scoring rule.
5. Refit the selected pipeline on all training data.
6. Evaluate once on the untouched test set.

Repeatedly changing the search after inspecting test performance turns the test set into a validation set.

## Grid versus random search

- **GridSearchCV:** Evaluates every specified combination; useful for small, deliberate grids.
- **RandomizedSearchCV:** Samples combinations from distributions; often more efficient for larger spaces.
- **Successive halving / Bayesian optimization:** Can allocate resources more adaptively, but add complexity.

## Example

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(StandardScaler(), SVC())
search = RandomizedSearchCV(
    pipe,
    param_distributions={
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.01, 0.1, 1],
        "svc__kernel": ["rbf"]
    },
    n_iter=8,
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

## Report more than the winner

Record search space, metric, fold strategy, compute budget, mean and spread of scores, and the final test result. If many configurations are tried on a small dataset, selection itself can overfit noisy validation scores.

## Exercise

Tune a decision tree's maximum depth and minimum leaf size. Compare the best cross-validation score with training score and explain what a large gap suggests.

**Takeaway:** Tuning is model selection; preserve an untouched test set for the final estimate.