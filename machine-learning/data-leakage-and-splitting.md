# Leakage-Safe Dataset Splitting

A model must be evaluated on examples that were not used to fit its parameters or make data-dependent preprocessing decisions. **Data leakage** occurs when information unavailable at prediction time influences training or evaluation, producing an overly optimistic score.

## Choose a split that matches the problem

- **Random split:** Reasonable for independent, identically distributed rows.
- **Stratified split:** Preserves approximate class proportions; useful for imbalanced classification.
- **Group split:** Keeps related entities (such as multiple visits from one patient) in one partition.
- **Time-based split:** Trains on the past and evaluates on later observations when predicting the future.

Never randomly distribute near-duplicate records or records from the same person across train and test if that would not reflect deployment.

## Fit preprocessing only on training data

Imputers, scalers, encoders, feature selectors, and dimensionality reducers learn statistics or mappings. Fit them on training data only, then apply the fitted transformations to validation and test data. A scikit-learn Pipeline helps enforce this rule during cross-validation.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## Leakage checklist

1. Is every feature available at the moment a prediction is made?
2. Are groups or timestamps respected?
3. Were preprocessing steps fitted only on training folds?
4. Was the test set kept untouched during model selection?
5. Were target-derived columns or post-outcome fields removed?

## Exercise

Imagine predicting whether a student will pass an exam. Would final exam marks be a valid feature before the exam? Explain why a random split might also be misleading if the same student has several records.

**Takeaway:** A carefully designed split is part of the problem definition, not an afterthought.