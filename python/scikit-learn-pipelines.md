# Scikit-learn Pipelines: A Reproducible ML Workflow

A pipeline chains preprocessing and an estimator into one object. It reduces training-serving inconsistency and helps prevent leakage during cross-validation.

## Why pipelines help
- Fit transformations only on training data.
- Keep preprocessing and prediction steps together.
- Tune preprocessing and model parameters in one search.
- Make cross-validation code easier to audit and reproduce.

## Example
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric, numeric_columns),
    ("cat", categorical, categorical_columns),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
```

Define `numeric_columns` and `categorical_columns` from the dataset schema before running the example.

## Tuning
Use parameter names such as `classifier__C` in GridSearchCV or RandomizedSearchCV. Pass the full pipeline to the search object so each fold learns preprocessing only from its training portion.

## Exercise
Compare a manually preprocessed workflow with a pipeline under cross-validation. Confirm that unseen categories at inference do not crash the encoder.

**Key point:** Split first, then fit the pipeline. Never fit preprocessing on the complete dataset before splitting.