# Stacking and Blending Ensemble Methods

Ensembling combines predictions from multiple models. Bagging and boosting are common ensemble families; **stacking** and **blending** go a step further by training a meta-model to combine base-model predictions.

They are useful when different models capture complementary patterns. They are not guaranteed to improve performance: correlated errors, leakage, or a weak validation protocol can make an ensemble worse than its best component.

## Stacking

A stacking system has two levels:

- **Base learners:** Models trained on the original features, such as logistic regression, random forests, and gradient boosting.
- **Meta-learner:** A model trained on predictions from the base learners.

The crucial detail is how meta-training features are generated. For each training row, the base-model prediction used by the meta-learner must come from a model that did not train on that row.

### Out-of-fold predictions

With K-fold cross-validation:

1. Split the training data into K folds.
2. For each fold, train every base model on the other K-1 folds.
3. Predict the held-out fold.
4. Assemble those predictions into an out-of-fold (OOF) matrix.
5. Train the meta-learner on the OOF matrix and original training labels.
6. Refit base models on all training data for inference.

This prevents a meta-learner from seeing unrealistically optimistic in-sample base predictions.

## Blending

Blending usually trains base models on one portion of the training data and trains the meta-model on predictions from a separate holdout portion.

It is simpler to implement but sacrifices some data for the holdout. Stacking uses cross-validation to create OOF predictions and can use the training set more efficiently.

## Scikit-learn example

~~~python
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_models = [
    ("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("forest", RandomForestClassifier(n_estimators=300, random_state=42)),
    ("svc", make_pipeline(StandardScaler(), SVC(probability=True))),
]

model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv,
    stack_method="predict_proba",
    passthrough=False,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
~~~

Keep preprocessing inside each base learner's pipeline so each fold learns transformations only from its training portion. The example uses probability outputs; verify that each estimator supports the selected stack method.

## Design decisions

- **Model diversity:** Prefer learners with complementary strengths rather than many nearly identical models.
- **Meta-learner complexity:** A simple regularized model is often a safer starting point than a high-capacity meta-model.
- **Class probabilities:** Ensure columns correspond to the same class order across estimators.
- **Feature passthrough:** Adding original features to meta-features can help, but increases complexity and overfitting risk.
- **Calibration:** A stacker's probabilities may still need calibration on data separate from model fitting.

## Leakage checklist

- Never create meta-training predictions from a base model trained on the same row.
- Keep the final test set untouched until all choices are fixed.
- Use group-aware or time-aware folds when samples are related or ordered.
- Put feature selection, scaling, imputation, and encoding inside fold-specific pipelines.
- Do not tune the ensemble repeatedly against the final test set.

## How to evaluate

Compare each base model with the stacker using the same splits and metric. For classification, consider macro-F1, ROC-AUC or PR-AUC as appropriate, log loss, and calibration. For regression, compare MAE and RMSE. Also measure inference latency, memory, and operational complexity.

## Mini-project

Build a stack from a linear classifier, random forest, and gradient-boosting model. Compare it with the strongest base learner using repeated stratified cross-validation. Report score variation, training time, inference time, and whether the ensemble's gain justifies the added complexity.

## References

- Wolpert, *Stacked Generalization* (1992): https://doi.org/10.1016/S0893-6080(05)80023-1
- scikit-learn StackingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
