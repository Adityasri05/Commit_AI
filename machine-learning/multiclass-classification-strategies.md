# Multiclass Classification Strategies

Binary classification predicts one of two classes. Multiclass classification predicts one label from three or more mutually exclusive classes—for example, classifying a flower as setosa, versicolor, or virginica.

The strategy affects training cost, probability interpretation, and how errors are evaluated.

## Three common approaches

### 1. Native multinomial classification

A model produces one score per class and applies a softmax to obtain probabilities:

    P(y = k | x) = exp(z_k) / sum_j exp(z_j)

The probabilities sum to one. Multinomial logistic regression learns class scores jointly and is a useful baseline when its assumptions and feature representation are reasonable.

### 2. One-vs-rest (OvR)

Train one binary classifier per class. Class k is positive; every other class is negative. At prediction time, compare the class scores.

OvR is easy to understand and works with estimators that only support binary classification. Scores from independently trained classifiers may not be calibrated or directly comparable, so do not assume they form a valid probability distribution.

### 3. One-vs-one (OvO)

Train a classifier for every pair of classes. With K classes, this creates K(K-1)/2 classifiers. Each classifier sees only the two classes in its pair; predictions are combined by voting or an estimator-specific rule.

OvO can work well with some algorithms, but the number of models grows quadratically with the number of classes.

## Scikit-learn examples

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Native multinomial-capable baseline
softmax_model = LogisticRegression(max_iter=1000)
softmax_model.fit(X_train, y_train)

# Explicit one-vs-rest wrapper
ovr_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)
ovr_model.fit(X_train, y_train)

# SVC can use a one-vs-one decision structure internally
ovo_model = SVC(decision_function_shape="ovo")
ovo_model.fit(X_train, y_train)

print(classification_report(y_test, softmax_model.predict(X_test)))
```

The exact behavior depends on the estimator and scikit-learn version. Check the estimator documentation before interpreting decision scores as probabilities.

## Evaluate beyond overall accuracy

- **Per-class precision and recall:** reveals which classes are confused.
- **Macro average:** computes a metric per class and gives each class equal weight.
- **Weighted average:** weights each class by its support.
- **Micro average:** aggregates decisions across classes before computing the metric.
- **Confusion matrix:** shows the direction of class-to-class mistakes.
- **Top-k accuracy:** useful when a ranked shortlist is acceptable, but not a replacement for top-1 accuracy in every application.

For imbalanced data, macro-F1 and per-class recall can reveal failures hidden by a high accuracy score. Use stratified splits when observations are independent and the class counts permit it; use group- or time-aware splits when the data structure requires them.

## Practical checklist

1. Confirm whether labels are mutually exclusive or multilabel.
2. Establish a simple baseline.
3. Inspect class counts and the cost of each error.
4. Compare native, OvR, or OvO only when there is a reason.
5. Tune thresholds or calibrate probabilities on validation data if decisions require them.
6. Report per-class metrics and the confusion matrix.

## Practice

Use a three-class dataset. Compare multinomial logistic regression with OvR. Inspect per-class recall and the confusion matrix, then explain whether the aggregate metric hides a weak class.

## References

- [Scikit-learn: Multiclass classification](https://scikit-learn.org/stable/modules/multiclass.html)
- [Scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

**Key takeaway:** Choose a multiclass strategy based on the estimator and task, then evaluate every class—not just the overall accuracy.