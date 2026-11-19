# Anomaly Detection: Finding Unusual Observations

Anomaly detection identifies observations that differ from expected patterns. Examples include unusual network traffic, defective manufactured parts, or suspicious transactions. An anomaly is not automatically an error or fraud; it is a candidate for investigation.

## Problem settings

- **Supervised:** Labeled examples of normal and anomalous behavior are available.
- **Semi-supervised / one-class:** Training data mainly represents normal behavior.
- **Unsupervised:** The algorithm searches for unusual structure without trusted labels.

## Common methods

- **Z-score:** Measures distance from the mean in standard-deviation units; useful for simple, roughly well-behaved univariate data.
- **Isolation Forest:** Randomly partitions features; unusual points tend to be isolated in fewer splits.
- **Local Outlier Factor (LOF):** Compares local density with that of nearby observations.
- **One-Class SVM:** Learns a boundary around a region of normal examples; scaling and parameter selection matter.

## Isolation Forest example

```python
from sklearn.ensemble import IsolationForest

detector = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)
detector.fit(X_train_normal)
labels = detector.predict(X_new)  # 1 = inlier, -1 = outlier
scores = detector.decision_function(X_new)
```

Fit on a suitable reference dataset. The contamination setting or score threshold should reflect the use case; the default is not a universal anomaly rate.

## Evaluation

When labels exist, use precision, recall, PR curves, and the cost of missed anomalies. Without labels, inspect flagged cases with domain experts, test stability, and monitor false-alert burden. Avoid evaluating only on the same data used to fit the detector.

## Exercise

Generate a two-dimensional dataset with a dense cluster and a few distant points. Compare Isolation Forest and LOF, plot their scores, and explain why each method may flag different observations.

**Takeaway:** Anomaly scores prioritize review; they do not establish intent or wrongdoing.