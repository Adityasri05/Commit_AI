# Model Interpretability: Permutation Importance and SHAP

Interpretability methods help investigate how a model uses input features. They answer different questions and should not be mistaken for causal explanations.

## Permutation importance

1. Measure a model's baseline score on held-out data.
2. Shuffle one feature column, breaking its association with the target and other columns.
3. Recompute the score.
4. The score decrease estimates how much the fitted model relies on that feature for this dataset and metric.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    scoring="f1", n_repeats=20, random_state=42
)
for idx in result.importances_mean.argsort()[::-1]:
    print(feature_names[idx], result.importances_mean[idx])
```

Use representative held-out data. Correlated features can substitute for one another, making individual permutation scores appear small.

## SHAP at a glance

SHAP assigns feature contributions to predictions using a game-theoretic framework. Values explain a prediction relative to a chosen background/reference distribution and depend on the explainer and feature-dependence assumptions.

## Important cautions

- Importance is association with model behavior, not proof of causality.
- Global importance can hide subgroup differences.
- Explanations can be unstable for correlated features.
- A plausible explanation does not establish that a model is fair or safe.
- Document the model version, sample, metric, and method.

## Exercise

Compute permutation importance for a classifier using accuracy and then F1. Compare the rankings and explain why a feature can matter differently under different scoring objectives.

**Takeaway:** Use explanations as diagnostic evidence alongside domain knowledge, subgroup evaluation, and error analysis.