# Partial Dependence and ICE: Inspecting Feature Effects

Partial dependence plots (PDPs) and individual conditional expectation (ICE) plots help visualize how a fitted model's predictions change as one or two features vary.

They describe the behavior of the **model**, not a causal effect in the real world.

## Partial dependence

For feature (X_j), a partial dependence function averages predictions while replacing that feature with a chosen value (z):

[
PD(z) = rac{1}{n}sum_{i=1}^{n}
hat{f}(z, x_{i,-j})
]

Here, (x_{i,-j}) represents all other features from row (i). The plot shows the average model response as (z) changes.

## ICE curves

An ICE curve performs the same replacement for one row at a time rather than averaging all rows. Plotting many ICE curves reveals whether the feature effect differs across individuals.

- Similar, nearly parallel ICE curves suggest a fairly consistent effect in the model.
- Curves that cross or have different slopes can indicate interactions or heterogeneous model behavior.
- A PDP can hide these patterns because averaging may cancel them out.

## Scikit-learn example

```python
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X_reference,
    features=["age", "income"],
    kind="both",       # show PDP and ICE
    subsample=100,
    random_state=42,
)
plt.tight_layout()
plt.show()
```

Use a representative reference dataset and a fitted estimator that supports the inspection API. For classification, choose the response method deliberately (for example, a probability for the positive class) and state what the vertical axis represents.

## The correlation problem

PDP creates synthetic combinations by pairing a selected feature value with other features from existing rows. When features are strongly correlated, many combinations may be unrealistic—for example, a very young age paired with a retirement status that is common only among older people.

The model may be asked to predict in regions with little or no training support. The resulting curve can be misleading even if the plotting code is correct.

ICE does not remove this problem: each row is still evaluated after one feature is changed while the others are held fixed.

## Good interpretation habits

1. Inspect feature distributions and correlations before reading the plot.
2. Mark regions with sparse data and avoid strong claims there.
3. Compare PDP with ICE to look for interactions.
4. Use domain constraints when choosing valid feature ranges.
5. Compare plots across important subgroups when appropriate.
6. Treat the plot as a model diagnostic, not proof that changing a real-world feature will cause the prediction to change.

For strongly dependent features, consider conditional approaches or accumulated local effects (ALE), which reduce some extrapolation problems.

## Practice

Train a model on a dataset with correlated numeric features. Plot PDP and ICE for each feature, identify a region where the PDP may combine unrealistic values, and compare the result with an ALE plot if available.

## References

- [Scikit-learn: Partial dependence and ICE](https://scikit-learn.org/stable/modules/partial_dependence.html)
- [Molnar: Interpretable Machine Learning — Partial Dependence](https://christophm.github.io/interpretable-ml-book/pdp.html)

**Key takeaway:** PDP communicates an average model response; ICE exposes variation hidden by that average. Both require careful attention to the data distribution.