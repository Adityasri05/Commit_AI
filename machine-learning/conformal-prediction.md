# Conformal Prediction: Prediction Sets with Coverage Guarantees

A conventional classifier returns one label and a regressor returns one number. Conformal prediction wraps a fitted model to produce a **set of plausible labels** or an **interval of plausible values**, with a finite-sample coverage guarantee under an exchangeability assumption.

It is a way to communicate uncertainty without requiring the base model's probabilities to be perfectly calibrated.

## Split conformal regression

Divide labeled data into three disjoint parts:

- **Training:** fit the base prediction model.
- **Calibration:** measure how large its errors are on unseen examples.
- **Test or future data:** construct prediction intervals using the calibration errors.

For calibration example (i), define a nonconformity score such as absolute residual:

[
s_i = |y_i - hat{f}(x_i)|
]

Sort the (n) calibration scores. For target miscoverage (alpha), choose the order statistic at rank:

[
k = minleft(n,leftlceil(n+1)(1-alpha)ightceilight)
]

Let (q) be the score at that rank. The interval for a new example is:

[
[hat{f}(x)-q, hat{f}(x)+q]
]

Under exchangeability of calibration and future examples, this construction provides marginal coverage of at least approximately (1-alpha), with the finite-sample rank handling the correction.

## Minimal implementation

```python
import numpy as np
from math import ceil

# model is fitted on X_train, y_train.
calibration_residuals = np.abs(
    y_calibration - model.predict(X_calibration)
)

alpha = 0.10  # target 90% marginal coverage
n = len(calibration_residuals)
if n == 0:
    raise ValueError("A non-empty calibration set is required")

k = min(n, ceil((n + 1) * (1 - alpha)))
q = np.sort(calibration_residuals)[k - 1]

center = model.predict(X_test)
lower = center - q
upper = center + q
```

The code assumes the calibration set was not used to fit or select the model. For bounded targets, you may optionally clip intervals to known physical limits, but document that post-processing choice.

## Classification sets

For classification, define a nonconformity score from the model's class probabilities. A simple score for the true class is (1 - p(y mid x)). Calibration determines a threshold, and a future example receives every class whose score is below that threshold. More advanced methods, such as adaptive prediction sets, use cumulative probability mass to build sets.

A set can contain one label, several labels, or—in difficult cases—all labels. Larger sets are less decisive but can provide the desired coverage.

## What the guarantee does and does not say

- Coverage is **marginal** over future examples; it does not promise the same coverage for every subgroup or every individual.
- Exchangeability is plausible for randomly sampled i.i.d.-like data, but can fail under temporal, geographic, or behavioral drift.
- A wide interval is an honest signal of uncertainty, not necessarily a poor implementation.
- Coverage alone does not measure usefulness. Also report average interval width or prediction-set size.
- If the model or preprocessing is changed after looking at calibration outcomes, recalibrate using a fresh calibration set.

## Practice

Fit a regression model, reserve a calibration set, and build 90% and 95% intervals. Measure empirical coverage and average width on a separate test set. Repeat the experiment with a time-ordered split and discuss why exchangeability may be less credible.

## References

- [MAPIE documentation](https://mapie.readthedocs.io/)
- [Angelopoulos and Bates: A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)

**Key takeaway:** Conformal prediction turns a model's errors on a held-out calibration set into prediction sets with explicit assumptions and measurable coverage.