# Bootstrap Confidence Intervals for Model Metrics

A test score is an estimate, not an exact description of future performance. A confidence interval communicates how much the estimate could vary because the evaluation sample is finite.

The **bootstrap** approximates this sampling variability by repeatedly drawing samples with replacement from the observed evaluation data and recomputing the metric.

## Basic procedure

For a fixed model and an independent test set:

1. Store each test row's true label and prediction together.
2. Draw a bootstrap sample of row indices with replacement, usually the same size as the test set.
3. Compute the metric on that resampled set.
4. Repeat many times (for example, 1,000–5,000 replicates).
5. Use chosen quantiles of the bootstrap metric distribution as an interval.

Keep each row's truth and prediction paired. Resampling them independently would destroy the relationship being evaluated.

## Example: bootstrap F1 interval

```python
import numpy as np
from sklearn.metrics import f1_score

def bootstrap_metric_interval(y_true, y_pred, metric, *,
                              n_boot=2000, seed=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Truth and predictions must be non-empty and aligned")

    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        scores.append(metric(y_true[indices], y_pred[indices]))

    return np.quantile(scores, [0.025, 0.975]), np.asarray(scores)

interval, samples = bootstrap_metric_interval(
    y_test, y_pred,
    lambda truth, pred: f1_score(truth, pred, zero_division=0),
)
print("95% percentile interval:", interval)
```

This is a percentile bootstrap interval. It is a useful teaching baseline, not a universal best interval for every metric or sample size.

## Comparing two models fairly

When models A and B predict the **same** test examples, use the same bootstrap indices for both models. For every replicate, calculate:

[
Delta_b = M_b(A) - M_b(B)
]

Then inspect the interval for the differences. This paired comparison preserves the fact that both models were tested on the same rows and is usually more informative than comparing two unrelated intervals.

## Important choices and limitations

- **Small or rare classes:** Some ordinary bootstrap samples may contain no positive examples, making metrics such as recall undefined. A stratified bootstrap can preserve observed class counts, but it answers a conditional question and should be chosen deliberately.
- **Grouped observations:** If rows from one person, household, device, or session are correlated, resample groups rather than individual rows.
- **Time series:** Independent row resampling breaks temporal structure. Consider block bootstrap methods or a time-aware evaluation design.
- **Model-training uncertainty:** Bootstrapping fixed test predictions captures test-sample variability, not the variability caused by retraining the model. To include training variability, the full training procedure must be repeated.
- **Distribution shift:** An interval cannot guarantee performance on a future population that differs from the test sample.
- **Repeated experimentation:** Trying many models and reporting the most favorable interval introduces selection bias.

A 95% interval does not mean there is a 95% probability that this particular fixed parameter lies inside the computed interval. It describes the long-run coverage behavior of the procedure under its assumptions.

## Practice

Evaluate two classifiers on the same held-out data. Bootstrap their paired F1 difference, plot the difference distribution, and explain what conclusions are supported if the interval crosses zero.

## References

- [SciPy: Bootstrap confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Efron and Tibshirani: An Introduction to the Bootstrap](https://doi.org/10.1201/9780429246593)

**Key takeaway:** Report uncertainty alongside a metric, and resample at the unit that was independently sampled.