# Domain Adaptation and Covariate Shift

A model can perform well on its training distribution and still fail after deployment because the target population, sensors, language, workflow, or environment differs. **Domain adaptation** studies how to transfer predictive behavior from a labeled source domain to a different target domain.

This is related to transfer learning, but the emphasis here is the mismatch between source and target distributions and the assumptions required to correct it.

## Source and target domains

- Source domain: distribution P_s(X, Y), often with many labeled examples.
- Target domain: distribution P_t(X, Y), where labels may be scarce or delayed.
- Goal: achieve useful target-domain performance without assuming the two distributions are identical.

First identify what changed. Different shifts require different remedies.

| Shift | What changes | Typical response |
|---|---|---|
| Covariate shift | P(X) changes, while P(Y given X) is assumed stable | Importance weighting, representative target sampling |
| Label shift | P(Y) changes, while P(X given Y) is assumed stable | Prior correction using class-conditional estimates |
| Concept shift | P(Y given X) changes | New labels, adaptation, or model redesign |
| Feature/schema shift | Measurement or feature definitions change | Repair the pipeline before statistical adaptation |

These assumptions are not interchangeable. For example, weighting for covariate shift cannot repair a changed relationship between features and labels.

## Covariate-shift importance weighting

Under the covariate-shift assumption, source examples can be weighted by:

w(x) = p_t(x) / p_s(x)

The weighted source risk estimates target risk when the conditional label distribution is stable and target support is covered by source support.

A practical density-ratio estimate can be built by training a classifier to distinguish source rows from target rows. If the domain classifier was trained with equal source and target sample counts, then:

w(x) is approximately P(domain=target given x) / P(domain=source given x)

With unequal sample counts, include the source-to-target sample-count correction. Clip extreme weights and inspect their distribution; a few huge weights can make training unstable.

## Example workflow

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# X_source_unlabeled and X_target_unlabeled have the same feature schema.
X_domain = np.vstack([X_source_unlabeled, X_target_unlabeled])
domain_y = np.r_[
    np.zeros(len(X_source_unlabeled), dtype=int),
    np.ones(len(X_target_unlabeled), dtype=int),
]

domain_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
domain_model.fit(X_domain, domain_y)

p_target = domain_model.predict_proba(X_source_train)[:, 1]
p_source = 1.0 - p_target

# Correct for unequal domain sample counts.
ratio = (p_target / np.clip(p_source, 1e-6, None))
ratio *= len(X_source_unlabeled) / len(X_target_unlabeled)

weights = np.clip(ratio, 0.1, 10.0)
weights /= weights.mean()

predictor.fit(X_source_train, y_source_train, sample_weight=weights)
```

This is a starting point, not a universal recipe. The source and target features must be comparable, the domain classifier must be evaluated, and the target test labels must remain untouched. Some estimators do not support sample weights.

## Other adaptation strategies

- **Fine-tuning:** Use a small labeled target set to update some or all model parameters.
- **Feature alignment:** Learn representations that reduce domain-specific differences while retaining task-relevant information.
- **Domain-specific normalization:** Adapt normalization statistics when measurement conditions change.
- **Target self-training:** Use carefully filtered target pseudo-labels; this inherits the confirmation-bias risks of semi-supervised learning.
- **Domain generalization:** Train across several source environments to improve robustness to an unseen target, without using target data during training.

## Diagnostics before adapting

1. Verify that feature definitions, units, missing-value handling, and label meanings are consistent.
2. Compare source and target feature distributions and important subgroups.
3. Train a domain classifier; unusually high discrimination suggests a measurable domain gap.
4. Check whether target examples lie outside source support.
5. Obtain a small, representative, labeled target evaluation set whenever possible.
6. Compare the adapted model with the unchanged source model and a simple target baseline.

## Common mistakes

- Calling every distribution change covariate shift.
- Assuming importance weighting fixes concept drift.
- Evaluating on the same target labels used to tune adaptation.
- Ignoring extreme weights and low effective sample size.
- Using a target sample that does not represent production traffic.
- Forgetting that adaptation can improve an average metric while harming a particular subgroup.

## Practice

Train a classifier on one region or device type and evaluate it on another. Measure a domain classifier's AUC, compare feature distributions, and test importance weighting only after stating the covariate-shift assumption. Report target performance and subgroup metrics.

## References

- [A Tutorial on Domain Adaptation](https://arxiv.org/abs/0907.1815)
- [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526)
- [scikit-learn: Importance weighting with sample_weight](https://scikit-learn.org/stable/glossary.html#term-sample_weight)

**Key takeaway:** Diagnose which part of the data-generating process changed before choosing an adaptation method. Statistical correction is only as reliable as its assumptions and target-domain evaluation.
