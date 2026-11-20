# Missing Data: Diagnose Before Imputing

Missing values can arise from data-entry errors, unavailable measurements, or a process that selectively omits information. The reason for missingness affects the safest strategy.

## Understand the pattern
- **MCAR:** Missingness is independent of observed and unobserved values.
- **MAR:** Missingness depends on observed variables.
- **MNAR:** Missingness depends on the missing value itself or other unobserved factors.

These assumptions cannot usually be proven from the observed data alone; use domain knowledge and sensitivity analysis.

## Practical options
- Drop rows or columns when missingness is limited and removal is defensible.
- Use median or mean imputation for numeric features as a simple baseline.
- Use most-frequent or an explicit category for categorical features.
- Add a missingness indicator when the fact that a value is absent may carry signal.
- Consider model-based or multiple imputation for more demanding analyses.

## Leakage-safe implementation
Fit imputers inside a pipeline using training folds only. Do not calculate medians or modes on the full dataset before splitting.

## Checklist
1. Measure missingness by column and by target class.
2. Inspect whether missingness has a meaningful operational cause.
3. Compare a simple baseline with indicator-based alternatives.
4. Evaluate on untouched validation/test data.
5. Document assumptions and limitations.

## Exercise
Create a missingness summary, train median-imputation and indicator-based pipelines, and compare validation performance.

**Key point:** Imputation is an assumption about unseen values, not a neutral cleanup step.