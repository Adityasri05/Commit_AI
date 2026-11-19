# Probability and Statistics Foundations for Machine Learning

Probability describes uncertainty; statistics uses observed data to reason about populations and unknown quantities. Both are essential for evaluating models and interpreting predictions.

## Core ideas

- **Random variable:** A variable whose value depends on a random outcome.
- **Distribution:** Describes the possible values and their probabilities.
- **Mean and variance:** Summarize location and spread, but can be sensitive to outliers.
- **Conditional probability:** Probability of A given B, written P(A | B).
- **Bayes' theorem:** P(A | B) = P(B | A)P(A) / P(B), when P(B) > 0.
- **Sampling:** A sample is only informative about a population when its collection process is understood.

## Why base rates matter

Suppose a condition affects 1% of a population. A screening test has 90% sensitivity and 95% specificity. In 10,000 people, about 100 have the condition: 90 test positive. Of 9,900 without it, 495 test positive. Thus only 90 of 585 positive results are true positives (about 15.4%). A seemingly accurate test can have a low positive predictive value when the event is rare.

## Confidence intervals

A confidence interval is produced by a procedure that, over repeated samples under its assumptions, covers the true parameter at the stated rate. It is not a probability statement about a fixed parameter after one interval has been computed.

## Student exercise

1. Define precision and recall in terms of conditional probabilities.
2. Explain why a non-representative sample can bias a model regardless of sample size.
3. Recalculate the screening example if specificity rises to 99%.

**Takeaway:** Always consider the data-generating process and base rates before interpreting a metric or probability.