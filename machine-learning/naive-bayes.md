# Naive Bayes Classifiers

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem and a conditional-independence assumption between features given the class.

## Bayes' theorem
P(C|X) = P(X|C)P(C) / P(X)

A classifier predicts the class with the largest posterior probability.

## Why the assumption is called naive
The model commonly assumes that features are conditionally independent given the class. This assumption is often unrealistic, but the classifier can still work well in practice.

## Common variants
- Gaussian Naive Bayes: models continuous features with class-conditional Gaussian distributions.
- Multinomial Naive Bayes: commonly used with count features, such as word counts.
- Bernoulli Naive Bayes: designed for binary features.

## Laplace smoothing
For count-based models, smoothing prevents unseen feature events from receiving zero probability. A common additive smoothing parameter is alpha.

## Strengths and limitations
Naive Bayes is fast, simple, and often effective for text classification. Its probability estimates may be poorly calibrated, and correlated features can violate its assumptions.

## Exercise
Build a spam classifier using a bag-of-words representation and Multinomial Naive Bayes. Compare its precision and recall with a TF-IDF plus logistic regression baseline.