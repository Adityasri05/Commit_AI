# Supervised vs. Unsupervised Learning

## Supervised learning
A supervised dataset contains input features X and a target y. The model learns a mapping from inputs to known targets.

**Typical tasks**
- Classification: predict a discrete label, such as whether an email is spam.
- Regression: predict a numeric value, such as delivery time.

**Example:** Given house area, location, and room count with known sale prices, train a regression model to estimate a new house's price.

## Unsupervised learning
An unsupervised dataset contains features but no supplied target labels. The goal is to discover structure.

**Typical tasks**
- Clustering: group similar observations, such as customer segments.
- Dimensionality reduction: represent data with fewer features, such as PCA.

## Comparison

| Aspect | Supervised | Unsupervised |
|---|---|---|
| Target labels | Available | Not provided |
| Goal | Predict target | Discover structure |
| Example algorithms | Linear regression, decision trees, SVM | K-means, hierarchical clustering, PCA |
| Evaluation | Compare predictions with labels | Use internal metrics and domain validation |

## Student exercise
1. Predicting student exam scores from study hours: which paradigm?
2. Grouping news articles by similarity without categories: which paradigm?
3. Explain why clustering results still need human interpretation.

**Key idea:** Choose the learning paradigm based on the problem and available labels, not on which algorithm is most popular.