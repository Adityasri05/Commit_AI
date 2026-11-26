# K-Nearest Neighbors and Distance-Based Learning

K-Nearest Neighbors (KNN) is a supervised learning method that predicts an output from the labels or values of nearby training examples. It is a simple baseline for classification and regression, and it makes the geometry of a dataset directly affect predictions.

## How KNN works

For a query point:

1. Compute its distance to the training observations.
2. Select the k closest observations.
3. For classification, use a majority vote or distance-weighted vote.
4. For regression, average or distance-weight the neighbors' target values.

KNN is a lazy learner: it stores the training examples and performs much of its work at prediction time rather than learning a compact parametric model during fitting.

## Distance metrics

- **Euclidean distance:** Straight-line distance; common for continuous features on comparable scales.
- **Manhattan distance:** Sum of absolute coordinate differences; can be useful when movement along each dimension is more meaningful.
- **Minkowski distance:** A family that includes Manhattan and Euclidean distance.
- **Cosine distance:** Compares vector direction and is often used for normalized text embeddings.
- **Hamming distance:** Counts differing positions for binary or categorical representations.

The right metric depends on feature meaning. Encoding a category as an arbitrary integer and using Euclidean distance can create a false notion of closeness.

## Feature scaling and weighting

A feature measured in thousands can dominate one measured between zero and one. Fit scaling on training data only, then apply the same transformation to validation and test examples.

KNN can also use distance-weighted voting so that nearer observations contribute more. This may help when the local neighborhood has a meaningful density gradient, but it can amplify noisy or duplicated points.

## Practical scikit-learn example

~~~python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=7, weights="distance", p=2)
)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
~~~

In the Minkowski metric, p=2 gives Euclidean distance and p=1 gives Manhattan distance. Tune k, p, and the weighting strategy using cross-validation inside the training data.

## The curse of dimensionality

As dimensions increase, distances can become less informative: nearest and farthest observations may have increasingly similar distances. Irrelevant features can overwhelm the geometry.

Possible responses include:
- selecting meaningful features;
- reducing dimensionality with a fitted transformation;
- using domain-specific distance functions;
- comparing against models that do not rely on local distance.

Dimensionality reduction must be fitted within each training fold to avoid leakage.

## Scaling to larger datasets

A brute-force query compares a new point with many stored observations. Tree-based indexes can help in some low-dimensional settings, but their advantage often weakens in high dimensions. Approximate-nearest-neighbor methods trade exactness for speed and are covered separately in retrieval systems.

## Common pitfalls

- Choosing k using the test set.
- Failing to scale continuous features.
- Treating arbitrary category codes as meaningful distances.
- Ignoring class imbalance in majority voting.
- Assuming a larger k is always more stable or accurate.
- Measuring only fit time while ignoring prediction latency and memory.

## Practice

Create a two-dimensional classification dataset with noise. Plot decision regions for k=1, k=5, and k=25. Compare uniform and distance-weighted voting, then add irrelevant features to observe how neighborhood quality changes.

## References

- [scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

**Key takeaway:** KNN is easy to understand, but its quality depends on meaningful feature geometry, careful scaling, a justified distance metric, and prediction-time cost.