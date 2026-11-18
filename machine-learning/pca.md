# Principal Component Analysis (PCA)

Principal Component Analysis is a linear dimensionality-reduction technique. It finds orthogonal directions, called principal components, that capture as much variance as possible.

## High-level procedure
1. Arrange observations as rows and features as columns.
2. Center each feature using the training-set mean.
3. Often standardize features when scales differ.
4. Compute the covariance matrix or use singular value decomposition (SVD).
5. Order components by explained variance.
6. Project observations onto selected components.

## Explained variance
Each component has an associated variance. The explained-variance ratio indicates the fraction of total variance represented by that component. A cumulative plot can help decide how many components to retain.

## Uses
- Visualizing high-dimensional data in two or three dimensions.
- Compressing correlated features.
- Reducing noise in some settings.
- Making downstream models faster or simpler.

## Limitations
PCA is linear, components may be difficult to interpret, and high variance does not necessarily mean high predictive value. Fit PCA on training data only to avoid data leakage.

## Exercise
Take a dataset with correlated numeric features. Standardize it, fit PCA, plot cumulative explained variance, and compare a classifier trained on original features with one trained on a reduced component set.

**Key idea:** PCA preserves directions of high variance, not necessarily information most useful for a target.