# Random Forests

A random forest is an ensemble of decision trees. It combines predictions from many trees to improve robustness over relying on a single tree.

## Main ideas
1. Draw bootstrap samples from training data to build different trees.
2. At each split, consider a random subset of features.
3. Grow trees with controlled settings.
4. Aggregate predictions.

For classification, aggregation commonly uses majority vote or averaged class probabilities. For regression, predictions are commonly averaged.

## Why randomness helps
Individual trees can vary substantially. Training trees on different samples and considering different feature subsets reduces correlation among them. Averaging less-correlated models can reduce prediction variance.

## Important hyperparameters
- n_estimators: number of trees.
- max_depth: maximum depth of each tree.
- max_features: number of features considered at a split.
- min_samples_leaf: minimum observations in a leaf.

## Out-of-bag evaluation
Observations omitted from a tree's bootstrap sample can provide an out-of-bag estimate of performance.

## Cautions
Random forests can be large, are less interpretable than a small tree, and may not extrapolate well beyond the training target range in regression.

## Exercise
Compare a single decision tree with a random forest on the same train/test split. Record accuracy or RMSE, training time, and model complexity.