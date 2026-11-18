# Decision Trees

A decision tree predicts an output through a sequence of feature-based tests. Internal nodes contain tests, branches represent outcomes, and leaf nodes contain predictions.

## How a tree is built
At each step, an algorithm searches for a split that makes resulting groups more useful or pure.

Common classification criteria:
- Gini impurity measures class mixture at a node.
- Entropy measures uncertainty in a class distribution.
- Information gain is the reduction in entropy after a split.

Regression trees often choose splits that reduce squared error or variance within groups.

## Strengths
- Small trees can be easy to explain.
- Can model nonlinear decision boundaries.
- Usually needs little feature scaling.
- Many implementations handle mixed feature types.

## Limitations
- Deep trees can overfit.
- Small data changes can produce different trees.
- Greedy split selection does not guarantee a globally optimal tree.

## Regularization
Limit maximum depth, minimum samples per leaf, or minimum samples required to split. Tune settings with validation data or cross-validation.

## Exercise
Design a small tree to classify whether a student should carry an umbrella using forecast probability and cloud cover. Identify the root split and explain your choice.