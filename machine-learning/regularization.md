# Regularization and Overfitting

Overfitting occurs when a model captures noise or accidental patterns in its training data and performs poorly on unseen data.

Regularization discourages unnecessarily complex models or parameter values.

## L1 and L2 regularization

A common objective is:

Loss_total = Loss_data + λ × Penalty(parameters)

- **L1 (Lasso-style):** adds the sum of absolute parameter values. It can drive some weights exactly to zero, creating sparse models.
- **L2 (Ridge-style):** adds the sum of squared parameter values. It shrinks weights smoothly toward zero.

The coefficient λ controls the strength of the penalty. A value that is too large can cause underfitting.

## Other ways to reduce overfitting
- Collect more representative training data.
- Use cross-validation for model selection.
- Reduce model complexity.
- Apply data augmentation when valid for the task.
- Use early stopping based on validation performance.
- Use dropout in suitable neural networks.

## Learning curves
Compare training and validation performance as training data or training duration increases:
- Very good training score but much worse validation score often indicates overfitting.
- Poor performance on both often indicates underfitting or insufficient features.

## Practical caution
Do not repeatedly tune against the test set. Doing so leaks information from the test set into model selection.

## Exercise
A model has 99% training accuracy and 72% validation accuracy. List three possible interventions and describe how you would verify whether each helps.