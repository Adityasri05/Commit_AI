# Bias–Variance Trade-off

Prediction error can arise from different sources. The bias–variance perspective helps explain why a model that is too simple or too flexible may generalize poorly.

## Bias
Bias is error caused by restrictive assumptions. A model with high bias may fail to capture important structure and underfit.

## Variance
Variance describes how much a model's predictions change when trained on different samples. Highly flexible models can have high variance and overfit.

## Typical pattern
- Simple model: often higher bias, lower variance.
- Very complex model: often lower training error, but potentially higher variance.
- Suitable complexity: balances fit and generalization.

The exact decomposition depends on the setting and assumptions; irreducible noise also limits achievable performance.

## How to diagnose
Compare training and validation errors as model complexity changes. Use cross-validation rather than repeatedly tuning against the final test set.

## Ways to respond
- High bias: add useful features, reduce regularization, or choose a more expressive model.
- High variance: collect more data, regularize, simplify the model, or use ensembling.

## Exercise
Fit decision trees with increasing maximum depth. Plot training and validation error and identify where validation performance begins to worsen.