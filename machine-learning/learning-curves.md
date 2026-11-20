# Learning Curves for Diagnosing Model Performance

A learning curve plots model performance against the amount of training data. It helps distinguish data limitations from model-capacity or optimization issues.

## Typical patterns
- **High bias:** Training and validation scores are both poor and close together. A more expressive model or better features may help.
- **High variance:** Training performance is strong while validation performance is much worse. More data, regularization, or a simpler model may help.
- **Data-limited model:** Validation performance keeps improving as training size grows; additional representative data may be valuable.

Interpret curves according to the metric direction: some metrics are better when higher, while loss metrics are better when lower.

## How to build one
1. Choose a suitable cross-validation strategy.
2. Evaluate increasing fractions of the training data.
3. Record training and validation scores for each size.
4. Plot the mean and variability across folds.
5. Repeat after changing one modeling choice.

## Cautions
Learning curves can be noisy on small datasets. For grouped or time-dependent observations, use group-aware or chronological splits rather than random folds.

## Exercise
Generate learning curves for a decision tree and a regularized logistic regression model. Compare the gap and the trend, then propose one targeted next experiment.

**Key point:** A learning curve is a diagnostic, not a guarantee that adding data will solve every problem.