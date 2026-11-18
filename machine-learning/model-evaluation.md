# Model Evaluation: Metrics and Data Splits

A model should be evaluated on data that was not used to fit its parameters. This estimates how well it may generalize to new examples.

## Dataset splits
- **Training set:** used to learn model parameters.
- **Validation set:** used for model selection and hyperparameter tuning.
- **Test set:** held back for a final, relatively unbiased evaluation.

For small datasets, cross-validation can provide a more stable estimate. Keep related or time-ordered observations together when random splitting would leak information.

## Classification metrics

For binary classification:
- True positive (TP): positive example predicted positive.
- False positive (FP): negative example predicted positive.
- True negative (TN): negative example predicted negative.
- False negative (FN): positive example predicted negative.

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall / (Precision + Recall)

Accuracy is the fraction of all predictions that are correct. It can be misleading when classes are imbalanced.

## Regression metrics
- MAE: average absolute prediction error.
- MSE: average squared prediction error; penalizes large errors more strongly.
- RMSE: square root of MSE, in the target's units.
- R-squared: compares model error with a mean-prediction baseline.

## Practical example
For disease screening, missing a true case may be more costly than a false alarm. Recall may therefore be especially important, alongside precision and clinical review.

## Checklist
1. Choose metrics based on real-world costs.
2. Compare with a simple baseline.
3. Inspect errors, not just one aggregate score.
4. Tune only on training/validation data.
5. Evaluate the final model once on held-out test data.

**Exercise:** Construct a confusion matrix for 80 correct predictions, 10 false positives, and 10 false negatives. Calculate precision, recall, and accuracy.