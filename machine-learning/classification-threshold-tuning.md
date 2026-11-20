# Choosing a Classification Threshold

Many classifiers output a score or probability. The default threshold of 0.5 is a convention, not a universal optimum.

## Confusion-matrix trade-offs
- Lowering the threshold usually increases positive predictions, often improving recall while reducing precision.
- Raising it usually reduces false positives but can miss more true positives.
- The best operating point depends on the costs, capacity, and consequences of errors.

## A disciplined selection process
1. Train the model using training data.
2. Generate scores on a validation set or out-of-fold predictions.
3. Evaluate candidate thresholds using the business-relevant metric or explicit cost function.
4. Choose the threshold before opening the final test set.
5. Freeze the threshold and evaluate once on the test set.
6. Monitor performance and prevalence after deployment.

## Cost-based example
If missing a positive case is much more expensive than reviewing a false alarm, a lower threshold may be appropriate. If human review capacity is limited, precision or the number of alerts per day may constrain the choice.

## Common mistakes
- Choosing a threshold on the test set.
- Optimizing accuracy on highly imbalanced data.
- Assuming predicted scores are calibrated probabilities.
- Reporting only one metric without the confusion matrix.

## Exercise
Plot precision, recall, and F1 across thresholds. Mark the chosen operating point and explain why it matches the application's error costs.

**Key point:** The model produces scores; the application chooses an operating policy.