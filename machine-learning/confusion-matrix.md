# Confusion Matrix

A confusion matrix summarizes classification predictions by comparing predicted labels with true labels.

For binary classification:
- TP: positive examples predicted positive.
- TN: negative examples predicted negative.
- FP: negative examples predicted positive.
- FN: positive examples predicted negative.

## Derived metrics
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

Specificity = TN / (TN + FP)

F1 = 2 × Precision × Recall / (Precision + Recall)

Handle zero denominators explicitly; libraries may define a configurable zero-division behavior.

## Example
Suppose TP=40, TN=50, FP=5, FN=5. Accuracy is 90/100 = 0.90, precision is 40/45 ≈ 0.889, and recall is 40/45 ≈ 0.889.

## Why it matters
Accuracy alone can hide poor minority-class performance. The cost of false positives and false negatives depends on the application.

## Exercise
Create a confusion matrix for a fraud detector and explain whether precision or recall should receive more emphasis. State the business cost behind your choice.