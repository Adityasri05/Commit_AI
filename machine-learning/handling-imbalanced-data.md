# Handling Imbalanced Classification Data

A dataset is imbalanced when one class appears much less often than another. A model can achieve high accuracy by mostly predicting the majority class while missing the cases that matter.

## Diagnose before changing the data
- Inspect class counts and proportions in every split.
- Use a stratified split for independent, identically distributed classification data.
- Check whether imbalance reflects the real deployment population.
- Identify the cost of false positives and false negatives.

## Better evaluation
Report the confusion matrix, per-class precision and recall, F1-score, balanced accuracy, and PR-AUC when the positive class is rare. ROC-AUC can still be useful, but precision-recall behavior is often more informative under severe imbalance.

## Training strategies
- **Class weights:** Penalize mistakes on minority examples more heavily.
- **Random oversampling:** Repeat minority examples; can overfit.
- **Random undersampling:** Remove majority examples; may discard useful information.
- **Synthetic sampling:** Methods such as SMOTE create interpolated minority examples; apply only to training folds.

## Leakage-safe experiment
Put resampling inside an imbalanced-learn or scikit-learn pipeline so it runs only on each training fold. Keep validation and test distributions untouched.

## Practical exercise
Compare a baseline, class-weighted model, and training-only resampling. Select the approach using a metric tied to the real error costs.

**Key point:** Do not balance the test set just to make scores look better. Evaluate on data representative of deployment.