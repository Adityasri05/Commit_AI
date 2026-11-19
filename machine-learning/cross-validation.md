# Cross-Validation

Cross-validation estimates how a model may perform on unseen data by repeating training and validation across different splits.

## K-fold procedure
1. Divide the training data into K folds.
2. Train on K−1 folds.
3. Validate on the remaining fold.
4. Repeat until every fold has served as validation data.
5. Summarize the scores, often with a mean and standard deviation.

## Variants
- Stratified K-fold preserves approximate class proportions for classification.
- Group K-fold keeps related samples together.
- Time-series validation respects chronological order and avoids training on the future.

## When to use
Cross-validation is useful for model selection and hyperparameter tuning, especially when data is limited.

## Important cautions
- Keep a final test set separate when possible.
- Fit scaling, imputation, feature selection, and other learned preprocessing inside each training fold, preferably using a pipeline.
- Choose splits that reflect how the model will be used in production.

## Exercise
Compare 5-fold cross-validation scores for a baseline classifier and a tuned classifier. Report mean and standard deviation, then evaluate the chosen model once on a held-out test set.