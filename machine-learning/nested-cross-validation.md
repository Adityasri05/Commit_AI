# Nested Cross-Validation for Honest Model Selection

Ordinary cross-validation is useful for estimating model performance, but using the same folds both to select hyperparameters and to report the selected model's score can produce an optimistic estimate. Nested cross-validation separates those two jobs.

## The two loops

- **Inner loop:** compares models or hyperparameter settings using only the current outer-training portion.
- **Outer loop:** evaluates the configuration selected by the inner loop on data that was not used for that selection.

For each outer fold:

1. Hold one fold aside as the outer validation fold.
2. Run the hyperparameter search on the remaining outer-training data. Its inner cross-validation must stay entirely inside that data.
3. Refit the selected pipeline on all outer-training data.
4. Score it once on the outer validation fold.
5. Repeat for every outer fold and summarize the outer scores.

The mean outer score estimates the performance of the *selection procedure*, not one final model trained on every available row. The spread across outer folds is useful context, but folds are not fully independent samples.

## Scikit-learn pattern

Pass a complete preprocessing-and-model pipeline to the inner search. This keeps learned transformations inside each inner training fold.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=11)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=29)

search = GridSearchCV(
    estimator=pipeline,  # preprocessing + classifier
    param_grid={
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    scoring="f1",
    cv=inner_cv,
    refit=True,
)

results = cross_validate(
    search,
    X_train,
    y_train,
    cv=outer_cv,
    scoring={"f1": "f1", "recall": "recall"},
    return_train_score=False,
)

print("Outer-fold F1:", results["test_f1"])
print("Mean F1:", results["test_f1"].mean())
```

Define `pipeline`, `X_train`, and `y_train` for the dataset before running the example. The search is cloned and fitted independently within each outer fold.

## When it is worth the cost

Nested CV is especially useful when datasets are small and the reported score will support a model comparison or research conclusion. It can be expensive: with 5 outer folds and 4 inner folds, many candidate configurations are fitted repeatedly.

For a large dataset, a carefully separated training, validation, and final test workflow may be more practical. Do not repeatedly inspect the final test score while changing the model.

## Split design still matters

- Use stratified folds when preserving class proportions is appropriate.
- Use group-aware folds when several rows belong to the same person, device, patient, or session.
- Use chronological splits for forecasting; never let a future observation train a model evaluated on the past.
- Keep all learned preprocessing, feature selection, and resampling inside the inner pipeline.

## Common mistakes

- Tuning on all data before starting the outer loop.
- Applying feature selection once to the complete dataset.
- Using random folds for grouped or time-ordered observations.
- Reporting the best inner-fold score as the final unbiased estimate.
- Treating the outer-fold mean as a guarantee for every future population.

## Practice

Compare a fixed baseline with a tuned pipeline using nested CV. Report every outer-fold score, the mean, the standard deviation, the selected hyperparameters per fold, and the total number of model fits.

**Key takeaway:** The inner loop chooses; the outer loop evaluates that choice. Keeping those roles separate reduces selection bias.

## References

- [Scikit-learn: Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Scikit-learn: Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
