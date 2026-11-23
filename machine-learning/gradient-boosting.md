# Gradient Boosting for Tabular Machine Learning

Gradient boosting builds an ensemble sequentially. Each new weak learner is trained to improve the current ensemble, rather than training every tree independently and averaging them as in a random forest.

## The core idea

Start with a simple prediction, such as the mean target for regression. At each boosting stage:

1. Compute how the current model is wrong under the chosen loss.
2. Fit a small learner—commonly a shallow decision tree—to the negative gradient of that loss.
3. Add the learner's contribution to the ensemble, scaled by a learning rate.
4. Repeat for a fixed number of stages or stop when validation performance no longer improves.

For squared-error regression, this resembles fitting residuals. For other losses, the target is a loss-dependent pseudo-residual, not necessarily the raw difference between target and prediction.

A simplified additive model is:

    F_m(x) = F_(m-1)(x) + learning_rate * h_m(x)

where h_m is the learner added at stage m.

## Important controls

- **Number of estimators:** more stages can improve fit, but may overfit or increase training time.
- **Learning rate:** smaller steps often need more estimators; tune the pair together.
- **Tree depth / leaf count:** controls how complex each stage can be.
- **Subsampling:** stochastic gradient boosting trains each stage on a sample and can improve generalization.
- **Early stopping:** monitor a validation set and stop when improvement stalls.

Modern implementations such as histogram-based gradient boosting bin continuous features to make split search more efficient.

## Scikit-learn example

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=300,
    max_leaf_nodes=15,
    early_stopping=True,
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

Keep the test set untouched until model selection is complete. If preprocessing is needed, fit it only on training data, preferably inside a scikit-learn pipeline.

## Random forest vs. gradient boosting

| Random forest | Gradient boosting |
|---|---|
| Trees are built independently, often in parallel | Trees are added sequentially |
| Bootstrap sampling and feature randomness reduce correlation | Each stage focuses on current loss |
| Averaging often gives a robust baseline | Can achieve strong tabular accuracy with careful tuning |
| Usually less sensitive to learning-rate choices | Learning rate and number of stages interact |

Neither method wins on every dataset. Compare them with the same split strategy, metric, and tuning budget.

## Common pitfalls

- Tuning against the test set repeatedly.
- Using deep trees and too many stages without validation.
- Ignoring class imbalance or the cost of different errors.
- Assuming feature importance proves causation.
- Forgetting that categorical-feature support differs by implementation.

## Practice

Train a random forest and a histogram gradient-boosting model on the same tabular dataset. Compare macro-F1, ROC-AUC where appropriate, training time, inference latency, and calibration. Record the split strategy and all important hyperparameters.

## References

- [Scikit-learn: Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Friedman, Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

**Key takeaway:** Gradient boosting is stage-wise optimization with weak learners. Tune it using validation data and compare it against simpler baselines.