# Quantile Regression for Conditional Uncertainty

Ordinary least-squares regression estimates the conditional mean of a target. That is useful when the mean is the quantity of interest, but it can hide asymmetric errors and changing uncertainty. **Quantile regression** estimates a chosen conditional quantile, such as the median or the 90th percentile.

For a feature vector x, the conditional quantile q_tau(x) is defined so that approximately a fraction tau of outcomes fall at or below it, conditional on x.

## Pinball loss

For residual u = y - prediction, the quantile loss is:

    L_tau(u) = tau * u                 when u >= 0
             = (tau - 1) * u          when u < 0

Equivalently, L_tau(u) = u * (tau - I[u < 0]). The asymmetric penalty makes under-prediction more expensive for high quantiles and over-prediction more expensive for low quantiles.

- tau = 0.50 estimates the conditional median.
- tau = 0.10 estimates a lower-tail quantile.
- tau = 0.90 estimates an upper-tail quantile.

A pair of models can form a prediction band. Unlike a conformal interval, quantile regression directly learns how the conditional distribution changes with features; its nominal coverage is not automatically guaranteed.

## Scikit-learn example

~~~python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

lower_model = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.10, max_iter=200,
    learning_rate=0.05, random_state=7
)
median_model = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.50, max_iter=200,
    learning_rate=0.05, random_state=7
)
upper_model = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.90, max_iter=200,
    learning_rate=0.05, random_state=7
)

lower_model.fit(X_train, y_train)
median_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

lower = lower_model.predict(X_test)
median = median_model.predict(X_test)
upper = upper_model.predict(X_test)

print("P90 pinball loss:",
      mean_pinball_loss(y_test, upper, alpha=0.90))
~~~

Check the installed scikit-learn version because estimator parameter names can change across releases. Keep preprocessing inside a pipeline and fit it using training data only.

## Evaluate quantile predictions

For each quantile, report pinball loss. For a lower/upper pair, also measure:

- empirical coverage: fraction of targets between the two predictions;
- average interval width;
- coverage and width across important subgroups or value ranges;
- calibration over time if the data is temporal.

A narrow interval is not useful if it misses too many outcomes. A wide interval may achieve high coverage while providing little decision value.

## Important pitfalls

- **Quantile crossing:** Independently fitted lower and upper models can produce lower(x) > upper(x). Detect it and use a principled monotonicity or joint-modeling strategy rather than silently accepting invalid bands.
- **Small tail samples:** Extreme quantiles need enough data, especially in important subgroups.
- **Distribution shift:** Quantiles can become miscalibrated when the future population changes.
- **Metric mismatch:** RMSE evaluates mean-like point predictions; it does not replace pinball loss for quantiles.
- **False guarantee:** A model trained for the 90th percentile does not guarantee 90% coverage on every future subgroup.

## Practice

Predict delivery times at the 10th, 50th, and 90th percentiles. Plot interval width against distance and compare empirical coverage for short and long routes.

## References

- [Scikit-learn: Quantile regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
- [Koenker: Quantile Regression](https://www.cambridge.org/core/books/quantile-regression/)

**Key takeaway:** Quantile regression models conditional tails and asymmetric uncertainty; evaluate both coverage and sharpness on data that was not used for fitting.