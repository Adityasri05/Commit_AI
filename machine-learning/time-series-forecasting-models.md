# Time-Series Forecasting Models

Time-series forecasting estimates future values from observations ordered in time. Examples include daily demand, website traffic, energy consumption, sensor readings, and financial measurements.

The forecasting problem must specify the target, forecast horizon, update frequency, and information available at prediction time.

## Start with honest baselines

A complex model should beat a baseline that reflects the time structure:

- **Last-value naive:** Every future value equals the latest observation.
- **Seasonal naive:** Repeat the value from the same season, such as the previous week.
- **Drift:** Extend the average change observed over time.
- **Historical average:** Useful when the series is stable and has no meaningful trend or seasonality.

Baselines expose whether a sophisticated model is learning anything useful.

## Decompose the signal

A series may contain:

- **Trend:** Long-term upward or downward movement.
- **Seasonality:** A repeating pattern with a known period.
- **Cycles:** Longer movements without a fixed seasonal period.
- **Residual variation:** Irregular changes not explained by the model.

A decomposition is a diagnostic view, not proof that the components are independent or stable.

## Exponential smoothing

Exponential smoothing updates estimates by assigning more weight to recent observations.

- **Simple Exponential Smoothing:** Suitable for data without a clear trend or seasonality.
- **Holt's method:** Adds a trend component.
- **Holt-Winters:** Adds seasonal structure, with additive or multiplicative seasonality.

Additive seasonality is often suitable when seasonal swings stay roughly constant. Multiplicative seasonality may be useful when the swing grows with the level, provided the data is positive and the assumption is defensible.

## ARIMA and SARIMA

ARIMA models represent autocorrelation after differencing a series to make its statistical behavior more stable.

    ARIMA(p, d, q)

- p: autoregressive order.
- d: number of differences.
- q: moving-average order.

Seasonal ARIMA adds seasonal orders:

    SARIMA(p, d, q) x (P, D, Q, s)

where s is the seasonal period. ACF and PACF plots can inform candidate orders, but they do not replace validation and residual diagnostics.

Differencing should be used only as needed. Excessive differencing can remove useful structure and create artificial dependence.

## Forecasting with statsmodels

~~~python
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# y is a regularly spaced training Series with a DatetimeIndex.
# Keep the final forecast horizon out of this fit.
model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    initialization_method="estimated",
)
fit = model.fit(optimized=True)
forecast = fit.forecast(steps=14)
print(forecast)
~~~

The seasonal period must match the sampling frequency and domain. A period of seven is only an example for daily data with a weekly pattern.

## Exogenous variables

ARIMAX and regression-based forecasting can use external predictors such as promotions, holidays, or weather. For a future forecast, each predictor must either be known in advance or itself forecast. Using the realized future weather or future sales promotion outcome can create leakage.

## Evaluate forecasts

Use rolling-origin evaluation: fit using data available at a cutoff, forecast the next horizon, move the cutoff forward, and repeat.

Useful metrics include:
- MAE for average absolute error in the target's units;
- RMSE when large errors should receive more weight;
- MASE for comparison against a naive benchmark;
- sMAPE when a scale-free percentage-like metric is useful, while checking behavior near zero.

Report errors by horizon and important time segments. A one-step score does not describe a 30-day forecast.

## Residual diagnostics

Residuals should not retain predictable structure. Inspect residual plots, autocorrelation, bias, and changing variance. Prediction intervals should be checked empirically across forecast origins.

## Common pitfalls

- Randomly shuffling a time series into train and test sets.
- Fitting scalers or transformations using future observations.
- Ignoring missing timestamps and irregular sampling.
- Choosing seasonal periods from intuition without checking the domain.
- Comparing models on different forecast horizons or cutoffs.
- Assuming a good in-sample fit implies good future forecasts.

## Practice

Use a public daily demand dataset. Compare seasonal-naive, Holt-Winters, and SARIMA forecasts with identical rolling origins and a 14-step horizon. Report MAE and MASE by horizon, then inspect residual autocorrelation.

## References

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [statsmodels: Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)

**Key takeaway:** Forecasting quality depends on realistic baselines, time-aware evaluation, and careful handling of seasonality and future information—not simply on choosing a more complicated model.