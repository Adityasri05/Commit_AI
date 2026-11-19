# Time-Series Validation: Respecting Time Order

Time-series forecasting predicts future observations from past information. Unlike ordinary independent-row problems, randomly shuffling timestamps can leak future patterns into training and produce unrealistic scores.

## Basic forecasting split

- Train on the earliest period.
- Validate on a later period for model selection.
- Test on the most recent untouched period for a final estimate.

For example, use January–September for training, October–November for validation, and December for testing. Choose windows that reflect the deployment horizon and available history.

## Rolling-origin evaluation

A rolling-origin strategy repeatedly trains on past data and evaluates on the next time window. Expanding-window evaluation keeps all past observations; sliding-window evaluation uses a fixed recent history. The right choice depends on whether older data remains relevant.

## Avoid leakage in features

A feature for time t must use only information available at prediction time. Rolling averages should be shifted when necessary; future target values, revised reports, and post-event metadata must not enter the feature set.

## Example split

```python
# df is sorted by timestamp; target is the value to forecast.
train = df[df["timestamp"] < "2025-10-01"]
valid = df[
    (df["timestamp"] >= "2025-10-01") &
    (df["timestamp"] < "2025-12-01")
]
test = df[df["timestamp"] >= "2025-12-01"]
```

Use actual dates appropriate to the dataset. If multiple entities have their own timelines, ensure the evaluation matches the intended forecasting setup.

## Baselines and metrics

Compare against simple baselines such as the last observed value or seasonal naive forecast. MAE is interpretable in target units; RMSE penalizes large errors more. MAPE is problematic when actual values are zero or near zero.

## Exercise

For daily electricity demand, design a one-week-ahead evaluation. State the forecast origin, training window, validation windows, and naive baseline.

**Takeaway:** A valid forecasting score must simulate the information available at the moment each forecast is made.