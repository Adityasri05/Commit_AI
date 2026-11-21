# Point-in-Time Correct Features for Machine Learning

A training dataset must represent only the information that would have been available when each prediction was made. If a feature includes information from the future—or information that arrived late—the model can look excellent in offline evaluation and fail in production.

This is a form of **temporal leakage**.

## Two timestamps matter

For time-dependent features, distinguish:

- **Event time:** when something happened in the real world.
- **Availability or ingestion time:** when the system actually knew about it.

A record can describe an event from Monday but arrive on Wednesday. A prediction made Tuesday must not use that record, even though its event timestamp is Monday.

For a prediction at time (t), a feature record should generally satisfy both:

[
event_time leq t
qquad	ext{and}qquad
available_at leq t
]

The exact rule depends on the business definition and source system.

## As-of join pattern

Suppose each prediction row has an entity ID and prediction timestamp. The following SQL pattern selects the most recent eligible feature record for that entity:

```sql
WITH eligible AS (
    SELECT
        p.prediction_id,
        p.entity_id,
        p.prediction_time,
        f.feature_value,
        f.event_time,
        f.available_at,
        ROW_NUMBER() OVER (
            PARTITION BY p.prediction_id
            ORDER BY f.event_time DESC, f.available_at DESC
        ) AS rn
    FROM predictions AS p
    LEFT JOIN feature_events AS f
      ON f.entity_id = p.entity_id
     AND f.event_time <= p.prediction_time
     AND f.available_at <= p.prediction_time
)
SELECT *
FROM eligible
WHERE rn = 1;
```

This is a conceptual as-of join. Real schemas may require additional keys, feature names, validity intervals, tie-breaking rules, and handling for missing history.

## Other temporal leakage traps

- Aggregating a full account history when the prediction was made halfway through that history.
- Using a label-derived status that is updated after the outcome.
- Computing rolling averages without shifting the window.
- Using a corrected historical record that was not available at prediction time.
- Randomly splitting repeated entities across train and test when deployment predicts future events for known or new entities.

For forecasting, define the prediction timestamp, feature lookback window, and target horizon explicitly. A feature window must end no later than the prediction time; the label window begins after it.

## Training-serving parity

Use the same feature definitions and time semantics for offline training and online inference. Store feature timestamps and lineage so a prediction can be reconstructed later. If a feature store supports point-in-time joins, verify the semantics rather than assuming the feature store automatically prevents every form of leakage.

## Practice

Create a small event table with event time and ingestion time. Generate predictions at several timestamps, including one where a late-arriving event has an earlier event time. Compare a naive event-time-only join with the point-in-time-correct join and identify the leaked value.

## References

- [Feast: Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)
- [Google Cloud: Data leakage in ML](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/)

**Key takeaway:** Reconstruct what was knowable at prediction time—not merely what the database contains today.