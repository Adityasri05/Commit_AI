# Feature Stores: Architecture and Training-Serving Parity

A feature store manages reusable feature definitions and helps deliver consistent feature values to both offline model training and online prediction. It is especially useful when multiple models share features or when online inference needs low-latency access to recent entity data.

## Core components

- **Feature definitions:** Versioned transformations, data types, entity keys, and timestamps.
- **Offline store:** Historical feature values for training, analysis, and point-in-time retrieval.
- **Online store:** Latest materialized values optimized for low-latency reads.
- **Registry or catalog:** Metadata about feature views, ownership, freshness, and dependencies.
- **Materialization pipeline:** Computes and publishes values from source data into serving stores.
- **Retrieval API:** Fetches a consistent set of features for a prediction request.

The offline and online stores may use different storage technologies. The goal is semantic consistency, not necessarily identical physical storage.

## Training-serving skew

Training-serving skew occurs when the feature used during training differs from the feature computed or retrieved at inference time. Causes include:
- separate implementations of the same transformation;
- timezone or null-handling differences;
- different aggregation windows;
- stale online materialization;
- late-arriving source events;
- mismatched entity keys or join logic.

A feature store can reduce duplication, but it does not automatically eliminate skew. Feature definitions, timestamps, and materialization behavior still need tests.

## Point-in-time training

Historical training rows must use only feature values that were available at the prediction timestamp. An offline query should perform a point-in-time join, not simply join each entity to its latest record today. This protects against future information and late-arriving data.

## Online retrieval pattern

A request for a prediction might need several features for one customer:

~~~python
# Conceptual pseudocode; adapt to the chosen feature-store SDK.
features = feature_store.get_online_features(
    feature_service="risk_model_v3",
    entity_rows=[{"customer_id": "C-1042"}],
)

prediction = model.predict(features.to_df())
~~~

The exact API differs by platform. Validate that all required feature values are present, have expected types, and meet freshness requirements before invoking the model.

## Reliability and governance checklist

1. Give each feature an owner, definition, entity key, and event-time semantics.
2. Version transformations and record source lineage.
3. Test offline/online parity on the same entity and timestamp.
4. Monitor freshness, missing values, materialization lag, and retrieval latency.
5. Define behavior for missing or stale features: fallback, abstention, or controlled failure.
6. Enforce authorization and data retention rules at both storage layers.
7. Remove obsolete features only after checking model dependencies.

## Common pitfalls

- Assuming a shared feature name guarantees shared semantics.
- Training on current values instead of historical as-of values.
- Ignoring feature freshness when online predictions depend on recent activity.
- Silently replacing missing values with zero when zero has a real meaning.
- Letting feature definitions change without a version or model compatibility plan.

## Practice

Choose three features for a transaction-risk model. Define their source, entity key, event timestamp, lookback window, and null behavior. Create a test that compares offline historical retrieval with online retrieval for the same entity and cutoff time.

## References

- [Feast documentation](https://docs.feast.dev/)
- [Feast: Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)

**Key takeaway:** A feature store is a shared feature-management and serving layer. Its value depends on clear time semantics, parity tests, freshness monitoring, and governance.