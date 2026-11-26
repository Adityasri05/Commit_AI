# Online Learning and Incremental Model Updates

Online learning updates a model as new observations arrive instead of retraining from scratch on the entire accumulated dataset. It is useful when data arrives continuously, storage is limited, or the relationship between inputs and targets changes over time.

Online learning is not automatically better than periodic batch retraining. It introduces ordering, monitoring, and reproducibility challenges.

## Batch versus incremental training

**Batch learning** fits a model on a prepared dataset. The training set is usually revisited during each training run.

**Incremental learning** updates model parameters using one observation or a small batch at a time. Some libraries expose this through an operation such as partial_fit.

A streaming system often uses mini-batches rather than single records to improve hardware efficiency and reduce noisy updates.

## The prequential evaluation pattern

For a stream of labeled examples, a basic evaluation loop is:

1. Predict the label for the next example using the current model.
2. Record the prediction and its error.
3. Once the true label is available, update evaluation metrics.
4. Train on the example or mini-batch.
5. Continue to the next event.

This is called test-then-train or prequential evaluation. It ensures the model is evaluated before learning from that observation.

If labels arrive late, keep a prediction record keyed by a stable event identifier and score it when the label arrives. Do not update the model with an outcome that is not yet known.

## A simple scikit-learn pattern

~~~python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

model = SGDClassifier(loss="log_loss", random_state=42)
classes = np.array([0, 1])

# Use an initial warm-up batch to fit preprocessing once.
X0, y0 = next(stream)
scaler = StandardScaler().fit(X0)
model.partial_fit(scaler.transform(X0), y0, classes=classes)

for X_batch, y_batch in stream:
    # Keep the feature coordinate system fixed while this model is updated.
    X_scaled = scaler.transform(X_batch)

    # Score before updating with the current batch.
    predictions = model.predict(X_scaled)
    record_metrics(y_batch, predictions)

    # Update only after the labels are available.
    model.partial_fit(X_scaled, y_batch)
~~~

This example deliberately keeps the scaler fixed after warm-up. Updating scaling statistics while retaining the same classifier coefficients changes the feature coordinate system and can make the model's weights inconsistent. If preprocessing must change, version it and retrain or carefully transform the model state; do not silently change it.

## Important design choices

- **Learning rate:** Too high can make updates unstable; too low can adapt slowly.
- **Replay or recent windows:** Reusing selected past examples can reduce catastrophic forgetting, but requires storage and privacy review.
- **Class availability:** Incremental classifiers may need the full class list at initialization.
- **Feature normalization:** Maintain consistent preprocessing statistics and checkpoint them with the model.
- **Delayed labels:** Separate prediction time from label-arrival time.
- **Update cadence:** Per-event, micro-batch, or scheduled updates have different latency and reliability trade-offs.

## Handling changing data

A stream may change because of seasonality, user behavior, policy changes, sensors, or upstream systems. Incremental updates can adapt, but they can also learn transient noise or a malicious data burst.

Keep a stable evaluation window, compare the updated model with a champion baseline, and define rollback conditions. Drift detection is a separate monitoring task; a detected shift should trigger investigation, not an automatic assumption that every incoming example is trustworthy.

## Operational safeguards

1. Validate schemas and ranges before accepting a batch.
2. Track event time, arrival time, and label time separately.
3. Check for duplicates and out-of-order events.
4. Save model and preprocessing checkpoints atomically.
5. Preserve a reproducible update log and training data references.
6. Use a shadow or staged rollout before replacing a production model.
7. Keep a rollback artifact and a batch-trained fallback.

## Common pitfalls

- Evaluating an example after the model has already learned from it.
- Updating on corrupted, duplicated, or adversarial observations.
- Forgetting that a scaler is part of the model state.
- Allowing one class or segment to disappear from the update stream.
- Confusing a change in data distribution with evidence that retraining will improve outcomes.
- Losing reproducibility because update order and random seeds were not recorded.

## Practice

Simulate a binary classification stream whose decision boundary changes halfway through. Compare a frozen model, an incremental SGD classifier, and periodic batch retraining. Use prequential metrics and record both adaptation speed and temporary performance degradation.

## References

- [scikit-learn: Out-of-core learning](https://scikit-learn.org/stable/computing/scaling_strategies.html)
- [River: Online Machine Learning](https://riverml.xyz/)

**Key takeaway:** Incremental learning is a model-lifecycle strategy. It works well only when prediction-before-update evaluation, label timing, preprocessing state, data quality, and rollback are designed explicitly.