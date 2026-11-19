# Model Monitoring in Production

A deployed model can degrade as user behavior, data sources, and operating conditions change. Monitoring helps detect problems after deployment.

## What to monitor
- Service health: latency, throughput, error rate, availability.
- Data quality: missing values, schema changes, invalid ranges.
- Input drift: changes in feature distributions.
- Prediction behavior: output distributions and confidence patterns.
- Model quality: performance metrics when ground-truth labels become available.
- Fairness and safety: relevant group-level outcomes and harmful failure modes.

## Drift concepts
- Data drift: input distribution changes.
- Concept drift: the relationship between inputs and targets changes.
- Performance degradation: observed predictive quality worsens.

Drift signals are alerts for investigation, not automatic proof that a model is unusable.

## Operational practices
1. Log model version and relevant metadata.
2. Define service-level objectives.
3. Establish baseline distributions and metrics.
4. Set alert thresholds based on impact.
5. Investigate alerts and document decisions.
6. Maintain rollback and retraining procedures.

## Exercise
Design a monitoring dashboard for a student performance prediction service. Include service, data-quality, prediction, and delayed accuracy metrics.