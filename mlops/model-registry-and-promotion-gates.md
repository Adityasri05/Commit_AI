# Model Registries and Safe Promotion Gates

A trained model should not move directly from a notebook into production without traceability and validation. A model registry provides a catalog of model versions and the metadata needed to manage their lifecycle.

## What a registry entry should contain

- Model name and immutable version.
- Artifact location and checksum.
- Source-code commit and training configuration.
- Dataset snapshot and feature definitions.
- Evaluation metrics and test-set identity.
- Dependency and runtime information.
- Model signature: expected inputs and outputs.
- Creation time, owner, and approval history.

The registry does not prove a model is good. It makes versions discoverable and connects artifacts to evidence.

## A promotion workflow

A common lifecycle is:

1. **Candidate:** training completed and artifact registered.
2. **Validation:** schema, quality, robustness, fairness, and operational checks run.
3. **Staging:** candidate runs in an environment close to production.
4. **Canary or shadow:** compare behavior with the current model under controlled traffic or mirrored requests.
5. **Production:** promote only after defined gates pass.
6. **Rollback or archive:** retain a known-good version and document why a version was replaced.

Names for stages vary by platform. Focus on the evidence and permissions behind each transition.

## Example promotion gates

For a student-support classifier, gates might require:

- Validation F1 above a pre-agreed minimum.
- No unacceptable recall regression for a high-impact class.
- Input schema and inference contract tests pass.
- P95 latency stays within the service objective.
- Group-level error review is complete.
- A rollback artifact and owner are recorded.

Thresholds must be selected for the use case. Do not invent a universal F1 or latency cutoff.

## Safe deployment patterns

- **Shadow:** send a copy of production inputs to the candidate but do not use its predictions for user decisions.
- **Canary:** send a small, controlled fraction of eligible traffic to the candidate and monitor outcomes.
- **Blue-green:** keep two deployment environments and switch traffic after checks.

A canary is not automatically safe for high-impact decisions. Use approvals and impact controls appropriate to the application.

## Common mistakes

- Registering a model without the dataset or code lineage.
- Overwriting a model version in place.
- Promoting based on one aggregate metric.
- Forgetting dependency compatibility or feature-schema changes.
- Having no rollback owner or tested rollback procedure.
- Confusing registry status with actual deployment status.

## Practice

Design a registry record and promotion checklist for a house-price model. Include artifact version, training-data snapshot, evaluation evidence, runtime signature, staging checks, and rollback plan.

## References

- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [Google Cloud: MLOps continuous delivery and automation](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

**Key takeaway:** A registry gives models identity and lineage; promotion gates provide evidence that a particular version is ready for a particular environment.