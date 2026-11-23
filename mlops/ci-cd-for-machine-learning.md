# CI/CD for Machine Learning Systems

Continuous integration and continuous delivery (CI/CD) apply software-engineering discipline to ML code and deployment. ML pipelines add extra concerns: datasets, feature definitions, model artifacts, evaluation evidence, and changing data distributions.

CI/CD does not mean retraining a production model on every code push. It means making changes testable, traceable, and promotable through controlled stages.

## Separate the pipeline stages

### 1. Continuous integration

Run checks whenever relevant code changes:

- Formatting, linting, type checks, and dependency checks.
- Unit tests for preprocessing, feature logic, and metric calculations.
- Data-schema and input-contract tests.
- Small smoke tests for training and inference.
- Security checks for dependencies and secrets.

Keep CI fast. Use small fixtures for ordinary pull requests and reserve expensive training for appropriate triggers.

### 2. Training and validation pipeline

A versioned training run should record:
- Source-code commit.
- Dataset snapshot or immutable data reference.
- Feature and preprocessing versions.
- Hyperparameters, random seeds, and runtime environment.
- Model artifact checksum.
- Evaluation metrics and evaluation-data identity.

Run data validation before training. Compare the candidate with a baseline on a held-out evaluation set. Use task-specific quality, robustness, fairness, and operational checks rather than a single universal threshold.

### 3. Continuous delivery

If all required checks pass:
1. Register the candidate artifact and metadata.
2. Deploy to a staging environment.
3. Run integration, load, and compatibility tests.
4. Obtain any required human approval.
5. Roll out gradually using a deployment strategy suitable for the risk.
6. Monitor and keep a tested rollback path.

A deployment should reference an immutable artifact version, not "whatever model was trained most recently."

## Example pipeline outline

```text
Pull request
  -> lint and type checks
  -> unit tests
  -> schema and feature-contract tests
  -> small training smoke test

Approved merge or scheduled data trigger
  -> validate versioned dataset
  -> train candidate
  -> evaluate against baseline
  -> register artifact and lineage
  -> deploy to staging
  -> integration and load tests
  -> approval and controlled rollout
  -> monitor, rollback if required
```

The exact triggers depend on the product. A batch model may retrain on a schedule; a model with a stable dataset may only retrain after an approved code or data change.

## Test levels for ML

- **Unit:** deterministic feature transformations and utility functions.
- **Contract:** input schema, output schema, feature names, and model signature.
- **Data validation:** ranges, null rates, uniqueness, and distribution checks.
- **Model evaluation:** quality on a fixed, protected evaluation set.
- **Integration:** API, feature store, and storage compatibility.
- **Operational:** latency, throughput, resource use, and rollback behavior.

Avoid flaky tests that compare floating-point results with exact equality. Use tolerances and document nondeterministic hardware behavior where relevant.

## Common mistakes

- Training and evaluating on the same data.
- Running long training jobs on every tiny documentation change.
- Losing the dataset version behind a model artifact.
- Promoting based only on training loss.
- Allowing an unreviewed metric threshold to deploy a high-impact model.
- Treating successful deployment as proof of model quality.

## Practice

Create a GitHub Actions workflow that runs linting, unit tests, and a small training smoke test on pull requests. Add a separate manually approved job that evaluates a candidate artifact and deploys only after required checks pass.

## References

- [Google Cloud: MLOps continuous delivery and automation](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

**Key takeaway:** CI/CD for ML connects code, data, evaluation, artifacts, and deployment. Automate repeatable checks, preserve lineage, and keep promotion decisions evidence-based.