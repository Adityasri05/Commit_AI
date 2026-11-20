# Experiment Tracking for Machine Learning Projects

Experiment tracking records what changed between model runs so results can be reproduced and compared.

## What to record
- Dataset version or immutable data reference.
- Train/validation split strategy and random seed.
- Feature and preprocessing configuration.
- Model type, hyperparameters, and software versions.
- Metrics, plots, and evaluation artifacts.
- Code revision and experiment timestamp.

Never log credentials, private records, or raw sensitive data merely for convenience.

## A simple workflow
1. Define a baseline experiment.
2. Track configuration and metrics for every run.
3. Save the best model with its preprocessing pipeline.
4. Record the evaluation dataset and decision threshold.
5. Compare runs using the same split and metric definitions.
6. Promote a model only after reproducible validation.

## Tools
MLflow, Weights & Biases, and similar platforms can store parameters, metrics, artifacts, and run metadata. A CSV or JSON log can be enough for a small learning project, provided it is versioned and consistently structured.

## Reproducibility checklist
- Can another person identify the exact data and code?
- Are random seeds and library versions recorded?
- Is the final metric tied to a held-out evaluation?
- Can the saved artifact reproduce predictions?

## Exercise
Run three model configurations, log parameters and validation metrics, and create a short comparison table. Include the code commit and dataset version for each run.

**Key point:** A metric without experiment context is difficult to trust or reproduce.