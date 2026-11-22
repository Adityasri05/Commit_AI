# Dataset Versioning and Lineage for Machine Learning

A model result is difficult to reproduce if the exact training data cannot be identified. Dataset versioning records which data snapshot, transformations, labels, and splits produced an experiment.

Versioning is more than saving a CSV with a date in its filename.

## What to record

A useful dataset manifest can include:

- Dataset name and immutable version or snapshot identifier.
- Storage URI and content checksum for files or partitions.
- Schema version, column definitions, units, and label meaning.
- Source systems and extraction timestamp.
- Transformation code revision and configuration.
- Train, validation, and test split identifiers or a reproducible split seed.
- Row counts, time range, and quality-check results.
- Data owner, access classification, and retention policy.

A checksum can detect whether bytes changed, but it does not explain why they changed. Pair hashes with human-readable metadata and lineage.

## A simple manifest

    dataset: student-outcomes
    version: 2026-11-22-v1
    schema_version: 2
    source_snapshot: warehouse://analytics/student_outcomes/snapshot-184
    transform_commit: 8f2a1c7
    split_seed: 42
    train_rows: 8000
    validation_rows: 1000
    test_rows: 1000
    label: passed_course
    created_at: 2026-11-22T10:00:00Z

This is an illustrative YAML-like record. Real manifests should also identify the data files or partitions and their checksums.

## Common approaches

- **Git plus small files:** appropriate for tiny public teaching datasets and configuration, not large private datasets.
- **DVC:** stores data references and metadata alongside code while keeping large artifacts in configured storage.
- **Object-store snapshots or table formats:** useful for large partitioned datasets and time-travel queries.
- **LakeFS or similar versioned data layers:** can provide branch-like workflows for data changes.

Choose a method that supports access control, retention, and the data size involved.

## Lineage through the ML lifecycle

Track the chain:

    source data -> cleaning -> feature generation -> split -> training run -> model artifact -> deployment

Each arrow should identify code and configuration versions. A model registry entry should point back to the dataset and training run that produced it.

## Common mistakes

- Overwriting a dataset in place and calling it the same version.
- Recording a random seed but not the split logic or source snapshot.
- Storing credentials or personal data in Git.
- Assuming a file hash alone makes a dataset understandable.
- Keeping test labels in a broadly accessible location during model development.
- Forgetting that schema or label-definition changes can invalidate comparisons.

## Practice

Create two versions of a small tabular dataset. Change one source row and one transformation rule. Produce manifests with checksums, compare the versions, and explain which model experiments must be rerun.

## References

- [DVC documentation](https://dvc.org/doc)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)

**Key takeaway:** Reproducible ML requires a traceable data snapshot and transformation history, not just a saved model file.