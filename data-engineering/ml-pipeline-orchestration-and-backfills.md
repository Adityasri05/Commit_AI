# Orchestration and Backfills for ML Pipelines

An ML pipeline is a sequence of dependent tasks such as extracting data, validating it, building features, training a model, evaluating it, and publishing approved artifacts. Workflow orchestration coordinates when these tasks run, what depends on what, and how failures are handled.

A pipeline should be treated as a reproducible workflow, not merely a collection of scripts executed in a particular order on one laptop.

## Model the workflow as a DAG

A Directed Acyclic Graph (DAG) represents tasks and dependencies. For example:

~~~text
extract -> validate -> build_features -> train -> evaluate -> register
                                \-> data_quality_report
~~~

Independent tasks can run in parallel. A task should start only when its required upstream outputs are ready.

Common orchestration platforms include Apache Airflow, Prefect, Dagster, and managed cloud workflow services. Their APIs differ, but the core design principles are similar.

## Make tasks idempotent

An idempotent task can be retried without causing duplicate or inconsistent side effects. For example, writing a partition to a deterministic location using an atomic replace or merge keyed by a stable identifier is safer than blindly appending rows every time a task retries.

Useful patterns:
- Use a stable run ID and data interval.
- Write outputs to a temporary location, validate them, then publish atomically.
- Record input and output artifact identifiers.
- Use unique constraints or merge keys for database writes.
- Make external side effects explicit and guarded.
- Separate task computation from publication.

Do not assume an orchestrator makes a task idempotent automatically.

## Retries, timeouts, and failure handling

Retries help with transient network or infrastructure failures. They do not fix invalid data, deterministic code bugs, or incompatible schemas.

Define:
- Retry count and exponential backoff.
- Task timeout and overall workflow deadline.
- Failure notifications and ownership.
- Whether downstream tasks should be skipped or run with partial data.
- A recovery procedure for partially published artifacts.

Avoid retry storms against rate-limited APIs. Use bounded retries and respect service retry guidance.

## Backfills

A backfill runs a pipeline for historical data intervals that were missed or need recomputation. It is common after:
- A pipeline outage.
- A corrected transformation.
- A late-arriving data source.
- A new feature definition.
- A model retraining requirement.

Backfills must use the data that was available for the historical interval. Using today's corrected or future information can introduce leakage into historical training examples.

Before a backfill:
1. Identify the exact interval and expected partitions.
2. Confirm the code and schema version to apply.
3. Estimate compute and downstream impact.
4. Decide whether outputs are overwritten, versioned, or merged.
5. Check that retries are safe.
6. Validate counts, freshness, and key metrics after completion.

## Scheduling versus data intervals

A job may run at 02:00 but process the previous day's data. Distinguish the wall-clock execution time from the logical data interval. Use explicit interval boundaries and timezone rules, especially around daylight-saving transitions in global systems.

A successful process exit does not prove that the correct data interval was processed.

## ML-specific artifact lineage

For every training run, record:
- Source dataset snapshot and interval.
- Transformation code version.
- Feature definitions.
- Training configuration and random seeds.
- Model artifact and evaluation report.
- Upstream task run IDs.

This makes it possible to reproduce a model and determine which historical data was used.

## Example design checklist

~~~python
def build_daily_features(interval_start, interval_end, source_version):
    # Read only events belonging to the requested interval.
    events = load_events(
        start=interval_start,
        end=interval_end,
        version=source_version,
    )
    features = transform(events)
    validate_schema(features)
    write_partition_atomically(
        features,
        partition=interval_start.date().isoformat(),
    )
~~~

The functions are placeholders: the important contract is that interval boundaries and source version are explicit, validation happens before publication, and the output partition is deterministic.

## Common pitfalls

- Appending duplicate records on retries.
- Backfilling with current features that were unavailable at historical prediction time.
- Running overlapping backfills that overwrite each other's outputs.
- Treating a task's success status as proof of data correctness.
- Failing to version code, schema, and source snapshots.
- Allowing a model-registration step to run before evaluation has completed.

## Mini-project

Create a daily pipeline with extract, validate, transform, train, and evaluate tasks. Simulate a missed day, backfill it, and intentionally retry one task. Demonstrate that outputs remain correct and that the model artifact records the exact data interval and code version.

## References

- Apache Airflow documentation: https://airflow.apache.org/docs/
- Prefect documentation: https://docs.prefect.io/
- Dagster documentation: https://docs.dagster.io/
