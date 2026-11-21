# Data Contracts for Machine-Learning Pipelines

A machine-learning pipeline can fail long before model training begins. A source system may rename a column, change a data type, introduce unexpected nulls, or silently change the meaning of a field.

A **data contract** makes the expectations between a data producer and a consumer explicit and testable.

## What a useful contract describes

### Structural rules
- Required column names and data types.
- Nullability and uniqueness constraints.
- Primary keys and relationships.
- Allowed categorical values and schema version.

### Semantic rules
- Units and definitions (for example, income per month rather than per year).
- Valid ranges and business constraints.
- Timestamp timezone and event-time meaning.
- Whether a value is measured, estimated, imputed, or derived.

### Operational rules
- Expected freshness and delivery frequency.
- Minimum volume or completeness.
- Ownership and escalation contact.
- Compatibility policy for schema changes.

A schema that says a column is a float does not explain whether the value is Celsius or Fahrenheit. Semantic definitions matter as much as data types.

## Example with Pandera

```python
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check

schema = DataFrameSchema({
    "customer_id": Column(str, nullable=False, unique=True),
    "age": Column(
        int,
        Check.in_range(0, 120),
        nullable=False,
    ),
    "monthly_income": Column(
        float,
        Check.ge(0),
        nullable=True,
    ),
})

validated = schema.validate(incoming_frame, lazy=True)
```

This is an illustrative batch schema. Adapt data types and constraints to the source system, and decide whether a failed batch should be rejected, quarantined, or partially accepted.

## A practical operating pattern

1. Agree on the meaning and owner of each important field.
2. Version the contract and review breaking changes.
3. Validate incoming data at ingestion or before feature generation.
4. Separate hard failures from warnings and statistical drift alerts.
5. Quarantine invalid records with safe diagnostics rather than silently coercing everything.
6. Monitor freshness, volume, null rates, and category changes.
7. Test the contract in CI using representative fixtures.

## Compatibility and evolution

Adding an optional column may be backward-compatible if consumers ignore unknown fields. Renaming a required column or changing units is usually breaking. Prefer an announced migration period, versioned schemas, and explicit consumer testing.

Do not let a producer silently redefine a feature while the model assumes the old meaning. Such a change can invalidate training data, inference, and monitoring baselines.

## Common mistakes

- Validating only column names and types.
- Treating every outlier as invalid without domain review.
- Automatically filling missing values before recording their rate.
- Logging entire sensitive rows in validation errors.
- Having no owner for failed data checks.
- Updating the schema without versioning or retraining impact analysis.

## Practice

Define a contract for a student-performance dataset with student ID, attendance rate, exam date, and score. Specify units, valid ranges, null rules, freshness, and which changes are breaking. Write tests for both valid and invalid records.

## References

- [Pandera documentation](https://pandera.readthedocs.io/)
- [Great Expectations documentation](https://docs.greatexpectations.io/)

**Key takeaway:** A data contract makes assumptions visible before they become silent model failures. Validate structure, meaning, and operational expectations together.