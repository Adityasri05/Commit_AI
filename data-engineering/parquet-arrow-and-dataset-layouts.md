# Parquet, Arrow, and Efficient ML Dataset Layouts

Machine-learning pipelines often spend more time reading and moving data than training a small model. File format and dataset layout affect scan cost, memory use, parallelism, and reproducibility.

## Row-oriented versus columnar storage

A row-oriented format stores fields for one record together. A columnar format stores values from the same column together. Apache Parquet is a compressed columnar file format designed for analytical workloads.

Columnar storage is useful when a job reads only a subset of many columns. Parquet also organizes data into row groups and column chunks, enabling readers to skip irrelevant data when statistics and filters allow it.

## Apache Arrow

Arrow is a language-independent columnar in-memory format. It supports efficient interchange between tools and can reduce serialization overhead when systems share compatible Arrow buffers. Parquet is primarily a persistent file format; Arrow is an in-memory representation. They complement one another.

## A small PyArrow example

~~~python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({
    "user_id": [101, 102, 103],
    "country": ["IN", "US", "IN"],
    "event_count": [4, 8, 2],
})

pq.write_table(table, "events.parquet", compression="zstd")

# Read only the columns needed by a training job.
subset = pq.read_table(
    "events.parquet",
    columns=["user_id", "event_count"]
)
print(subset.schema)
print(subset.to_pandas())
~~~

For larger datasets, use a dataset scanner and filters so the reader can apply projection and predicate pushdown rather than loading every file and filtering afterward.

## Partitioning

A directory layout such as:

    events/year=2026/month=11/country=IN/part-000.parquet

can help skip partitions when a query filters on year, month, or country. Choose partition columns based on common filters and reasonable cardinality.

Avoid partitioning by highly unique values such as user ID. It can create thousands or millions of tiny directories and files, increasing metadata and scheduling overhead.

## Practical design choices

- Use stable schemas and explicit data types.
- Choose row-group sizes based on scan patterns and parallelism.
- Compress to reduce storage and I/O; benchmark CPU cost as well.
- Keep file sizes large enough to avoid small-file overhead but manageable for parallel reads.
- Store dataset version, source snapshot, schema version, and generation time.
- Apply access controls to both files and partition metadata.
- Validate null rates, ranges, uniqueness, and row counts after writes.

## Common pitfalls

- Reading all columns when only a few are required.
- Assuming partition pruning works when filters do not match partition columns.
- Creating a huge number of tiny files.
- Changing a column's type without a schema migration plan.
- Treating a successful file write as proof that the dataset is complete or correct.
- Forgetting that partition values can reveal sensitive attributes.

## Practice

Generate a synthetic event dataset with several months and countries. Write it as one large Parquet file and as partitioned files. Compare bytes scanned and wall-clock read time for a month/country filter, then inspect the schema and row counts.

## References

- [Apache Parquet documentation](https://parquet.apache.org/docs/)
- [Apache Arrow documentation](https://arrow.apache.org/docs/)
- [PyArrow Parquet guide](https://arrow.apache.org/docs/python/parquet.html)

**Key takeaway:** Efficient ML data access depends on columnar formats, sensible partitioning, stable schemas, and measured file sizes—not just compression.