# Change Data Capture for ML Pipelines

Change Data Capture (CDC) records changes made to a source database so downstream systems can react without repeatedly copying the entire table. It is useful for keeping feature pipelines, analytics stores, search indexes, and operational data products up to date.

## How CDC works

A CDC system commonly reads a database's transaction log or change stream and emits events for inserts, updates, and deletes. A change event may contain:

- Source database and table.
- Primary key of the changed row.
- Operation type: create, update, or delete.
- Before and/or after values, depending on connector configuration.
- Transaction or log position, timestamp, and schema metadata.

A typical pipeline is:

    Database transaction log
        -> CDC connector
        -> durable event broker
        -> validation and transformation
        -> feature / analytics store
        -> downstream ML consumers

Log-based CDC is often less intrusive than frequent full-table polling, but connector behavior and database retention settings matter.

## Snapshot plus streaming

A new consumer often needs both historical state and future changes:

1. Take a consistent initial snapshot of the required rows.
2. Record the log position associated with the snapshot.
3. Consume changes from that position.
4. Apply the snapshot and stream without losing or double-applying changes.

The exact handoff is connector-specific. Understand its snapshot mode, offsets, transaction boundaries, and recovery behavior before relying on it.

## Delivery semantics and idempotency

Many event systems provide at-least-once delivery. A consumer may see the same event more than once after retries or recovery. Design updates to be idempotent:

- Use a stable event ID or source primary key plus log position.
- Store the last applied offset or version.
- Ignore duplicates that have already been applied.
- Preserve ordering where updates to one entity require it.
- Handle delete events explicitly, often with a tombstone or deletion marker.

Do not claim end-to-end exactly-once processing merely because one broker supports a transactional feature. The guarantee must cover the source, connector, broker, consumer, and destination.

## Schema evolution

Source tables change. A new column, renamed field, changed type, or altered meaning can break downstream features.

- Version schemas and define compatibility rules.
- Validate events before transformation.
- Decide how consumers handle unknown fields.
- Quarantine invalid events with enough metadata for investigation.
- Coordinate backfills when a feature definition changes.
- Avoid silently changing the meaning of an existing feature column.

## CDC for ML feature pipelines

CDC can maintain near-real-time features such as order counts or account status. Pay attention to:
- Event identity and entity keys.
- Event time versus ingestion time.
- Late and out-of-order updates.
- Deletes and corrections.
- Point-in-time correctness for training data.
- Offline/online feature consistency.
- Privacy requests and deletion propagation.

CDC does not automatically make a feature historically correct. Training data still needs point-in-time joins so a row only uses information available at the prediction timestamp.

## Example event

```json
{
  "op": "u",
  "source_table": "orders",
  "primary_key": {"order_id": 812},
  "log_position": "00000016:000004A0",
  "after": {"order_id": 812, "status": "shipped"}
}
```

This is an illustrative event shape, not a universal connector schema. Real connectors use their own envelope and offset formats.

## Practice

Design a CDC pipeline that updates a customer's rolling order-count feature. Specify snapshot behavior, event key, duplicate handling, delete semantics, schema evolution, and how you would rebuild the offline training table without future-data leakage.

## References

- [Debezium Documentation](https://debezium.io/documentation/)
- [Apache Kafka: Message Delivery Semantics](https://kafka.apache.org/documentation/#semantics)

**Key takeaway:** CDC turns database changes into a stream, but reliable ML consumers still need snapshot coordination, idempotency, schema management, delete handling, and point-in-time correctness.