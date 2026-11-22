# Stream Processing: Event Time, Windows, and Watermarks

Streaming systems process records as they arrive. Arrival order is not always the same as the order in which events happened, so a reliable pipeline must define which notion of time it uses.

## Event time vs processing time

- **Event time:** when the event occurred in the source domain.
- **Processing time:** when the stream processor handled the record.
- **Ingestion time:** when the system received or stored the record.

A mobile device may upload an event several minutes late. If a dashboard groups solely by processing time, the event may appear in the wrong reporting interval.

## Window types

- **Tumbling window:** fixed, non-overlapping intervals, such as every five minutes.
- **Sliding window:** overlapping intervals, such as a ten-minute window updated every minute.
- **Session window:** groups events separated by no more than a configured inactivity gap.

Choose a window based on the question being answered. A rolling fraud signal and a daily sales report may need different window semantics.

## Watermarks and late data

A watermark is the system's estimate that event-time progress has advanced far enough that older events are unlikely to arrive. It allows the processor to emit results for event-time windows while balancing completeness and latency.

A watermark is not a guarantee that no older event will ever arrive. Systems need an explicit late-data policy:

- Drop events beyond an allowed lateness threshold.
- Update previously emitted results when late events arrive.
- Send very late events to a correction or side-output workflow.

The correct policy depends on whether downstream consumers can handle revised results.

## Delivery and correctness

Distributed stream processors may replay records after failures. Design consumers to be idempotent where possible, use stable event IDs for deduplication, and understand the source and sink guarantees.

Exactly-once claims are end-to-end properties, not magic words. A processor may provide exactly-once state updates while an external API call still happens twice if retries are not protected.

## Example scenario

A campus attendance service receives events from phones. A student scans a QR code at 09:02, but the device reconnects at 09:12 and uploads the event. The event belongs to the 09:00 event-time window even though it is processed later. The system might update the attendance count if the window allows late corrections.

## Operational checklist

1. Define the authoritative event timestamp and timezone.
2. Monitor event-time lag and out-of-order arrival.
3. Choose window and allowed-lateness policies.
4. Track duplicates and invalid timestamps.
5. Make writes idempotent or deduplicate with stable IDs.
6. Test replay, backfill, and late-arriving records.
7. Document whether downstream reports can change after publication.

## Practice

Generate events with event times out of order and processing times in arrival order. Compute a five-minute tumbling window, then simulate a watermark and a late event. Compare drop-late and update-late policies.

## References

- [Apache Flink: Time and Order in Streams](https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/)
- [Apache Beam: Watermarks and late data](https://beam.apache.org/documentation/programming-guide/#watermarks-and-late-data)

**Key takeaway:** A stream is not just a sequence of arrivals. Define event-time semantics, lateness, and replay behavior before trusting its aggregates.