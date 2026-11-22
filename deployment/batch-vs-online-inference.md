# Batch vs Online Inference

Inference is the process of using a trained model to produce predictions. The right serving pattern depends on freshness, latency, traffic, and cost.

## Batch inference

Batch inference scores many records together on a schedule or when a dataset is ready.

**Good fit:** nightly churn scores, weekly product recommendations, or processing a large backlog.

**Advantages**
- High throughput and efficient accelerator utilization.
- Easier to retry and audit a complete batch.
- Often simpler and cheaper than keeping a service continuously available.

**Trade-offs**
- Predictions may be stale between runs.
- Large jobs can take time and need checkpointing.
- Downstream systems must know which prediction version and timestamp they are reading.

## Online inference

Online inference returns a prediction in response to a request.

**Good fit:** a fraud check during checkout or a recommendation requested when a page opens.

**Advantages**
- Fresh predictions based on the latest eligible features.
- Direct integration with interactive applications.
- Supports per-request explanations or decision metadata.

**Trade-offs**
- Requires availability, low latency, capacity planning, and request validation.
- Traffic spikes can create queueing and tail-latency problems.
- Feature retrieval can dominate model execution time.

## Streaming and micro-batching

Streaming inference processes events continuously. Micro-batching groups events over a short interval to improve throughput while accepting a small delay. It can be a useful middle ground for monitoring or near-real-time scoring.

## Design comparison

| Concern | Batch | Online |
|---|---|---|
| Latency | Minutes to hours may be acceptable | Often milliseconds to seconds |
| Throughput | Large volumes per job | Many independent requests |
| Freshness | Depends on schedule | Can be current if features are current |
| Failure handling | Checkpoints and reruns | Timeouts, retries, fallbacks |
| Cost profile | Compute during jobs | Capacity kept ready for traffic |

## Avoid training-serving skew

Both paths must use compatible feature definitions, preprocessing, and model versions. Store prediction timestamps and model identifiers. For online systems, validate feature freshness and define behavior when a feature service is unavailable.

## Decision checklist

1. How quickly must a prediction be available?
2. How often do inputs change?
3. What is the expected request or record volume?
4. Can the product tolerate stale predictions?
5. What are the availability and latency objectives?
6. How will failed jobs or requests be retried safely?
7. How will outputs be versioned and audited?

## Practice

Design a recommendation system for a college portal. Use batch inference for weekly course suggestions and online inference for a live search query. Identify which features can be precomputed and which must be fetched at request time.

## References

- [Google Cloud: Online prediction](https://cloud.google.com/vertex-ai/docs/predictions/online-predictions)
- [Google Cloud: Batch prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions)

**Key takeaway:** Choose inference architecture from the product's latency and freshness needs, not from a preference for one serving technology.