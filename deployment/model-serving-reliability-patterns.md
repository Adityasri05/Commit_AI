# Reliable Model Serving Patterns

A prediction endpoint is a distributed service. Even when the model code is correct, slow dependencies, traffic spikes, malformed inputs, cold starts, and partial outages can make the system unreliable.

Design the serving layer so failures are bounded, visible, and recoverable.

## Health checks have different jobs

- **Liveness:** is the process alive, or should the orchestrator restart it?
- **Readiness:** can this instance safely receive prediction traffic right now?
- **Startup:** has a slow-starting process completed initialization?

A process can be alive but not ready because its model artifact is still loading or a required dependency is unavailable. Do not make liveness depend on every downstream service; a temporary database outage should not necessarily restart every model worker.

## Bound every request

Set explicit limits for:
- Request body size and input sequence length.
- Queue depth and concurrent inference requests.
- Model execution time and downstream feature-store timeouts.
- Maximum batch size and batch wait time.
- Per-user or per-client request rate.

Unbounded queues turn overload into growing latency and memory pressure. Prefer backpressure or a clear overload response.

## Retry carefully

Retries can help with transient failures, but they can also amplify an outage.

- Retry only operations that are safe to repeat, or use idempotency keys.
- Use a small retry budget with exponential backoff and jitter.
- Respect an overall request deadline; do not let retries exceed it.
- Avoid retrying validation errors or deterministic model failures.
- Consider a circuit breaker to stop repeatedly calling an unhealthy dependency.

## Isolate failures and degrade gracefully

A bulkhead limits how much one dependency or workload can consume. Separate resource pools or concurrency limits can prevent slow requests from exhausting all workers.

A fallback may use a cached result, a simpler baseline model, or a clear "temporarily unavailable" response. The fallback must preserve authorization and disclose reduced capability. Never silently substitute a stale or low-confidence prediction for a high-impact decision.

## Minimal FastAPI health endpoints

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/health/live")
def liveness():
    return {"status": "alive"}

@app.get("/health/ready")
def readiness():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ready"}
```

Load and assign the model during application startup. In a real service, readiness may also check required local resources, but keep checks bounded and avoid exposing internal details.

## Observability and service objectives

Track request rate, error rate, P50/P95/P99 latency, queue time, model execution time, timeouts, saturation, and model version. Use request IDs and distributed traces to separate preprocessing, feature lookup, network, and model time.

Define service-level objectives from user needs. Averages alone can hide slow requests that affect a meaningful share of users.

## Practice

Load-test a FastAPI model service with increasing concurrency. Introduce an artificial slow feature lookup. Measure tail latency, queue growth, timeout behavior, and recovery after the dependency returns. Verify that readiness and liveness behave differently.

## References

- [Kubernetes: Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
- [Google SRE Book: Handling Overload](https://sre.google/sre-book/handling-overload/)

**Key takeaway:** Reliable model serving needs bounded work, carefully designed retries, health checks, observability, and an explicit failure policy—not only a fast prediction function.