# gRPC and Streaming Inference APIs

A model-serving API is not only a prediction function. Its communication protocol affects latency, payload size, compatibility, cancellation, and how clients receive partial results.

REST over HTTP is a practical default for many services. gRPC is useful when strongly typed contracts, efficient service-to-service communication, or streaming are important.

## What gRPC provides

gRPC commonly uses HTTP/2 and Protocol Buffers (Protobuf) to define service methods and message schemas.

- **Unary RPC:** one request produces one response.
- **Server streaming:** one request produces a sequence of responses.
- **Client streaming:** a sequence of client messages produces one response.
- **Bidirectional streaming:** both sides exchange message streams.

Typical model-serving uses:

| Pattern | Example |
|---|---|
| Unary | One image or feature vector produces one prediction |
| Server streaming | LLM generation sends token or text chunks |
| Client streaming | Client uploads a sequence of sensor windows |
| Bidirectional | Live speech or interactive session processing |

Streaming is useful when partial output matters. It is not automatically faster for every workload; connection setup, serialization, buffering, and server scheduling still matter.

## Define a typed contract

A small Protobuf service might look like this:

```proto
syntax = "proto3";

package inference.v1;

service TextGeneration {
  rpc Generate(GenerateRequest) returns (stream GenerateChunk);
}

message GenerateRequest {
  string request_id = 1;
  string prompt = 2;
  uint32 max_new_tokens = 3;
}

message GenerateChunk {
  string request_id = 1;
  string text = 2;
  bool is_final = 3;
}
```

The contract should define field meaning, units, limits, and compatibility expectations. Use stable field numbers and do not reuse deleted field numbers for unrelated data. Add a new versioned package or service when a breaking change cannot be handled compatibly.

## Server-streaming behavior

A server can emit chunks as they become available. In an asynchronous Python gRPC handler, the conceptual pattern is:

```python
async def Generate(self, request, context):
    validate_request(request)

    async for piece in model_stream(
        request.prompt,
        max_new_tokens=request.max_new_tokens,
    ):
        if not context.is_active():
            return

        yield GenerateChunk(
            request_id=request.request_id,
            text=piece,
            is_final=False,
        )

    yield GenerateChunk(
        request_id=request.request_id,
        text="",
        is_final=True,
    )
```

This is an illustrative handler; generated Protobuf classes and service base classes come from the compiled .proto file. Production code must also handle model errors, deadlines, cancellation, and cleanup of generation state.

## Streaming design decisions

- Define whether chunks contain tokens, text fragments, structured events, or final metadata.
- Include a request or trace ID so clients can correlate events.
- Decide how the client learns that generation is complete.
- Specify whether partial output remains valid after a cancellation or failure.
- Avoid emitting one network message for every tiny token if batching a few tokens improves efficiency without unacceptable delay.
- Bound queue sizes so slow clients cannot cause unbounded memory growth.
- Propagate cancellation to the underlying model generation task.

For browser clients, Server-Sent Events (SSE) may be simpler when the communication is one-way text streaming over HTTP. WebSockets are useful when the browser and server both need continuous interactive messages. Choose based on the client environment and operational requirements.

## Reliability and security

- Set deadlines and enforce maximum prompt, token, and message sizes.
- Use TLS; consider mutual TLS for trusted service-to-service identity.
- Authenticate and authorize every request.
- Apply per-user and per-tenant quotas.
- Respect HTTP/2 flow control and implement bounded buffering.
- Avoid automatic retries for non-idempotent generation requests unless the request has a safe idempotency strategy.
- Record time to first token, inter-chunk delay, total latency, cancellation rate, and token throughput.
- Redact prompts and outputs from logs when they may contain sensitive information.

## Compatibility

Protobuf supports additive schema evolution when field numbers and types are handled correctly. Do not change the meaning of an existing field in place. Test old clients against new servers and new clients against old servers where rolling upgrades are required.

Versioning the API contract does not replace versioning the model artifact or preprocessing behavior. Record the model version used for each request when that information is operationally relevant.

## Practice

Build a unary classification endpoint and a server-streaming text-generation endpoint. Test slow clients, client cancellation, deadline expiration, oversized prompts, and model errors. Measure time to first token, P95 completion latency, throughput, and memory under concurrent streams.

## References

- [gRPC documentation](https://grpc.io/docs/)
- [Protocol Buffers documentation](https://protobuf.dev/)
- [gRPC Python basics](https://grpc.io/docs/languages/python/basics/)
- [HTTP/2 specification](https://www.rfc-editor.org/rfc/rfc9113)

**Key takeaway:** Choose an API protocol around the interaction pattern. For streaming inference, define chunk semantics, cancellation, flow control, limits, and compatibility as carefully as the model itself.
