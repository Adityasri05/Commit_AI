# Practical Model Inference Optimization

A model that performs well in a notebook may be too slow or expensive for production. Optimize only after measuring a realistic workload and defining acceptable prediction quality.

## Start with a baseline

Measure:
- End-to-end latency, including preprocessing and network time.
- Median and tail latency such as P95 and P99.
- Throughput under realistic concurrency.
- Peak memory and accelerator utilization.
- Cost per request or per thousand predictions.
- Prediction quality against a trusted baseline.

A faster model is not useful if it silently changes output quality beyond the application's tolerance.

## Common optimization techniques

### Batching
Combine multiple requests into one model call. This can improve hardware utilization, but waiting to fill a batch adds latency. Use a maximum batch size and a maximum wait time.

### Quantization
Represent weights or activations with lower precision, such as INT8 or lower-bit formats. This can reduce memory use and improve speed on supported hardware. Calibration, hardware kernels, and model architecture affect the result; validate quality on representative data.

### Pruning and sparsity
Remove or zero less useful parameters. Sparse weights do not guarantee faster inference unless the runtime and hardware support efficient sparse operations.

### Knowledge distillation
Train a smaller student model to approximate a larger teacher. This can reduce serving cost, but the student must be evaluated on the target task and important edge cases.

### Compilation and graph optimization
A runtime may fuse operations, optimize memory movement, or compile a model for a target device. Benchmark warm-up and steady-state behavior separately.

### Caching
Cache deterministic results when inputs and authorization context make reuse safe. Avoid caching personalized or sensitive outputs across users.

### Transformer KV cache
During autoregressive generation, a key-value cache can avoid recomputing attention keys and values for earlier tokens. It reduces repeated computation but consumes memory, especially for long contexts and many concurrent sequences.

## A safe optimization loop

1. Save a quality and performance baseline.
2. Profile to find the actual bottleneck.
3. Change one optimization at a time.
4. Run correctness and regression tests.
5. Benchmark with realistic input lengths and concurrency.
6. Compare quality, tail latency, memory, and cost.
7. Roll out gradually and keep a rollback path.

## Common mistakes

- Optimizing model compute when network or feature lookup dominates.
- Reporting only average latency.
- Benchmarking only one tiny input.
- Ignoring cold starts and compilation warm-up.
- Comparing different hardware without stating it.
- Assuming quantization or pruning always improves latency.

## Practice

Benchmark a small image classifier in full precision and with a supported quantized runtime. Record batch size, hardware, P50/P95 latency, memory, and accuracy. Explain whether the trade-off is acceptable for a mobile or server use case.

## References

- [PyTorch: Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime: Performance tuning](https://onnxruntime.ai/docs/performance/tune-performance/)
- [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

**Key takeaway:** Profile first, optimize the real bottleneck, and preserve a measured quality baseline throughout the process.