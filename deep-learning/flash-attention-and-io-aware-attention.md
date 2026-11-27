# FlashAttention and IO-Aware Attention

Standard scaled dot-product attention computes scores between every query and key position. For a sequence of length N, the score matrix has N by N entries per head. Materializing and repeatedly reading large intermediate matrices can make memory traffic a bottleneck even when the arithmetic is manageable.

FlashAttention is an exact attention algorithm designed around the GPU memory hierarchy. It reduces high-bandwidth-memory (HBM) traffic by computing attention in tiles and maintaining numerically stable softmax statistics without storing the full attention matrix.

## Standard attention

For query Q, key K, and value V:

Attention(Q, K, V) = softmax(Q K-transpose / sqrt(d_k)) V

The arithmetic cost is quadratic in sequence length for dense attention. A conventional implementation may materialize the score and probability matrices, each of which grows with N squared.

The important distinction is between arithmetic complexity and memory traffic. Two algorithms can perform similar mathematical operations but have very different runtime because one moves far more data between GPU memory levels.

## The IO-aware idea

FlashAttention divides Q, K, and V into blocks that fit in fast on-chip memory such as shared memory and registers.

For each query block, it streams over key/value blocks and updates:

- Partial attention outputs.
- Running row-wise maximum values.
- Running normalization sums for the softmax.

The online softmax update allows the algorithm to combine results from successive key/value blocks while remaining mathematically equivalent to dense attention, up to floating-point rounding differences.

The full N-by-N attention matrix does not need to be written to HBM. This is the central IO improvement.

FlashAttention is not the same as approximate sparse attention. It computes exact dense attention, although floating-point results can differ slightly because operations are reordered.

## Use PyTorch scaled dot-product attention

Recent PyTorch versions expose an optimized scaled dot-product attention API. Depending on hardware, dtype, shapes, masks, and installed backend, PyTorch may select a FlashAttention-style kernel or another implementation.

```python
import torch
import torch.nn.functional as F

# q, k, v: [batch, heads, sequence_length, head_dimension]
def causal_attention(q, k, v, dropout_p=0.0, training=False):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p if training else 0.0,
        is_causal=True,
    )

output = causal_attention(q, k, v)
```

The function chooses an eligible backend automatically. Calling this API does not prove that a FlashAttention kernel was selected. Inspect PyTorch's backend controls or profiler output when measuring a particular implementation.

For non-causal attention, use the appropriate mask and set is_causal accordingly. Avoid combining incompatible mask arguments.

## Why it can be faster

- It reduces intermediate attention-matrix writes to HBM.
- Tiling improves reuse of loaded Q, K, and V blocks.
- The algorithm is designed to use GPU on-chip memory efficiently.
- It can reduce memory pressure for long sequences and make larger workloads practical.

The benefit depends on sequence length, head dimension, batch size, dtype, GPU architecture, and kernel availability. Small workloads may not benefit because launch overhead can dominate.

## Benchmarking correctly

Compare implementations using the same:

- Model weights and input tensors.
- Batch size, sequence length, head count, and head dimension.
- Dtype, masks, dropout behavior, and causal setting.
- GPU, software versions, and warm-up procedure.

Measure both forward and backward passes for training workloads. Synchronize CUDA before reading wall-clock timings, or use CUDA events. Report peak allocated memory and throughput as well as latency.

Check numerical agreement with a reference implementation using a tolerance rather than exact equality. Also test padded batches, long sequences, and the exact masks used in the application.

## Common pitfalls

- Assuming the optimized backend is always selected.
- Forgetting that dropout behavior must be disabled during evaluation.
- Comparing different attention masks or precision settings.
- Measuring without warm-up or CUDA synchronization.
- Ignoring backward-pass memory when optimizing training.
- Assuming FlashAttention removes the quadratic arithmetic of dense attention.
- Treating a small floating-point discrepancy as a bug without checking tolerances and numerical stability.

## Practice

Implement a simple reference attention function that materializes the score matrix. Compare it with PyTorch scaled dot-product attention across several sequence lengths. Record forward latency, forward-plus-backward latency, peak GPU memory, and maximum output difference.

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [PyTorch scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

**Key takeaway:** FlashAttention improves dense attention primarily by reducing memory traffic through IO-aware tiling. Benchmark the selected kernel on the real workload rather than assuming the API name guarantees a speedup.
