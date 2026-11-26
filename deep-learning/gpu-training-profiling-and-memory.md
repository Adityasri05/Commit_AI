# GPU Training Profiling and Memory Optimization

A GPU training job can be slow even when a powerful accelerator is available. The bottleneck may be CPU preprocessing, data transfer, small kernels, synchronization, memory pressure, or communication between devices.

Profile first, identify the limiting stage, and then change one thing at a time.

## Understand GPU memory use

Training memory can include:

- **Parameters:** model weights.
- **Gradients:** gradient tensors for trainable parameters.
- **Optimizer state:** additional tensors maintained by the optimizer, such as momentum or Adam moments.
- **Activations:** intermediate tensors needed for backpropagation.
- **Temporary workspaces:** memory used by kernels and libraries.
- **Allocator reserve:** memory cached by the framework for reuse.

The largest category depends on model architecture, sequence length, batch size, precision, and optimizer.

## Profile the workload

Measure data loading, forward pass, backward pass, optimizer step, and synchronization separately. Use the PyTorch profiler to inspect operator time, shapes, memory, and CPU/GPU activity.

~~~python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(loader):
        with record_function("training_step"):
            train_one_step(batch)
        if step == 10:
            break

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=20,
))
~~~

The profiler adds overhead. Use it to diagnose a representative short run, then benchmark the optimized version without profiling.

## Common optimization techniques

### Mixed precision

Autocast can use lower-precision operations where supported. BF16 has a wider exponent range than FP16 on compatible hardware. FP16 training may need gradient scaling to reduce underflow.

~~~python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
~~~

Validate numerical behavior and model quality after changing precision. Do not assume every operation should use the same dtype.

### Activation checkpointing

Checkpointing stores fewer intermediate activations and recomputes selected forward operations during backward. It trades extra computation for lower activation memory.

It can enable larger batches or longer sequences, but recomputation may reduce throughput.

### Gradient accumulation

Accumulating gradients over several smaller micro-batches can approximate a larger effective batch when memory is limited. Divide or otherwise scale the loss correctly so gradient magnitude matches the intended batch objective.

### Efficient data loading

- Use enough workers to keep the GPU supplied, but benchmark rather than maximizing worker count.
- Consider pinned host memory and non-blocking transfers.
- Avoid expensive Python preprocessing in the critical path.
- Cache deterministic transformations when storage and freshness allow.
- Keep CPU and GPU work overlapped where practical.

### Optimizer and model choices

Optimizer state can be a substantial memory cost. Consider optimizer variants, parameter-efficient training, or sharding when appropriate. Validate convergence and numerical behavior after any change.

## Measure memory carefully

For a controlled benchmark, reset peak statistics before a training step and record allocated and reserved memory. CUDA operations are asynchronous, so synchronize only around timing measurements when accurate wall-clock timing is required.

Do not use cache-clearing calls as a substitute for understanding memory ownership. They do not free tensors that are still referenced by the program.

## A disciplined optimization loop

1. Fix random seeds and use the same model, data, and batch.
2. Record baseline throughput, peak memory, and validation quality.
3. Profile to identify the dominant cost.
4. Change one optimization at a time.
5. Repeat the benchmark after warm-up.
6. Check for numerical differences and quality regressions.
7. Keep the change only if the end-to-end result improves.

## Common pitfalls

- Timing asynchronous GPU work without synchronization.
- Profiling every training step and treating profiler timings as production timings.
- Increasing batch size until memory is full without measuring throughput.
- Assuming mixed precision always improves speed on every GPU.
- Ignoring CPU data preparation and transfer overhead.
- Calling cache-clearing functions while live tensors still occupy memory.

## Practice

Profile a small Transformer training loop. Compare full precision with supported mixed precision, then add activation checkpointing. Record step time, peak allocated memory, GPU utilization, and validation loss for each configuration.

## References

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [PyTorch CUDA memory management](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)

**Key takeaway:** GPU optimization is a measurement problem. Identify the actual bottleneck, understand which tensors consume memory, and validate every speed or memory improvement against model quality.