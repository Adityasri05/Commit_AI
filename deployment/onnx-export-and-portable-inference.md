# ONNX Export and Portable Model Inference

ONNX (Open Neural Network Exchange) is an open format for representing machine-learning models. It can help separate model development from deployment by allowing supported models to run in compatible runtimes such as ONNX Runtime.

Exporting to ONNX does not automatically make a model faster or guarantee that every operation is supported. Treat export as a compatibility and performance engineering step.

## Why use ONNX?

- **Interoperability:** move a supported model representation between frameworks and runtimes.
- **Deployment flexibility:** use CPU or accelerator execution providers supported by the runtime.
- **Graph optimization:** runtimes may fuse or optimize operations for a target device.
- **Operational separation:** deploy an exported artifact without installing the full training framework.

The exported graph is only part of the inference system. Tokenization, image resizing, normalization, feature ordering, and post-processing must remain consistent with training.

## Export a PyTorch model

Start with an evaluation-mode model and representative inputs. The following example exports a model with a fixed input shape; dynamic batch or sequence dimensions require an explicit dynamic-shape configuration supported by the installed PyTorch version.

```python
import torch

model.eval()
example = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    torch.onnx.export(
        model,
        (example,),
        "classifier.onnx",
        input_names=["image"],
        output_names=["logits"],
        opset_version=18,
    )
```

Export options evolve across PyTorch releases. Check the installed version's ONNX exporter documentation, especially when using the newer dynamo-based exporter or dynamic shapes.

## Run with ONNX Runtime

```python
import numpy as np
import onnxruntime as ort
import torch

session = ort.InferenceSession(
    "classifier.onnx",
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
x = torch.randn(1, 3, 224, 224)

model.eval()
with torch.inference_mode():
    torch_output = model(x).cpu().numpy()

onnx_output = session.run(
    None,
    {input_name: x.cpu().numpy()},
)[0]

np.testing.assert_allclose(
    torch_output,
    onnx_output,
    rtol=1e-3,
    atol=1e-4,
)
```

The tolerances depend on the model, dtype, runtime, and hardware. A failed comparison should be investigated rather than hidden by making tolerances arbitrarily large.

## Compatibility checklist

1. Confirm the model's operators are supported by the target ONNX opset and runtime.
2. Export representative input shapes and dtypes.
3. Test batch sizes and sequence lengths expected in production.
4. Check outputs for NaNs, infinities, and unexpected shape changes.
5. Validate preprocessing and post-processing parity.
6. Compare the exported model against the original framework on a fixed evaluation set.
7. Test the actual execution provider and hardware.

Some models need custom operators, graph simplification, or architecture changes. Dynamic control flow and custom Python operations may not export directly.

## Performance benchmarking

Measure the full serving path, not only the ONNX session call.

- Warm up the runtime before timing.
- Benchmark realistic batch sizes and concurrency.
- Record P50/P95 latency, throughput, peak memory, and CPU/GPU utilization.
- Include preprocessing, data transfer, and post-processing.
- Compare the same precision and hardware conditions.
- Verify task quality after graph optimization or quantization.

ONNX Runtime can use different execution providers. A provider may silently fall back for unsupported operations, so inspect the runtime configuration and profiling output.

## Artifact and deployment hygiene

- Store the ONNX file with a version and checksum.
- Record source framework version, exporter version, opset, and preprocessing contract.
- Keep a known-good original model artifact for comparison.
- Do not load untrusted model files without considering the security implications of the runtime and any custom operators.
- Test rollback and compatibility before replacing a production model.

## Common mistakes

- Exporting while the model is still in training mode.
- Forgetting preprocessing that was performed in the training pipeline.
- Assuming dynamic shapes work without declaring them.
- Ignoring operator-set and execution-provider compatibility.
- Treating ONNX as a compression method or guaranteed speedup.
- Checking one random input instead of a representative evaluation set.

## Practice

Export a small image classifier, run it with ONNX Runtime on CPU, and compare outputs across at least 100 held-out examples. Then benchmark several batch sizes. Record model size, P50/P95 latency, throughput, peak memory, and classification accuracy for both runtimes.

## References

- [ONNX documentation](https://onnx.ai/)
- [PyTorch ONNX export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)

**Key takeaway:** ONNX improves deployment portability when the graph and runtime are compatible. Validate numerical parity, preprocessing, shapes, and real workload performance before switching runtimes.
