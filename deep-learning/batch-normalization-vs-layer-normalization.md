# Batch Normalization vs Layer Normalization

Normalization layers rescale activations during neural-network training. Batch Normalization (BatchNorm) and Layer Normalization (LayerNorm) both help optimization, but they compute statistics over different dimensions and behave differently at inference time.

## Batch Normalization

For a typical dense layer, BatchNorm estimates the mean and variance of each feature across the mini-batch. It normalizes using those statistics and then learns a scale and shift:

[
hat{x} = rac{x-mu_B}{sqrt{sigma_B^2+epsilon}},
qquad y = gammahat{x}+eta
]

During training, the current batch statistics are used and running estimates are updated. During evaluation, the layer normally uses those stored running estimates.

**Strengths**
- Often effective in convolutional networks.
- Can make optimization less sensitive to initialization and learning-rate choices.
- Adds a small amount of learned scale and shift per channel.

**Cautions**
- Very small or highly non-representative batches can make batch statistics noisy.
- Training and inference use different statistics.
- Distributed training may need synchronized batch statistics when local batches are tiny.

## Layer Normalization

LayerNorm computes statistics across the selected feature dimensions of each individual example. It does not need statistics from other examples in the batch.

For a token representation shaped [batch, sequence, hidden], a common configuration normalizes over the final hidden dimension.

**Strengths**
- Works with batch size one and variable batch sizes.
- Does not maintain running batch statistics.
- Fits naturally in Transformer blocks, where each token representation is normalized across its hidden features.

**Cautions**
- The normalized dimensions must match the architecture.
- LayerNorm is not interchangeable with BatchNorm without considering the tensor layout and model design.

## PyTorch sketches

```python
import torch.nn as nn

# For convolutional activations [N, C, H, W]:
batch_norm = nn.BatchNorm2d(num_features=64)

# For token or hidden representations [N, T, D]:
layer_norm = nn.LayerNorm(normalized_shape=768)
```

The examples show typical shapes; choose channel and hidden sizes to match the actual network.

## Quick comparison

| Question | BatchNorm | LayerNorm |
|---|---|---|
| Statistics across | Batch dimension for each feature/channel | Selected feature dimensions within each example |
| Depends on other batch examples? | Yes, during training | No |
| Running statistics? | Usually yes | No |
| Common use | CNNs and many dense networks | Transformers and sequence models |
| Small batch sensitivity | Can be significant | Generally much lower |

## Common mistakes

- Forgetting to call model.eval() for inference with BatchNorm and dropout.
- Assuming LayerNorm normalizes across the batch.
- Applying normalization over the wrong tensor dimensions.
- Treating normalization as a cure for poor data, an unstable optimizer, or an unsuitable architecture.

## Practice

Train the same small image classifier with BatchNorm and without it. Then train a sequence model using LayerNorm. Compare convergence, batch-size sensitivity, and train/evaluation behavior.

## References

- [PyTorch BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

**Key takeaway:** BatchNorm uses batch-level feature statistics; LayerNorm normalizes within each example. Choose based on the architecture and tensor dimensions.