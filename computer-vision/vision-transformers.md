# Vision Transformers (ViT)

A Vision Transformer applies the Transformer encoder to an image by converting image patches into a sequence of tokens. Instead of using convolution as the main feature-mixing operation, self-attention lets each patch representation interact with other patches.

## From image to token sequence

For an image of height H and width W, patch size P, and C channels, a non-overlapping patchification creates:

    N = (H / P) * (W / P)

patches, assuming H and W are divisible by P. Each patch is flattened and projected to an embedding dimension D. A learned or otherwise defined positional representation is added because self-attention alone does not encode the original patch order.

Many ViT classifiers prepend a learnable class token. The Transformer encoder processes the sequence, and the class-token representation is passed to a classification head.

## Core components

1. Patch embedding: flatten each patch and project it into D dimensions.
2. Positional embedding: preserve spatial location information.
3. Transformer blocks: multi-head self-attention, feed-forward layers, normalization, and residual connections.
4. Classification head: map the selected representation to class scores.

Self-attention has a cost that grows roughly quadratically with the number of tokens. Smaller patches create more tokens and can increase memory and compute substantially.

## Why pretraining matters

Original ViT work showed strong results when trained on very large datasets. In practical projects, transfer learning from a pretrained model is often more feasible than training from scratch. Fine-tune on task-specific data and compare against a CNN baseline.

~~~python
import timm
import torch
from torch import nn

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes
)

# Example input: batch of RGB images resized and normalized
# according to the selected pretrained model's data config.
images = torch.randn(4, 3, 224, 224)
logits = model(images)
print(logits.shape)  # (4, num_classes)
~~~

The random tensor only demonstrates the expected shape; real inputs must use the model's documented resize, crop, and normalization pipeline. Verify the model name and preprocessing configuration for the installed timm version.

## ViT compared with CNNs

- CNNs use local kernels and translation-related inductive biases.
- ViTs use global attention over patch tokens, allowing flexible long-range interactions.
- CNNs can be data-efficient on smaller datasets.
- ViTs benefit from large-scale pretraining and careful fine-tuning.
- Hybrid models combine convolutional stems with attention blocks.

Neither architecture is universally better. Compare latency, memory, data size, robustness, and accuracy on the target distribution.

## Fine-tuning checklist

- Start with a pretrained checkpoint and documented preprocessing.
- Use a validation split that prevents subject, patient, or near-duplicate image leakage.
- Consider lower learning rates for pretrained backbone layers.
- Apply augmentations that preserve label semantics.
- Track per-class metrics and calibration where probabilities drive decisions.
- Test at the deployment image resolution and batch size.

## Common pitfalls

- Using the wrong normalization or channel order.
- Resizing images in a way that removes small objects or distorts geometry.
- Assuming patch size is a harmless hyperparameter.
- Comparing models with different input resolution or augmentation budgets.
- Reporting a single random split on a small dataset.

## Practice

Fine-tune a small pretrained ViT and a pretrained CNN on the same dataset. Keep the data split and augmentation policy comparable, then compare validation macro-F1, parameter count, latency, and memory.

## References

- [An Image is Worth 16x16 Words: ViT](https://arxiv.org/abs/2010.11929)
- [timm documentation](https://huggingface.co/docs/timm/index)

**Key takeaway:** ViT turns image patches into tokens. Its success depends on patch resolution, positional information, pretraining, and a fair comparison with convolutional baselines.