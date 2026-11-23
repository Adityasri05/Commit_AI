# Semantic Segmentation: Metrics and Losses

Semantic segmentation assigns a class to every pixel in an image. Unlike image classification, which produces one label for an image, segmentation must identify both the class and its spatial extent.

Examples include road-scene parsing, medical-image analysis, and land-cover mapping.

## Build a pixel-level confusion matrix

For each pixel, compare the predicted class with the ground-truth class. Aggregate counts into a confusion matrix. For class k:

- TP: pixels correctly predicted as k.
- FP: pixels predicted as k but belonging to another class.
- FN: pixels belonging to k but predicted as another class.

The class-wise intersection over union (IoU) is:

    IoU_k = TP_k / (TP_k + FP_k + FN_k)

Mean IoU averages IoU across the selected classes. State whether background is included and how classes absent from the ground truth are handled; libraries differ in these conventions.

## Dice score

The Dice coefficient is:

    Dice_k = 2 * TP_k / (2 * TP_k + FP_k + FN_k)

For the same non-empty binary sets, Dice and IoU are related by:

    Dice = 2 * IoU / (1 + IoU)

Dice is often used in medical segmentation because it emphasizes overlap, but it does not replace careful class-wise reporting.

## Why pixel accuracy can mislead

Suppose most pixels are background. A model that predicts background everywhere may achieve high pixel accuracy while failing to segment the object of interest. Report per-class IoU or Dice, macro averages, and the confusion matrix.

## Common training losses

- **Pixel-wise cross-entropy:** a standard multiclass objective; can be dominated by frequent classes.
- **Weighted cross-entropy:** increases the contribution of underrepresented classes.
- **Focal loss:** reduces the relative contribution of easy examples and emphasizes harder ones.
- **Dice loss:** optimizes overlap directly and can help with imbalanced foreground regions.
- **Combined losses:** may balance stable pixel-wise learning with overlap-based objectives.

Choose weights using validation data; extreme weights can destabilize training.

## Data pipeline details

- Resize categorical masks with nearest-neighbor interpolation, not bilinear interpolation, or class IDs can become invalid intermediate values.
- Apply geometric transforms identically to image and mask.
- Use an explicit ignore label for unlabeled or void pixels.
- Check class IDs and mask alignment after every transform.
- Split by patient, scene, video, or acquisition session when neighboring images are correlated.

## Minimal metric calculation

```python
import numpy as np

def iou_from_counts(tp, fp, fn):
    denominator = tp + fp + fn
    return np.divide(
        tp, denominator,
        out=np.full_like(np.asarray(tp, dtype=float), np.nan),
        where=denominator != 0,
    )
```

Compute TP, FP, and FN per class from the pixel-level confusion matrix. Decide and document how absent classes are treated before taking the mean.

## Practice

Evaluate a road-scene segmentation model. Report per-class IoU, mean IoU, Dice for rare classes, pixel accuracy, and inference time. Inspect overlays for boundary errors and small-object failures.

## References

- [Long et al., Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [TorchMetrics: Jaccard Index](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html)

**Key takeaway:** Segmentation quality is spatial and class-dependent. Use overlap metrics, handle absent and ignored pixels explicitly, and validate the image-mask pipeline.