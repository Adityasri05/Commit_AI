# Image Data Augmentation for Computer Vision

Data augmentation creates label-preserving variations of training images. It can improve robustness when the transformations resemble variations expected in real inputs.

## Common transformations
- Random horizontal flips when left-right orientation is not label-defining.
- Small rotations, crops, translations, and scale changes.
- Brightness, contrast, and color jitter for lighting variation.
- Random erasing or cutout to reduce reliance on one small region.

## Guard against invalid transformations
A transformation is not automatically safe. Vertical flips may be unrealistic for natural photographs; aggressive crops may remove the object; color changes may destroy labels in medical or industrial imagery.

## Correct data flow
Apply random augmentation only to training examples. Validation and test images should use deterministic preprocessing consistent with inference. Keep all augmented versions of an original image within the same split to prevent leakage.

## Practical experiment
1. Establish a baseline with resize and normalization.
2. Add one augmentation family at a time.
3. Track training and validation metrics.
4. Inspect augmented samples visually.
5. Compare robustness on a representative held-out set.

## Key takeaway
Augmentation encodes assumptions about which visual changes should not alter the label. Choose transformations from the task's real-world variation, not from a generic recipe.