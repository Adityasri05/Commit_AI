# Transfer Learning for Computer Vision

Transfer learning reuses representations learned on a source task, often from a large image dataset, to solve a related target task. It is especially useful when the target dataset is small or compute is limited.

## Two common strategies

1. **Feature extraction:** Freeze the pretrained backbone and train a new classification head. This is faster and reduces the number of trainable parameters.
2. **Fine-tuning:** Unfreeze some or all backbone layers and continue training with a smaller learning rate. This can adapt features to the target domain but may overfit small datasets.

## PyTorch sketch

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
```

Train the new head first. For fine-tuning, selectively unfreeze later layers and use a lower learning rate for pretrained weights than for the new head when using parameter groups.

## Important details

- Use the exact preprocessing expected by the pretrained weights (the weights object provides recommended transforms).
- Keep validation and test images separate from training augmentation.
- Split by subject, source, or capture session when images are correlated.
- Check whether source and target domains differ substantially.
- Verify the pretrained model's license and permitted use.

## Exercise

Train a frozen-backbone baseline, then unfreeze the final residual block. Compare validation performance, training time, and overfitting. Keep all other settings fixed.

**Takeaway:** Pretrained weights are a starting point, not a guarantee of target-domain accuracy.