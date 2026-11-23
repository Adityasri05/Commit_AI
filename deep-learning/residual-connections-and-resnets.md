# Residual Connections and ResNets

As neural networks become deeper, optimization can become harder. Residual connections provide a shortcut that lets information and gradients pass around a block of layers.

## The residual formulation

A plain block tries to learn a mapping H(x). A residual block instead learns a residual F(x) and adds the input:

    y = F(x) + x

If the desired mapping is close to the identity, the residual branch can learn a small change rather than reconstructing the entire mapping. The shortcut also creates a direct path for gradients during backpropagation.

A residual connection does not guarantee that a network will train well. Initialization, normalization, activation functions, data, and optimization still matter.

## Shape compatibility

The addition requires F(x) and x to have the same shape. If the block changes channel count or spatial resolution, use a projection shortcut, commonly a 1x1 convolution with a suitable stride.

## A small PyTorch block

```python
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)
```

The example uses a common post-activation arrangement. ResNet variants differ in where normalization and activation are placed.

## Why shortcuts help

- They create a shorter route for gradient flow.
- They make identity-like behavior easier to represent.
- They support much deeper convolutional networks.
- They can improve optimization compared with an equally deep plain network.

Residual connections are also used in Transformer blocks, though the surrounding operations and normalization choices differ.

## Common mistakes

- Adding tensors with incompatible channel or spatial dimensions.
- Forgetting that a stride changes the shortcut shape too.
- Treating residual connections as a substitute for normalization or good optimization.
- Comparing architectures with different training budgets and data augmentation.

## Practice

Build a plain CNN and a residual CNN with similar parameter counts. Train them using the same data split and optimizer. Compare training curves, validation performance, and gradient behavior.

## References

- [He et al., Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch: Building Models](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)

**Key takeaway:** A residual block learns a change to its input and adds a shortcut path. Keep tensor shapes aligned and test the full architecture empirically.