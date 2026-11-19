# Convolutional Neural Networks (CNNs)

CNNs are neural networks designed to exploit local structure, especially in grid-like data such as images.

## Main components
- Convolution: applies learned filters across local regions.
- Activation: introduces nonlinearity, often ReLU.
- Pooling or strided convolution: reduces spatial dimensions.
- Dense or global pooling head: maps learned features to outputs.

## Convolution intuition
A small filter slides across an image and computes local weighted sums. Early layers may learn edges or textures; deeper layers can combine these into more complex patterns.

## Important concepts
- Kernel size controls the local receptive field.
- Stride controls how far the filter moves each step.
- Padding controls border handling and output dimensions.
- Channels represent input or learned feature maps.

## Training tips
Normalize inputs, use appropriate augmentation, monitor validation metrics, and compare against a simple baseline. Ensure augmentations preserve the correct label.

## Transfer learning
A pretrained image backbone can provide useful representations when the target dataset is small. Replace or adapt the task-specific output head and fine-tune carefully.

## Exercise
Train a small CNN on a beginner image dataset. Compare it with a dense network and inspect a grid of misclassified images.