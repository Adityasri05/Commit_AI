# Deep Learning and Neural Networks

Deep Learning is a subfield of Machine Learning that uses neural networks with multiple layers to learn increasingly useful representations from data.

## Artificial neuron
A neuron receives inputs, multiplies them by weights, adds a bias, and applies an activation function:

z = w₁x₁ + w₂x₂ + ... + b  
a = f(z)

Here, x represents inputs, w represents learned weights, b is a bias, and f is an activation function.

## Network structure
- **Input layer:** Receives the feature representation.
- **Hidden layers:** Transform information and learn patterns.
- **Output layer:** Produces a prediction or score.

## Training basics
1. A forward pass computes the model's prediction.
2. A loss function measures prediction error.
3. Backpropagation computes gradients of the loss.
4. An optimizer updates parameters to reduce the loss.
5. The process repeats over batches and epochs.

## Common activation functions
- ReLU: commonly used in hidden layers.
- Sigmoid: maps values to the range 0–1.
- Softmax: converts output scores into a probability distribution.

## Common architectures
- CNNs: effective for spatial data such as images.
- RNNs and LSTMs: designed for sequential data.
- Transformers: use attention to model relationships between tokens.

## Practical cautions
Deep learning often needs substantial data and compute. Track validation performance, use suitable regularization, and compare against simpler baselines before choosing a large model.