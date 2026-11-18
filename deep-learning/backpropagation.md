# Backpropagation in Neural Networks

Backpropagation is an algorithm for computing how a loss changes with respect to each trainable parameter in a neural network. These gradients are then used by an optimizer.

## Two passes

1. **Forward pass:** inputs flow through layers to produce a prediction.
2. **Backward pass:** the chain rule propagates derivatives from the loss back through the network.

For a simple neuron:

z = wx + b

a = f(z)

If the loss is L, the chain rule gives:

∂L/∂w = (∂L/∂a) × f'(z) × x

This tells us how changing w affects the loss.

## Gradient descent update

w_new = w_old − η × ∂L/∂w

Here η is the learning rate. A very large learning rate can make training unstable; a very small one can make learning slow.

## Training loop
1. Initialize parameters.
2. Run a forward pass.
3. Compute the loss.
4. Compute gradients with backpropagation.
5. Update parameters with an optimizer.
6. Repeat over batches and epochs.

## Common optimizers
- SGD updates parameters using mini-batch gradients.
- Momentum smooths updates using a running direction.
- Adam adapts update sizes using running estimates of gradient moments.

## Debugging tips
Check tensor shapes, verify gradients are finite, monitor training and validation loss, and use gradient checking on tiny examples when implementing backpropagation manually.

**Exercise:** For z = wx + b and a = z (identity activation), derive ∂L/∂w for L = 1/2(a − y)².