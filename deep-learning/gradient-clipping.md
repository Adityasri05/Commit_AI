# Gradient Clipping for Stable Neural-Network Training

Backpropagation computes gradients that tell an optimizer how to update model parameters. In deep networks—especially recurrent networks or models trained with aggressive learning rates—gradient norms can become extremely large. This is called **exploding gradients** and can cause unstable updates or non-finite loss values.

Gradient clipping limits the size of gradients before the optimizer applies an update.

## Clip by global norm

For all parameter gradients (g_1, g_2, ldots, g_m), the global norm is:

[
|g|_2 = sqrt{sum_{j=1}^{m}|g_j|_2^2}
]

If this norm exceeds a chosen maximum (c), norm clipping rescales the gradients:

[
g_j leftarrow g_j cdot rac{c}{|g|_2}
]

The direction of the combined gradient vector is preserved while its magnitude is limited.

## PyTorch training step

```python
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
loss.backward()

# Clip after backward and before optimizer.step().
total_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0
)
optimizer.step()
```

The returned norm is the total norm measured before clipping. Choose the threshold based on observed training behavior; 1.0 is only a common starting point, not a universal optimum.

## Mixed-precision training

With automatic mixed precision and gradient scaling, unscale the gradients before clipping so the threshold applies to their real values:

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

For gradient accumulation, clip after all micro-batch gradients have been accumulated and immediately before the optimizer update.

## Norm clipping vs value clipping

- **Norm clipping** rescales the combined gradient when its norm is too large. It preserves the overall direction.
- **Value clipping** clamps each gradient element independently to a range. It can change the direction of the combined gradient.

Norm clipping is a common default for controlling exploding gradients.

## What clipping cannot fix

- A learning rate that is far too large.
- Incorrect loss scaling or a numerical bug.
- Badly initialized or incorrectly connected layers.
- Invalid inputs, targets, or labels.
- A model architecture that is unsuitable for the task.

Log gradient norms, loss, and the fraction of steps that actually clip. If clipping happens almost every step, investigate the learning rate, initialization, data scale, and architecture instead of merely lowering the threshold.

## Practice

Train a small recurrent or deep feed-forward network with and without norm clipping. Plot the loss and gradient norm, and compare how often clipping activates under two learning rates.

## References

- [PyTorch: clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [Pascanu, Mikolov, and Bengio: On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)

**Key takeaway:** Clip gradients after backpropagation and before the optimizer step. Treat frequent clipping as a diagnostic signal, not a substitute for sound training choices.