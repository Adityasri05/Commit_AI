# AdamW and Decoupled Weight Decay

AdamW is an adaptive optimization algorithm commonly used to train deep neural networks and Transformers. It combines Adam's per-parameter learning-rate adaptation with a weight-decay update that is applied separately from the gradient-based moment estimates.

Understanding this distinction helps when reproducing training recipes: Adam with an L2 penalty and AdamW are not generally equivalent.

## Adam recap

For gradient g_t at step t, Adam maintains moving averages:

- First moment m_t: an exponential average of gradients.
- Second moment v_t: an exponential average of squared gradients.

After bias correction, the parameter update is approximately:

theta_t = theta_(t-1) - learning_rate * m_hat_t / (sqrt(v_hat_t) + epsilon)

The adaptive denominator scales each parameter's update according to its recent gradient history.

## Why decoupled weight decay matters

A common regularization objective adds an L2 penalty to the loss. Its gradient is then mixed into the gradient passed to the optimizer.

With an adaptive optimizer, this penalty is also scaled by the adaptive denominator. The resulting shrinkage is not the same as applying a direct weight-decay step.

AdamW instead applies decay separately:

theta_t = theta_(t-1)
          - learning_rate * adaptive_gradient_update
          - learning_rate * weight_decay * theta_(t-1)

The decay term is not accumulated in Adam's first- and second-moment estimates. This is the central idea behind decoupled weight decay.

Weight decay discourages unnecessarily large parameters, but it does not guarantee better generalization. Tune it using validation data and compare with a suitable baseline.

## PyTorch example with parameter groups

Many Transformer training recipes apply weight decay to matrix-like weights but exclude biases and normalization parameters. This is a common convention, not a universal law; inspect the model architecture and follow the intended training recipe.

```python
import torch

decay_params = []
no_decay_params = []

for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue

    # Common heuristic: exclude biases and one-dimensional parameters.
    if name.endswith(".bias") or parameter.ndim == 1:
        no_decay_params.append(parameter)
    else:
        decay_params.append(parameter)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

A typical training step is:

```python
optimizer.zero_grad(set_to_none=True)
loss = criterion(model(inputs), targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

If using a learning-rate scheduler, follow its documented stepping convention. For many per-epoch schedulers, call the scheduler after the optimizer has completed that epoch.

## How to tune AdamW

- **Learning rate:** often has a larger effect than small changes in weight decay. Use a schedule appropriate to the task.
- **Weight decay:** test a logarithmic range and evaluate generalization, not just training loss.
- **Betas:** control the memory of gradient and squared-gradient estimates. Keep defaults as a baseline before changing them.
- **Epsilon:** stabilizes division and can matter for low-precision training.
- **Gradient clipping:** can limit extreme updates, but it is not a replacement for diagnosing unstable data or loss scaling.

When comparing optimizers, keep the data split, batch size, training budget, scheduler, and initialization protocol consistent.

## Common pitfalls

- Treating AdamW's weight decay as identical to an L2 penalty inside the loss.
- Applying decay indiscriminately to biases or normalization parameters without checking the recipe.
- Comparing optimizers with different learning-rate schedules or training budgets.
- Reporting only training loss and ignoring validation behavior.
- Forgetting to zero gradients between steps.
- Assuming a single weight-decay value transfers across model sizes and datasets.

## Practice

Train the same small Transformer or MLP with Adam and AdamW. Keep the training budget and learning-rate schedule fixed. Compare training loss, validation loss, parameter norms, and performance across several weight-decay values. Record the exact optimizer configuration.

## References

- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [PyTorch AdamW documentation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

**Key takeaway:** Adam adapts updates using gradient moments; AdamW adds a separate parameter-shrinkage step. Reproducible experiments must state the optimizer, parameter groups, weight decay, and learning-rate schedule.
