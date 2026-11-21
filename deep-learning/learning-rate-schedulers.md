# Learning-Rate Schedulers in Deep Learning

The learning rate controls the size of an optimizer's parameter updates. A rate that is too high can make training unstable; one that is too low can make learning painfully slow. A **learning-rate schedule** changes the rate during training according to a planned rule or observed validation behavior.

## Common schedules

- **Step decay:** reduce the rate by a fixed factor after selected epochs. Simple, but the drop points need tuning.
- **Exponential decay:** multiply the rate by a constant factor over time, producing a smooth decrease.
- **Cosine annealing:** gradually lowers the rate following a cosine curve; often used with a chosen training horizon.
- **Warmup:** begins with a small rate and increases it during early steps. It can stabilize training in large or sensitive models.
- **One-cycle policy:** increases and then decreases the rate over a planned number of optimizer steps.
- **Reduce on plateau:** monitors a validation metric and lowers the rate when progress stalls.

A schedule is part of the training configuration. Record it alongside the optimizer, batch size, number of steps, and random seed.

## PyTorch examples

### Step once per epoch

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)
    validate(model, valid_loader)
    scheduler.step()
```

### Reduce when validation loss plateaus

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)
    validation_loss = validate(model, valid_loader)
    scheduler.step(validation_loss)
```

The second scheduler needs the monitored metric. It is stepped after validation, whereas schedulers such as OneCycleLR are generally stepped after each optimizer update. Follow the specific scheduler's documentation.

## Practical checklist

1. Start with a defensible baseline learning rate.
2. Decide whether the schedule advances per batch, optimizer step, or epoch.
3. For step-based schedules, calculate the total number of optimizer updates correctly, including gradient accumulation.
4. Log the actual learning rate over time.
5. Compare schedules with the same data split and training budget.
6. Save the scheduler state when resuming training so the schedule does not restart unexpectedly.

## Common mistakes

- Calling a scheduler at the wrong frequency.
- Calling ReduceLROnPlateau without the validation metric.
- Using the number of batches when the scheduler expects optimizer steps.
- Changing the schedule and several other training settings at once, making results hard to interpret.
- Assuming a decreasing learning rate guarantees better validation performance.

## Practice

Train one model with a constant learning rate, one with cosine annealing, and one with ReduceLROnPlateau. Plot validation loss and learning rate on the same timeline. Explain whether the schedule responded to real validation behavior.

## References

- [PyTorch learning-rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [PyTorch LRScheduler API](https://pytorch.org/docs/stable/optim.html#learning-rate-scheduling)

**Key takeaway:** A scheduler is useful only when its update frequency, total training steps, and monitored metric match the training loop.