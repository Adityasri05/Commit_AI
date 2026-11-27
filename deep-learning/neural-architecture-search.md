# Neural Architecture Search (NAS)

Neural Architecture Search automates some of the decisions involved in choosing a neural network structure. Instead of only tuning learning rate or batch size, a NAS procedure searches over architectural choices such as depth, width, layer types, kernel sizes, attention blocks, and connectivity.

NAS is a model-selection problem with a potentially expensive objective: each candidate architecture may need to be trained and evaluated.

## The three parts of a NAS problem

### 1. Search space

The search space defines what architectures are allowed.

- **Macro search:** choose the number and arrangement of stages or blocks.
- **Micro search:** choose operations inside a repeated cell, such as convolution, pooling, or skip connections.
- **Continuous choices:** learn relaxed operation weights, then discretize the result.

A search space that is too narrow may exclude useful models. A search space that is too large can waste compute and make the search difficult to reproduce.

### 2. Search strategy

- **Random search:** sample valid architectures. It is a strong, simple baseline.
- **Evolutionary search:** mutate and select architectures based on measured fitness.
- **Bayesian optimization:** use a surrogate model to select promising candidates. It is related to hyperparameter optimization, but the candidate includes architectural structure.
- **Differentiable NAS:** relax discrete choices into continuous weights and optimize them with gradients.
- **Reinforcement-learning-based search:** train a controller to propose architectures using a reward signal.

### 3. Evaluation strategy

The simplest approach trains every candidate from scratch, which can be prohibitively expensive. Weight sharing, early stopping, lower-resolution training, and proxy datasets reduce cost, but their rankings may not match full training.

## A small architecture-search example

This example uses Optuna to compare MLP architectures. It illustrates the search loop; it is not a benchmark of NAS methods.

```python
import optuna
import torch
from torch import nn

def build_model(input_dim, width, depth, dropout, classes):
    layers = []
    in_features = input_dim
    for _ in range(depth):
        layers += [
            nn.Linear(in_features, width),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        in_features = width
    layers.append(nn.Linear(in_features, classes))
    return nn.Sequential(*layers)

def objective(trial):
    width = trial.suggest_categorical("width", [32, 64, 128, 256])
    depth = trial.suggest_int("depth", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = build_model(
        input_dim=X_train.shape[1],
        width=width,
        depth=depth,
        dropout=dropout,
        classes=num_classes,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Implement these using fixed, reproducible data loaders.
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer)
        score = evaluate_accuracy(model, validation_loader)
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return evaluate_accuracy(model, validation_loader)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params)
```

For a real experiment, implement the training and evaluation functions, set seeds where practical, and keep the test set out of the search. This example searches a small family of MLP architectures; CNN, Transformer, and graph search spaces need task-specific builders.

## Weight sharing and proxy evaluation

Some NAS methods train a supernetwork that contains many candidate paths. Candidate architectures inherit shared weights instead of being trained independently.

This can save compute, but it introduces a ranking problem: shared-weight validation scores may be biased by training interference, unequal path exposure, or optimization dynamics. Re-train selected candidates from scratch before trusting the final comparison.

Other proxies include fewer epochs, smaller images, reduced datasets, and lower precision. A proxy is useful only if it preserves the relative ranking that matters for the target deployment.

## Hardware-aware NAS

The most accurate architecture is not always the best deployable architecture. Include constraints such as:

- Parameter count and model size.
- Peak activation memory.
- Latency on the actual target device and batch size.
- Energy or serving cost when it can be measured reliably.
- Quality under the expected input distribution.

A FLOPs estimate is not a substitute for real latency: memory movement, kernel availability, and hardware utilization can dominate.

## Evaluation and reproducibility checklist

- State the search space and invalid-architecture rules.
- Record the search algorithm, budget, seeds, and compute used.
- Use the same data splits and evaluation protocol for candidates.
- Avoid tuning repeatedly against the final test set.
- Re-train top candidates independently when using weight-sharing or proxy scores.
- Compare with a hand-designed baseline under the same compute and deployment constraints.
- Report both model quality and operational cost.

## Common mistakes

- Reporting only the best score without the search budget.
- Comparing candidates trained for different amounts of compute.
- Treating validation performance from a weight-sharing supernetwork as final evidence.
- Ignoring inference latency and memory.
- Searching so many architectures that the validation set itself becomes overfit.
- Failing to publish the exact search space and seeds.

## Practice

Create two search spaces for a small image classifier: one with fixed convolution blocks and tunable width/depth, and another that also chooses between convolution, pooling, and skip operations. Compare random search with an evolutionary strategy. Re-train the top three architectures from scratch and benchmark them on the same hardware.

## References

- [NAS: A Survey](https://arxiv.org/abs/1808.05377)
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)
- [Optuna documentation](https://optuna.readthedocs.io/)

**Key takeaway:** NAS searches over model structure, but the result is only meaningful when the search budget, evaluation protocol, and deployment constraints are reported honestly.
