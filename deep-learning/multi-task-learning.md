# Multi-Task Learning with Shared Representations

Multi-task learning (MTL) trains one model to solve several related tasks together. Instead of building a completely separate network for every task, the model can share part of its representation and use task-specific output heads.

For example, a document model may predict both topic and sentiment. A vision model may predict object class and bounding-box coordinates. A shared representation can help when tasks rely on overlapping information.

## Hard parameter sharing

A common architecture contains:

1. A shared backbone that extracts features.
2. One task-specific head per objective.
3. A loss for each task.
4. A combined training objective.

\[
L_{\mathrm{total}} = \sum_{i=1}^{T} \lambda_i L_i
\]

Here, L_i is the loss for task i and lambda_i controls its contribution.

~~~python
import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, classes, regression_outputs=1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, classes)
        self.regressor = nn.Linear(hidden_dim, regression_outputs)

    def forward(self, x):
        features = self.shared(x)
        return self.classifier(features), self.regressor(features)

model = MultiTaskModel(input_dim=20, hidden_dim=64, classes=4)
class_logits, value_prediction = model(batch_x)
loss_class = nn.functional.cross_entropy(class_logits, class_labels)
loss_value = nn.functional.mse_loss(value_prediction, target_values)
loss = loss_class + 0.2 * loss_value
loss.backward()
~~~

The loss weight is illustrative. It should be selected using validation evidence, not copied blindly.

## Why tasks can help each other

- **Positive transfer:** Shared features learned for one task improve another.
- **Regularization:** Related objectives can discourage the shared model from relying on narrow shortcuts.
- **Efficiency:** Shared computation can reduce the cost of maintaining multiple independent models.

## Negative transfer

Tasks can also interfere. If two tasks require conflicting representations or have very different data distributions, joint training may reduce performance.

Symptoms include one task improving while another degrades, unstable loss curves, or gradients from different tasks pointing in opposing directions.

Ways to investigate or reduce interference:
- Compare against independent single-task baselines.
- Inspect per-task validation metrics, not only total loss.
- Adjust loss weights and sampling frequency.
- Use separate task-specific layers or partial sharing.
- Consider gradient-balancing approaches such as GradNorm or gradient surgery when justified.
- Check whether one task has much more labeled data than the others.

## Loss balancing

Loss magnitudes are not directly comparable. A regression loss may be numerically much larger than a classification loss. Static weights are simple, but uncertainty-based weighting and gradient-based methods can adapt task contributions.

Dynamic weighting is not a substitute for diagnosing incompatible tasks. A task can receive a small weight because it is noisy, yet still be essential to the product.

## Missing labels and task sampling

Real datasets often have labels for only some tasks. Compute each task loss only for examples with valid labels, and normalize by the number of valid examples. Do not treat missing labels as a real class or a zero-valued target.

When batches contain different task mixtures, track how often each task contributes to updates. Otherwise, a rare task can be effectively ignored.

## Evaluation protocol

- Use task-specific metrics and a shared validation protocol.
- Compare MTL with independent models under similar compute budgets.
- Report per-task performance and important subgroup results.
- Measure inference cost: a shared model may be cheaper, but multiple heads or large backbones still have runtime costs.
- Test the behavior when one task's labels are missing or delayed.

## Mini-project

Train a shared network to predict both product category and review rating. Compare joint training with two separate models. Sweep a small set of loss weights and report per-task accuracy/MAE, convergence, parameter count, and inference latency.

## References

- Caruana, *Multitask Learning* (1997): https://doi.org/10.1023/A:1007379606734
- Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing* (2018): https://arxiv.org/abs/1711.02257
