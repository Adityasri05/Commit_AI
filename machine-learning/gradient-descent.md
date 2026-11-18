# Gradient Descent

Gradient descent minimizes a differentiable objective such as a machine-learning loss.

## Update rule
For parameters θ and loss J(θ):

θ ← θ − η ∇J(θ)

θ represents parameters, ∇J(θ) is the loss gradient, and η is the learning rate. Subtracting the gradient moves parameters toward lower loss.

## Variants
- Batch gradient descent uses the full training set for each update.
- Stochastic gradient descent updates using one example at a time.
- Mini-batch gradient descent uses a small batch and is common in deep learning.

## Learning-rate effects
A rate that is too high may overshoot or diverge. A rate that is too low may converge very slowly. Learning-rate schedules and adaptive optimizers can help.

## Intuition
Imagine walking downhill in fog. Estimate the local slope, take a step downhill, and repeat. The step size is analogous to the learning rate.

## Exercise
For J(w) = (w − 3)², derive dJ/dw. Starting from w = 0 with η = 0.1, calculate the first two updates.