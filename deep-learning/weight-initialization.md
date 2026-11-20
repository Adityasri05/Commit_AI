# Weight Initialization in Neural Networks

Neural-network training starts with initial parameter values. Poor initialization can make gradients vanish or explode, slow convergence, or break symmetry between neurons.

## Why symmetry matters
If neurons in the same layer start with identical weights and receive identical updates, they can learn the same feature. Random initialization helps break this symmetry. Biases are often initialized to zero, while weights need scale-aware initialization.

## Common schemes
- **Xavier/Glorot:** Designed to keep signal variance reasonably stable for activations such as tanh.
- **He/Kaiming:** Often suitable for ReLU-family activations.
- **Orthogonal:** Uses orthogonal matrices and can help preserve signal norms in some architectures.
- **Pretrained initialization:** Starts from weights learned on another task or dataset.

## Practical guidance
Use framework defaults for standard layers unless there is evidence to change them. Match initialization to the activation and architecture, and monitor activation and gradient statistics when training is unstable.

## Exercise
Train the same small multilayer perceptron with Xavier and He initialization. Compare training curves across several random seeds rather than relying on one run.

**Key point:** Initialization is about setting useful signal and gradient scales, not finding the final solution before training.