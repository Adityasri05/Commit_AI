# Linear Algebra Essentials for Machine Learning

Linear algebra gives machine-learning data and model parameters a precise language. Most datasets can be represented as matrices, and many model operations are matrix operations.

## Vectors, matrices, and tensors

- A **vector** is an ordered list of values, often representing one observation or a set of model weights.
- A **matrix** is a rectangular collection of values, often representing a batch of observations.
- A **tensor** generalizes arrays to more than two dimensions.

A dataset with 100 rows and 5 features has a feature matrix of shape `(100, 5)`.

## Dot product

For vectors a = [2, 1] and b = [3, 4]:

`a · b = 2 × 3 + 1 × 4 = 10`

A linear model computes a dot product between an input vector and its weights, then adds a bias.

## Matrix multiplication and shapes

If A has shape `(m, n)` and B has shape `(n, p)`, then AB has shape `(m, p)`. The inner dimensions must match. Shape checks are one of the fastest ways to debug neural-network code.

## Norms and similarity

- **L1 norm:** sum of absolute component values.
- **L2 norm:** Euclidean length.
- **Cosine similarity:** compares vector direction and is commonly used with text embeddings.

## Where it appears

Linear regression, neural-network layers, PCA, and embedding search all rely on linear algebra.

## Student exercise

1. Compute the dot product of [1, 2, 3] and [4, 0, -1].
2. What is the shape of a matrix product between `(32, 128)` and `(128, 10)`?
3. Explain why a transpose may be needed before multiplying two matrices.

**Practice tip:** Write down tensor shapes beside each layer while implementing a model.
