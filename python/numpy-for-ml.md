# NumPy Essentials for Machine Learning

NumPy provides efficient n-dimensional arrays and vectorized numerical operations. It is a foundation for many scientific Python tools.

## Core concepts
- ndarray: an n-dimensional homogeneous array.
- shape: the size of each dimension.
- dtype: the element data type.
- axis: the dimension along which an operation is performed.
- broadcasting: rules that allow operations between compatible shapes.

## Example
```python
import numpy as np

X = np.array([[1.0, 2.0], [3.0, 4.0]])
mean_by_feature = X.mean(axis=0)
X_centered = X - mean_by_feature
print(X_centered)
```

## Useful operations
- `np.mean`, `np.std`: summary statistics.
- `np.reshape`: change array shape without changing data order.
- `np.concatenate`: join arrays.
- `np.random.default_rng`: reproducible random number generation.
- `@ ` or `np.matmul`: matrix multiplication.

## ML tips
Check shapes before matrix operations, use vectorization instead of Python loops when appropriate, and set random seeds for reproducible experiments. Avoid accidental data copies for large arrays.

## Exercise
Create a 100 × 3 array, standardize each column, and verify that each transformed column has mean near zero and standard deviation near one.