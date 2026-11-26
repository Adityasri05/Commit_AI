# Gaussian Process Regression

Gaussian Process Regression (GPR) is a Bayesian approach to regression that places a probability distribution over possible functions. Instead of returning only a point estimate, it can provide a predictive mean and a model-based measure of uncertainty.

GPR is especially useful for small or medium-sized datasets when uncertainty is important and a suitable covariance function can be chosen.

## From a function to a distribution

A Gaussian process is defined by a mean function and a covariance kernel:

    f(x) ~ GP(m(x), k(x, x'))

The mean function describes the prior expected output. The kernel describes how strongly function values at two inputs are related. A common zero-mean prior is used after centering or normalizing the target.

For training inputs X and observations y, the kernel matrix K contains pairwise covariances. With Gaussian observation noise of variance sigma_n squared, the predictive distribution at a new input x* has:

    predictive mean = k_*^T (K + sigma_n^2 I)^(-1) y

    predictive variance =
        k(x*, x*) - k_*^T (K + sigma_n^2 I)^(-1) k_*

The equations show why the kernel and noise assumptions matter: they determine how information from observed points influences predictions.

## Choosing a kernel

- **RBF:** assumes a very smooth function; controlled by a length scale.
- **Matérn:** allows rougher functions and is often a practical alternative to RBF.
- **Rational Quadratic:** can be viewed as a mixture of RBF length scales.
- **Periodic:** represents repeating structure.
- **White noise:** models independent observation noise.

Kernels can be combined by addition or multiplication to express different sources of variation. For example, a smooth trend plus a periodic component may suit seasonal measurements.

## Practical scikit-learn example

~~~python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel
)

rng = np.random.default_rng(7)
X = np.linspace(0, 8, 35).reshape(-1, 1)
y = np.sin(X[:, 0]) + 0.15 * rng.normal(size=len(X))

kernel = (
    ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
    + WhiteKernel(noise_level=0.02)
)
model = GaussianProcessRegressor(
    kernel=kernel, normalize_y=True, n_restarts_optimizer=5,
    random_state=7
)
model.fit(X, y)

X_new = np.linspace(-1, 9, 100).reshape(-1, 1)
mean, std = model.predict(X_new, return_std=True)
print(model.kernel_)
~~~

Plot the training points, predictive mean, and a band such as mean plus or minus 1.96 standard deviations. The band is an approximate model-based interval under the fitted Gaussian assumptions; it is not a distribution-free coverage guarantee.

## Hyperparameters and model checking

Kernel parameters such as length scale and signal variance are commonly fitted by maximizing the log marginal likelihood. This balances data fit with model complexity, but optimization can find local optima.

Check:
- whether residuals show structure the kernel has not captured;
- whether predictive intervals are too narrow or too wide on held-out data;
- sensitivity to kernel choice and initialization;
- behavior outside the range of observed inputs;
- whether noise is reasonably constant across the input space.

## Computational limits

Exact GPR usually requires cubic time in the number of training observations and quadratic memory for the covariance matrix. For large datasets, consider inducing-point or other sparse approximations, structured kernels, or a different model family.

## Common pitfalls

- Treating predictive standard deviation as guaranteed real-world uncertainty.
- Choosing an RBF kernel for a process with sharp discontinuities without checking fit.
- Forgetting to scale input dimensions when length scales are learned.
- Ignoring observation noise or fitting it to an unrealistic value.
- Extrapolating far beyond training data without inspecting uncertainty.

## Practice

Generate noisy samples from a smooth function with a periodic component. Fit RBF, Matérn, and periodic kernels. Compare held-out RMSE and interval behavior, including predictions beyond the training range.

## References

- [scikit-learn: Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [Rasmussen and Williams: Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/)

**Key takeaway:** GPR combines a kernel-based function prior with Bayesian updating. Its uncertainty is valuable, but only when the kernel, noise assumptions, and validation results are credible.