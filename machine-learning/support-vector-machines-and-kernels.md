# Support Vector Machines and Kernel Methods

Support Vector Machines (SVMs) are supervised models that learn a decision boundary with a large margin between classes. They are useful when the number of features is large relative to the number of observations, including text classification and some medium-sized tabular datasets.

## The geometric idea

For a binary classifier, a linear SVM learns a hyperplane:

    w · x + b = 0

The prediction depends on which side of the hyperplane an example falls. The margin is the distance from the boundary to the closest training examples. SVM training seeks a boundary with a large margin while allowing controlled violations when the data is not perfectly separable.

The closest observations that determine the boundary are called support vectors. Moving a distant observation often does not change the fitted boundary.

## Soft-margin objective

Real datasets are rarely perfectly separable. A soft-margin SVM introduces slack variables and balances margin size against classification errors:

    minimize  0.5 * ||w||^2 + C * sum(xi_i)

subject to:

    y_i * (w · x_i + b) >= 1 - xi_i
    xi_i >= 0

The parameter C controls the trade-off:
- Larger C penalizes margin violations more strongly and may fit the training data more closely.
- Smaller C allows more violations and usually produces a wider, more regularized margin.

## The kernel trick

A nonlinear boundary can be learned by comparing examples through a kernel function. A kernel computes an inner product in an implicit feature space without explicitly constructing all transformed features.

Common choices:

- **Linear:** useful for high-dimensional sparse data such as bag-of-words text.
- **Polynomial:** models interactions up to a chosen degree.
- **RBF:** creates flexible nonlinear boundaries based on distance.
- **Sigmoid:** resembles a neural-network activation in some parameter ranges, but is less commonly the default choice.

For the RBF kernel:

    K(x, z) = exp(-gamma * ||x - z||^2)

Gamma controls how local each example's influence is. Very large gamma can create highly irregular boundaries; very small gamma can make the boundary too smooth.

## Practical scikit-learn example

~~~python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
~~~

Scaling belongs inside the pipeline so it is fitted only on training folds during cross-validation. Tune C and gamma with a validation strategy that respects the data's structure.

## Choosing and diagnosing an SVM

1. Establish a simple baseline, such as logistic regression.
2. Scale continuous features; distance-based kernels are sensitive to feature scale.
3. Compare linear and RBF kernels before trying more complex choices.
4. Inspect per-class precision and recall, not only accuracy.
5. Use class weights when the cost of errors differs across classes.
6. If probabilities are needed, calibrate scores on data separate from model fitting and evaluate calibration independently.

## Limitations and pitfalls

- Kernel SVM training can become expensive as the number of training observations grows; it is not automatically the best choice for very large datasets.
- Unscaled features can dominate the kernel distance.
- A high training score does not prove a useful margin or good generalization.
- Probability estimates are not the same thing as the raw decision function.
- Support vectors are influential examples, not necessarily explanations of why a prediction is correct.

## Practice

Use a synthetic two-moons dataset. Compare a linear SVM with an RBF SVM, plot their decision regions, and tune C and gamma with cross-validation. Record training time, test performance, and the number of support vectors.

## References

- [scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Cortes and Vapnik: Support-Vector Networks](https://doi.org/10.1007/BF00994018)

**Key takeaway:** SVMs combine margin-based regularization with optional kernel functions. Scaling, kernel choice, and the C/gamma trade-off determine whether the learned boundary generalizes well.