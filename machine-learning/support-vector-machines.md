# Support Vector Machines (SVM)

A Support Vector Machine is a supervised learning algorithm used for classification and regression. In classification, it seeks a decision boundary with a large margin between classes.

## Core intuition
The support vectors are training observations closest to the decision boundary. They strongly influence the position of the separating hyperplane.

For a linear classifier, the decision function has the form:

f(x) = wᵀx + b

The predicted class depends on the sign of f(x).

## Soft margins and C
Real datasets may not be perfectly separable. A soft-margin SVM allows some margin violations:
- A larger C penalizes training errors more strongly and may fit the training data more closely.
- A smaller C allows a wider margin with more violations and stronger regularization.

## Kernel trick
Kernels allow SVMs to represent nonlinear boundaries without explicitly computing a high-dimensional feature mapping. Common choices include linear, polynomial, and RBF kernels.

## Practical considerations
- Scaling numeric features is usually important.
- Tune C and kernel parameters using cross-validation.
- SVMs can be expensive on very large datasets.
- Probability estimates may require additional calibration.

## Exercise
Train linear and RBF SVM classifiers on a small nonlinear dataset. Compare validation scores and plot their decision boundaries.