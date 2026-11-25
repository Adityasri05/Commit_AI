# Generalized Additive Models (GAMs)

A generalized additive model combines interpretable feature-wise functions instead of forcing every feature to have a straight-line effect. It is a useful middle ground between a simple linear model and a highly flexible black-box model.

For a regression problem with a Gaussian response, a GAM can be written as:

    E[Y | X] = beta_0 + f_1(X_1) + f_2(X_2) + ... + f_p(X_p)

For other response distributions, a link function connects the expected target to the additive predictor:

    g(E[Y | X]) = beta_0 + sum_j f_j(X_j)

Each f_j is a learned smooth function, often represented with splines. The contribution of one feature can be plotted while holding the additive structure in view.

## Why use a GAM?

- It can capture nonlinear relationships such as diminishing returns.
- Feature contributions are easier to inspect than those of many complex models.
- Smoothness penalties reduce overfitting by discouraging unnecessarily wiggly functions.
- Domain knowledge can sometimes be expressed through monotonicity or shape constraints.

A basic additive GAM does not automatically model interactions. If the effect of one feature depends on another, add selected interaction terms or use a model designed for interactions.

## Example with pyGAM

~~~python
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Smooth terms for the first three numeric features.
gam = LinearGAM(s(0) + s(1) + s(2)).gridsearch(
    X_train, y_train
)
predictions = gam.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print(gam.summary())
~~~

This example assumes at least three numeric columns. Encode categorical variables appropriately and keep preprocessing inside the training workflow. Inspect the installed pyGAM documentation for current API details.

## Understanding a feature effect

A partial-dependence-style term plot shows the estimated contribution f_j(x_j), usually centered around zero. It answers: as this feature changes, what contribution does the fitted additive model assign, with other additive terms held separate?

It is not a causal effect. Correlated features can make individual contributions unstable or difficult to interpret. A smooth curve should also be inspected near sparse regions, where the model has little evidence.

## Model selection and validation

1. Start with a linear or tree-based baseline.
2. Choose candidate spline complexity and smoothness using validation data or cross-validation.
3. Inspect residuals and calibration, not just a single error metric.
4. Check whether smooth terms behave plausibly and whether there are enough observations across the feature range.
5. Evaluate subgroup errors and important interactions.
6. Compare interpretability and performance against a simpler model.

## Common pitfalls

- Too many spline basis functions can overfit.
- Excessive smoothing can erase real structure.
- Extrapolation beyond observed feature ranges is unreliable.
- Correlated inputs can make attribution among terms ambiguous.
- A readable plot does not establish fairness or causality.

## Practice

Fit a linear regression and a GAM to a dataset where the target has a nonlinear relationship with one feature. Compare validation MAE and plot the learned term. Explain where the GAM improves fit and where its curve is uncertain.

## References

- [pyGAM documentation](https://pygam.readthedocs.io/)
- [Hastie and Tibshirani: Generalized Additive Models](https://doi.org/10.1214/ss/1177013604)

**Key takeaway:** GAMs provide flexible nonlinear prediction while keeping feature effects inspectable, but their additive assumptions and validation limits must be understood.