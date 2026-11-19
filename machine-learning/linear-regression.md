# Linear Regression

Linear regression models a numeric target as a linear combination of input features.

## Model
For one feature: ŷ = wx + b. For multiple features: ŷ = Xw + b.

The parameters are commonly learned by minimizing mean squared error (MSE):

MSE = (1/n) Σ(yᵢ − ŷᵢ)²

## Assumptions and diagnostics
Classical linear regression inference often relies on linearity, independent errors, constant error variance, and assumptions about residual distributions. Prediction can still be useful when assumptions are imperfect, but inspect residuals and validate on held-out data.

## Regularization
Ridge adds an L2 penalty; Lasso adds an L1 penalty. These can reduce variance and help with correlated or numerous features.

## Practical workflow
1. Plot target against important features.
2. Split data before fitting preprocessing.
3. Fit a baseline linear model.
4. Inspect MAE/RMSE and residual plots.
5. Compare with nonlinear baselines if patterns remain.

## Exercise
Create a simple dataset where y is approximately 4x + 2 plus noise. Fit linear regression, interpret the learned slope and intercept, and calculate test MAE.