# Logistic Regression

Despite its name, logistic regression is commonly used for classification. It estimates class probabilities using a linear score passed through a link function.

## Binary model
z = wᵀx + b

p(y=1|x) = σ(z) = 1 / (1 + e^(−z))

A threshold (often 0.5, but chosen for the application) converts probability into a class prediction.

## Training
The model commonly minimizes binary cross-entropy:

L = −[y log(p) + (1−y) log(1−p)]

The loss penalizes confident incorrect predictions strongly.

## Multiclass classification
Softmax logistic regression estimates a probability for each class. The probabilities sum to one.

## Strengths and limitations
It is fast, interpretable in many settings, and provides probabilities. A linear decision boundary may be inadequate for complex patterns; feature engineering or nonlinear models may help.

## Practical checklist
- Scale numeric features when useful.
- Encode categorical variables.
- Choose thresholds using validation data and error costs.
- Inspect precision, recall, F1, and calibration.

## Exercise
Train logistic regression to classify a binary dataset. Compare the default threshold with a threshold selected to improve recall, and report the precision trade-off.