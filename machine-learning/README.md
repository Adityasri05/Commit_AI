# Machine Learning Fundamentals

Machine Learning (ML) is a branch of AI in which a system learns patterns from data to make predictions or decisions, rather than relying only on manually written rules.

## Main learning paradigms

### Supervised learning
The training data contains inputs and known target labels.
- **Regression:** Predicts a continuous value, such as house price.
- **Classification:** Predicts a category, such as spam or not spam.

### Unsupervised learning
The data has no supplied target labels. The system looks for structure.
- **Clustering:** Groups similar observations, for example with K-means.
- **Dimensionality reduction:** Compresses features while preserving useful information, for example with PCA.

### Reinforcement learning
An agent interacts with an environment and learns through rewards and penalties. It aims to maximize long-term cumulative reward.

## Typical ML workflow
1. Define the task and success metric.
2. Collect and inspect representative data.
3. Clean data and engineer useful features.
4. Split data into training, validation, and test sets.
5. Train a baseline model.
6. Evaluate, tune, and compare models.
7. Deploy and monitor performance.

## Important concepts
- **Overfitting:** The model learns training details too closely and performs poorly on unseen data.
- **Underfitting:** The model is too simple to capture important patterns.
- **Generalization:** Performance on data not used for training.
- **Data leakage:** Information unavailable at prediction time accidentally enters training.

## Evaluation examples
- Classification: precision, recall, F1-score, confusion matrix.
- Regression: MAE, MSE, RMSE, and R-squared.

A good ML model is not simply the most complex one; it is the model that generalizes reliably and meets the real problem's requirements.