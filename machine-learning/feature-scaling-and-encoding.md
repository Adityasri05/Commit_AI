# Feature Scaling and Categorical Encoding

Many estimators expect numeric inputs, but raw features may use different units or contain categories. Scaling and encoding make these inputs usable without changing the meaning of the prediction target.

## Scaling numeric features

- **Standardization:** subtract the training mean and divide by the training standard deviation. Often useful for linear models, SVMs, PCA, and neural networks.
- **Min-max scaling:** maps a feature to a chosen interval, commonly [0, 1]. Sensitive to extreme values.
- **Robust scaling:** uses the median and interquartile range; can be more stable with outliers.
- **No scaling:** often fine for tree-based models, whose split decisions depend on feature ordering rather than distance.

Scaling does not automatically make a model better. Choose it based on the algorithm and validate the result.

## Encoding categories

- **One-hot encoding:** creates indicator columns for categories; useful for nominal categories with manageable cardinality.
- **Ordinal encoding:** maps categories to integers. Use only when an order is meaningful or when the estimator can handle arbitrary category codes appropriately.
- **Target encoding:** replaces a category with a target-derived statistic. It must be computed in a leakage-safe, cross-fitted way.

## Reproducible scikit-learn pattern

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

numeric = ["age", "income"]
categorical = ["city", "course"]

numeric_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
category_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", category_pipe, categorical)
])
model = make_pipeline(preprocess, LogisticRegression(max_iter=1000))
```

Fit this complete pipeline on training data; do not preprocess the full dataset first.

## Exercise

For a dataset with salary, age, city, and satisfaction level (low/medium/high), choose a scaler and encoder for each feature. Explain which choices depend on the model.

**Takeaway:** Treat preprocessing as part of the model and fit every learned transformation on training data only.