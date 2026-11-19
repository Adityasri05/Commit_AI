# Feature Selection: Choosing Useful Inputs

Feature selection chooses a subset of the available input variables. It can reduce cost, improve interpretability, reduce noise, and sometimes improve generalization. It is different from feature extraction (such as PCA), which creates transformed features.

## Three families

- **Filter methods:** Rank features using statistics such as mutual information, correlation, or chi-square scores. They are relatively fast and model-independent.
- **Wrapper methods:** Evaluate subsets by repeatedly fitting a model, such as recursive feature elimination. They can be expensive and must use validation correctly.
- **Embedded methods:** Select or shrink features during fitting, for example L1-regularized linear models or tree-based importance measures.

## Leakage-safe workflow

Feature selection is a learned transformation. Fit it inside a scikit-learn Pipeline so each cross-validation fold selects features using only its training portion.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    SelectKBest(score_func=mutual_info_classif, k=10),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)
```

The example assumes numeric features. Choose a selector compatible with your feature representation and task.

## Common pitfalls

- Selecting features using the full dataset before splitting leaks information.
- Correlated features can make rankings unstable.
- A feature's predictive association does not imply causation.
- Tree feature importance can favor variables with many possible split points; compare with permutation importance and domain knowledge.

## Exercise

Compare all features against the top 10 selected features using identical cross-validation folds. Record score, variance, and training time. Do not select features on the test set.

**Takeaway:** Feature selection is part of model fitting and belongs inside the validation pipeline.