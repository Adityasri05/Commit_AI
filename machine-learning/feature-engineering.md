# Feature Engineering for Machine Learning

Feature engineering transforms raw inputs into representations that make useful patterns easier for a model to learn.

## Common techniques

### Missing values
- Inspect why values are missing.
- Impute numeric values with a justified statistic or model.
- Use an explicit category for missing categorical values when appropriate.
- Consider a missingness indicator if absence itself carries information.

### Categorical variables
- One-hot encoding represents categories with binary columns.
- Ordinal encoding is appropriate when categories have a meaningful order.
- Target encoding must be fitted carefully within training folds to avoid leakage.

### Scaling
- Standardization: subtract the training mean and divide by the training standard deviation.
- Min-max scaling: map values to a chosen range.
- Tree-based models often need less scaling than distance- or gradient-based models.

### Date and text features
Dates can yield day of week, month, elapsed time, or seasonality. Text can be represented using token counts, TF-IDF, or learned embeddings.

## Avoid data leakage
Fit transformations on the training data only, then apply the learned transformation to validation and test data. In scikit-learn, a Pipeline helps keep preprocessing and model fitting together.

## Example workflow
1. Split the dataset.
2. Identify numeric, categorical, date, and text columns.
3. Build preprocessing steps for each column type.
4. Fit preprocessing on training data.
5. Train and evaluate the model.
6. Review feature importance or error patterns cautiously.

## Exercise
For a dataset containing age, city, monthly income, and signup date, propose a preprocessing method for each column. Explain which transformations must be learned from training data only.

**Key idea:** Better features can improve a simple model, but every transformation should be justified and validated.