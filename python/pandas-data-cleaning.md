# Data Cleaning with Pandas

Pandas provides DataFrame and Series structures for working with tabular data.

## Inspect first
```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))
```

## Common cleaning tasks
- Missing values: inspect patterns before choosing imputation or row removal.
- Duplicates: use `duplicated()` and `drop_duplicates()` only when repeated rows are truly redundant.
- Data types: parse dates and numeric columns explicitly.
- Categories: normalize spelling and whitespace carefully.
- Outliers: investigate whether values are errors or legitimate rare events.

## Example
```python
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
df["income"] = pd.to_numeric(df["income"], errors="coerce")
df["city"] = df["city"].str.strip().str.title()
```

## Avoid leakage
Perform data splitting before learning imputation values, scaling parameters, or target-informed transformations. Fit those transformations on training data only.

## Exercise
Load a public tabular dataset, summarize missing values, identify duplicates, fix incorrect data types, and document each decision. Do not delete rows without explaining why.