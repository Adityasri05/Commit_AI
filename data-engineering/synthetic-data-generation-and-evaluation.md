# Synthetic Data Generation and Evaluation

Synthetic data is artificially generated data designed to resemble selected properties of a real dataset. It can support software testing, prototyping, simulation, class-imbalance experiments, and research when access to real records is restricted.

Synthetic data is not automatically private, representative, or useful. A generator can memorize training records, reproduce harmful biases, or create unrealistic combinations. Evaluate it against the intended use before relying on it.

## Why generate synthetic data?

- **Development and testing:** create predictable fixtures for edge cases.
- **Prototyping:** let teams test schemas and workflows before production data access.
- **Data augmentation:** increase examples for underrepresented patterns, when the generated examples are valid.
- **Simulation:** explore scenarios that are difficult or costly to observe.
- **Research:** share selected statistical patterns when raw data cannot be shared, subject to privacy assessment.

Synthetic data should not be presented as a replacement for real-world validation.

## Generation approaches

### Rule-based generation

Domain rules create records with known constraints. This is excellent for test fixtures and edge cases, but may not capture complex real-world dependencies.

### Statistical models

Models such as Gaussian copulas approximate marginal distributions and relationships between columns. They can be effective for some tabular datasets but may struggle with complex nonlinear or multimodal structure.

### Deep generative models

GANs, VAEs, autoregressive models, and diffusion models can capture more complex patterns. They require careful training and may be expensive or difficult to validate.

### Resampling and augmentation

Methods such as bootstrap sampling and SMOTE create additional observations from existing data. They are useful in some settings but are not equivalent to learning a full data-generating process. SMOTE, for example, interpolates between minority-class neighbors and can create implausible records when feature geometry is unsuitable.

## A tabular generation example with SDV

The following illustrates a basic workflow for a single table. Review metadata and constraints manually before using generated rows.

```python
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# real_data is a pandas DataFrame containing approved training data.
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=real_data)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)

synthetic_data = synthesizer.sample(num_rows=1000)
print(synthetic_data.head())
```

This example does not provide a privacy guarantee. Check the installed SDV version for current API details, supported column types, constraints, and evaluation utilities.

## Evaluate synthetic data from several angles

### 1. Structural validity

- Required columns and data types are present.
- Values satisfy domain constraints and valid ranges.
- Keys and relationships remain consistent.
- Dates, sequences, and category combinations are plausible.

### 2. Statistical fidelity

Compare real and synthetic distributions:

- Numeric columns: quantiles, histograms, Wasserstein distance.
- Categorical columns: category frequencies and rare-category coverage.
- Relationships: correlations, conditional distributions, and important interactions.
- Time series: autocorrelation, seasonality, trend, and temporal ordering.

A close marginal distribution does not prove that important multivariate relationships are preserved.

### 3. Downstream utility

A useful test is **train on synthetic, test on real (TSTR)**:

1. Train a model using synthetic training data.
2. Evaluate it on a held-out real test set.
3. Compare with a model trained on real training data.
4. Report the task metric and subgroup results.

Also test the reverse direction when relevant. Utility is task-specific: data useful for a classifier may be poor for forecasting or rare-event detection.

### 4. Privacy risk

Assess whether generated records reveal information about training examples:

- Nearest-neighbor similarity to real records.
- Duplicate and near-duplicate rates.
- Membership-inference or attribute-inference attacks where appropriate.
- Exposure of rare combinations or unique individuals.
- Risk from auxiliary information an attacker could possess.

A synthetic label does not make data anonymous. Strong privacy claims require a defined threat model and appropriate formal guarantees, such as a correctly implemented differential-privacy mechanism where suitable.

## Common mistakes

- Comparing only column means and declaring the data realistic.
- Ignoring rare groups and unusual but valid combinations.
- Using synthetic data as the only evaluation set.
- Assuming a GAN or VAE cannot memorize training examples.
- Treating SMOTE as a universal solution to class imbalance.
- Publishing generated data without a privacy and disclosure review.
- Forgetting that generator quality can degrade when the source data is biased or incomplete.

## Practice

Choose a small tabular dataset with numeric and categorical columns. Generate synthetic rows, validate schema constraints, compare marginal and pairwise distributions, and run a TSTR experiment. Then inspect nearest-neighbor similarity and explain what privacy risks remain.

## References

- [SDV documentation](https://docs.sdv.dev/sdv/)
- [Synthetic Data: A Review of Generation and Evaluation Methods](https://arxiv.org/abs/2307.01752)
- [NIST: De-Identifying Government Datasets](https://www.nist.gov/publications/de-identifying-government-datasets-techniques-and-governance)

**Key takeaway:** Synthetic data must be evaluated for validity, statistical fidelity, downstream utility, and privacy risk. No single score establishes that it is safe or suitable.
