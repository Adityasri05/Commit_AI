# Probability Calibration for Classifiers

A classifier can rank examples well while producing unreliable probability estimates. Calibration asks whether predictions made with confidence 0.8 are correct roughly 80% of the time across comparable cases.

## Why calibration matters
- Risk scores often drive human review or automated decisions.
- A threshold such as 0.7 is meaningful only when predicted probabilities are reasonably calibrated.
- Good accuracy or ROC-AUC does not guarantee calibrated probabilities.

## Reliability diagram
1. Collect predictions on data not used to fit the model.
2. Divide predicted probabilities into bins.
3. For each bin, compare mean predicted probability with the observed positive fraction.
4. A perfectly calibrated model lies near the diagonal.

## Common methods
- **Platt scaling:** Fits a logistic mapping from model scores to probabilities.
- **Isotonic regression:** Learns a non-decreasing mapping; flexible but needs enough calibration data.
- **Temperature scaling:** Common for neural-network logits, especially multiclass classification.

## Safe workflow
Use a dedicated calibration set or cross-validation. Never calibrate on the final test set and then report performance on that same set. Re-check calibration after distribution shifts.

## Practical exercise
Train a classifier, plot a reliability diagram before and after calibration, and compare Brier score alongside discrimination metrics.

**Key point:** Calibration improves the trustworthiness of probabilities; it does not necessarily improve ranking or classification accuracy.