# Cost-Sensitive Machine Learning

Not all prediction errors have equal consequences. Cost-sensitive learning makes those differences explicit instead of treating every mistake as equally harmful.

## Build a cost model
For binary classification, define costs for:
- False negatives: a real positive case is missed.
- False positives: a negative case is incorrectly flagged.
- True positives and true negatives: include operational costs or benefits where relevant.

A simple expected-cost calculation combines the confusion-matrix counts with the corresponding costs.

## Ways to incorporate costs
- Adjust class weights during training.
- Tune the classification threshold on validation data.
- Optimize an application-specific expected-cost metric.
- Use decision analysis when actions have different downstream outcomes.

## Avoid false precision
Costs should come from stakeholders, observed operations, or defensible estimates. Run sensitivity analysis when costs are uncertain. Consider fairness, human review capacity, and the harm of incorrect decisions.

## Evaluation workflow
1. Define the action taken after each prediction.
2. Agree on the cost assumptions.
3. Train a baseline model.
4. Select weights or thresholds on validation data.
5. Report both standard metrics and expected cost.
6. Monitor costs after deployment.

## Exercise
Create a confusion matrix and calculate expected cost for three thresholds. Explain how the preferred threshold changes when false-negative cost increases.

**Key point:** Optimize the decision the system supports, not a metric detached from its consequences.