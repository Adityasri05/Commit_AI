# Counterfactual Explanations and Recourse

A counterfactual explanation describes how an input could change so that a model produces a different prediction. In a decision-support setting, it can help answer: “What would need to be different for this outcome to change?”

Counterfactuals are explanations of model behavior. They are not automatically causal statements about the real world.

## Basic formulation

Let f(x) be a model prediction for an input x. A counterfactual x' seeks a nearby, valid alternative whose prediction meets a desired outcome:

    find x' that minimizes distance(x, x')
    subject to f(x') meeting the target condition

The distance function encodes what “nearby” means. A useful explanation should not merely flip a score; it should be plausible and actionable for the person affected.

## Desirable properties

- **Validity:** The model prediction for the proposed alternative reaches the requested outcome.
- **Proximity:** Changes are not unnecessarily large.
- **Sparsity:** Only a small number of features change when appropriate.
- **Plausibility:** The alternative resembles a possible real observation.
- **Actionability:** A person or organization can actually change the proposed features.
- **Diversity:** Several meaningfully different options may be more useful than one brittle answer.
- **Stability:** Small harmless changes to the input should not cause wildly different explanations.

These goals can conflict. A very close counterfactual may be unrealistic, while a plausible one may require more changes.

## Respect immutable and constrained features

Separate features into categories:

- **Immutable:** age at application time, historical events, or other facts the person cannot change.
- **Actionable:** a payment amount, requested loan term, or missing document, depending on the setting.
- **Conditionally actionable:** features that can change only over time or with substantial effort.
- **Dependent:** features that must change together because of real-world constraints.

A counterfactual generator should enforce domain constraints. It should not suggest impossible combinations, such as an income change without a plausible source or a future event that contradicts historical records.

## Example: a fictional eligibility model

Suppose a model rejects an application based on debt-to-income ratio, income, and repayment history. A counterfactual might identify that a lower debt balance or a different requested amount changes the model score.

That does not establish that paying down debt will cause a lender to approve the application. The real decision may depend on policy, missing variables, human review, and constraints not represented by the model.

## Practical workflow

1. Define the prediction target and decision threshold.
2. Identify which features are immutable, actionable, or constrained.
3. Choose a distance function appropriate to numeric, categorical, and ordinal features.
4. Generate candidate alternatives using a method such as optimization, search, or a library.
5. Reject candidates that violate domain constraints or lie far outside the training distribution.
6. Validate each candidate with the actual deployed model version.
7. Evaluate validity, proximity, diversity, plausibility, and stability.
8. Present uncertainty and limitations in language that does not promise a real-world outcome.

Libraries such as DiCE can generate diverse counterfactual examples, but domain constraints and independent validation are still required.

## Fairness and governance

Counterfactual explanations can expose unfair model behavior, but they can also create harm if they recommend changes that are impossible, discriminatory, or unsafe. Do not use protected attributes as actionable recommendations. Review explanations across relevant groups and involve domain experts and affected users.

For consequential decisions, provide an appeal or human-review path. Store the model version and explanation context so the result can be audited later.

## Common pitfalls

- Claiming a counterfactual proves causation.
- Suggesting a person change an immutable or protected characteristic.
- Producing a mathematically valid but impossible feature combination.
- Ignoring correlated features and domain constraints.
- Showing one counterfactual as if it were the only path.
- Failing to validate explanations against the deployed model.

## Practice

Train a classifier on a public tabular dataset. Generate several counterfactuals for one rejected example. Mark each feature as immutable, actionable, or dependent. Compare a simple distance-based search with a constrained method and manually reject implausible alternatives.

## References

- [Wachter, Mittelstadt, and Russell: Counterfactual Explanations](https://doi.org/10.1007/s13347-019-00375-0)
- [DiCE: Diverse Counterfactual Explanations](https://github.com/interpretml/DiCE)

**Key takeaway:** Counterfactuals can make model behavior more actionable, but only when feasibility, immutability, plausibility, fairness, and the difference between prediction and causation are made explicit.