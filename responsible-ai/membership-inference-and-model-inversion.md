# Membership Inference and Model Inversion Risks

A trained model can reveal information about its training data through its outputs or internal behavior. Privacy evaluation should consider not only whether raw records are stored, but also what an attacker can infer from model access.

## Two different attack goals

### Membership inference

The attacker asks: **Was this particular record used to train the model?**

A common signal is that models may behave differently on training examples than on unseen examples, especially when they overfit. An attacker may compare confidence scores, losses, or other observable outputs against a threshold or a learned attack model.

### Model inversion

The attacker tries to infer sensitive features or reconstruct aspects of an input from model outputs, gradients, or repeated queries. The goal is not necessarily to prove that a record was in the training set; it may be to infer a private attribute associated with a person or example.

These attacks have different objectives and should be evaluated separately.

## Threat model first

Document:
- what the attacker knows about the model and training process;
- whether access is black-box, score-based, or white-box;
- what queries and rate limits are available;
- which records or attributes are sensitive;
- what success means and what false positives are acceptable.

A privacy claim without an explicit attacker model is difficult to interpret.

## How to evaluate defensively

For membership inference, use a controlled audit dataset with known members and non-members. Compare attack performance against a baseline and report ROC-AUC, attack advantage at relevant false-positive rates, and subgroup results. Avoid reporting only accuracy when member and non-member groups are imbalanced.

For inversion risk, define the sensitive attribute or reconstruction target in advance. Measure whether the attack reveals information beyond a reasonable baseline and whether the result is actionable. Use synthetic or authorized test data for experiments.

## Risk factors

- large train-test generalization gaps and overfitting;
- small or highly sensitive training datasets;
- unusually informative confidence or probability outputs;
- repeated unrestricted queries;
- access to gradients, embeddings, or intermediate representations;
- rare records that are easy to distinguish from the broader population.

No single signal proves that a model leaks private information. Risk depends on the model, data, interface, attacker capability, and population.

## Mitigations

- **Data minimization:** Do not train on sensitive fields unless necessary and authorized.
- **Generalization controls:** Use sound validation, regularization, and early stopping to reduce avoidable memorization.
- **Differential privacy:** Apply a correctly configured privacy mechanism such as DP-SGD when its utility and threat assumptions fit the task. Track the privacy budget.
- **Access controls:** Authenticate callers, limit query rates, and protect gradients and internal representations.
- **Output design:** Expose only what the product needs. Coarse outputs can reduce some attack surfaces, but hiding confidence scores alone is not a privacy guarantee.
- **Governance:** Define retention, deletion, incident response, and independent privacy review.

## Practice

Train two classifiers on the same authorized synthetic dataset: one deliberately overfit and one regularized. Build a simple membership audit using held-out members and non-members. Compare attack AUC, generalization gap, and subgroup results. Do not use real personal data for an uncontrolled attack demonstration.

## References

- [Shokri et al.: Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)

**Key takeaway:** Model privacy requires testing what an attacker can infer from the trained system, not merely checking whether the training database is exposed.