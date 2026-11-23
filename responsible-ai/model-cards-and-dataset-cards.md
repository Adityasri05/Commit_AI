# Model Cards and Dataset Cards

A model's accuracy score does not explain what the model is intended to do, where it may fail, or whether its training data is appropriate. Model cards and dataset cards make important context visible to people who build, evaluate, deploy, and use ML systems.

They support transparency, but they do not replace independent testing, governance, legal review, or human oversight.

## Model cards

A model card documents a particular model or version. Useful sections include:

- **Model details:** name, version, owner, architecture, license, and release date.
- **Intended use:** supported tasks, target users, and expected operating conditions.
- **Out-of-scope use:** uses the model was not designed or evaluated for.
- **Training data:** data sources and high-level selection or preprocessing methods.
- **Evaluation:** datasets, metrics, baselines, and evaluation conditions.
- **Subgroup performance:** relevant group-level results where lawful, ethical, and statistically meaningful.
- **Limitations:** known failure modes, uncertainty, and conditions where performance is weak.
- **Ethical considerations:** potential harms, misuse risks, and mitigation measures.
- **Operational guidance:** monitoring, review, and update expectations.

Report sample sizes and uncertainty when presenting subgroup metrics. A small subgroup estimate may be unstable; do not overstate it.

## Dataset cards

A dataset card focuses on the data itself, not the model trained from it. Document:

- Data origin, collection process, and provenance.
- Intended and prohibited uses.
- Licensing, consent, and access restrictions.
- Population, geography, time period, and sampling method.
- Label definitions, annotator process, and known disagreement.
- Missingness, known biases, and representation gaps.
- Transformations, filtering, and version history.
- Privacy, retention, and deletion considerations.

A dataset card should distinguish observed facts from assumptions. If demographic or sensitive attributes were not collected, state that clearly instead of guessing them.

## Keep documentation tied to versions

A model card should identify the exact model artifact and evaluation dataset. If weights, preprocessing, tokenizer, or data change, decide whether the documentation and evaluation must be updated.

A practical workflow is:
1. Create a draft card during data and model design.
2. Fill in training and evaluation evidence as experiments run.
3. Review limitations and intended use before release.
4. Publish the card beside the versioned artifact.
5. Update it after incidents, new evaluations, or material changes.

## Compact starter template

```text
Model / dataset name and version:
Owner and contact:
Intended use:
Out-of-scope use:
Data sources and provenance:
Evaluation setup:
Metrics and subgroup results:
Known limitations and failure modes:
Privacy, licensing, and access notes:
Human oversight and escalation:
Last reviewed:
```

Do not fill missing evidence with optimistic claims. Mark unknown fields as unknown and assign an owner to resolve important gaps.

## Practice

Write a model card for a student-support classifier and a dataset card for its training data. Include class definitions, collection period, subgroup evaluation limits, privacy safeguards, and scenarios where a human must review the prediction.

## References

- [Mitchell et al., Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Gebru et al., Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [Hugging Face: Model Cards](https://huggingface.co/docs/hub/model-cards)

**Key takeaway:** Model cards describe a model's behavior and boundaries; dataset cards describe the data behind it. Both should be versioned, evidence-based, and honest about what is unknown.