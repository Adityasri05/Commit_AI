# Evaluating LLM Applications with a Repeatable Test Set

A few impressive demos do not establish that an LLM application is reliable. Evaluation needs representative tasks, explicit criteria, and repeatable runs.

## Build an evaluation dataset
Include realistic user questions, expected behavior, relevant source documents, and known edge cases. Cover ordinary requests, ambiguous inputs, unsupported questions, adversarial prompts, and long-context cases. Remove private data unless it is authorized and appropriately protected.

## Useful dimensions
- **Correctness:** Is the answer factually and task-wise right?
- **Groundedness:** Are claims supported by supplied sources?
- **Relevance:** Does it answer the actual question?
- **Completeness:** Are required parts missing?
- **Safety:** Does it respect policy, privacy, and tool permissions?
- **Latency and cost:** Is the system practical to operate?

## Evaluation methods
Use deterministic checks for schema validity and exact requirements. Use human review for nuanced quality. LLM-as-judge can scale some comparisons, but calibrate it against human labels and watch for judge bias.

## Regression workflow
1. Version the dataset and prompt/model configuration.
2. Run the same cases after each meaningful change.
3. Compare scores by category, not only one average.
4. Inspect failures and preserve examples as regression tests.
5. Re-evaluate after model, retrieval, or data changes.

## Exercise
Create 15 test cases for a small study assistant. Record pass/fail criteria and compare two prompt versions.

**Key point:** Evaluate the complete application—including retrieval, tools, and validation—not just the base model.