# Prompt Engineering for LLM Applications

Prompt engineering is the practice of designing instructions and context to make a language model's behavior more useful and reliable.

## A useful prompt structure
1. Role or task framing.
2. Clear objective.
3. Relevant context and constraints.
4. Input data.
5. Desired output format.
6. Examples when they clarify expectations.

## Techniques
- Zero-shot: provide instructions without examples.
- Few-shot: include representative input/output examples.
- Decomposition: split a complex task into smaller steps.
- Structured output: request a schema such as JSON, then validate it in code.
- Grounding: provide source passages and instruct the model to use them.

## Reliability practices
- Separate trusted instructions from untrusted user-provided content.
- Validate structured outputs before using them.
- Treat model-generated citations as untrusted until checked.
- Test prompts on representative and adversarial inputs.
- Do not rely on a prompt alone for authorization or security.

## Evaluation
Create a small test set with expected outcomes. Track correctness, format validity, refusal behavior, latency, and cost across prompt revisions.

## Exercise
Write a prompt that extracts skills from a resume into a fixed JSON schema. Test missing fields, ambiguous skills, and malformed input, then validate the output programmatically.