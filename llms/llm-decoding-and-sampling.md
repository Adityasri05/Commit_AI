# LLM Decoding and Sampling Strategies

A language model produces a probability distribution over the next token. Decoding is the procedure that turns those probabilities into a sequence of generated tokens.

The model weights may stay unchanged while decoding settings produce outputs with very different levels of randomness, repetition, diversity, and reliability.

## Greedy decoding

Greedy decoding selects the highest-probability token at every step:

\[
x_t = \arg\max_x P(x \mid x_{<t}, prompt)
\]

It is deterministic for a fixed model and prompt, assuming deterministic inference. It is fast, but a locally best token may lead to a poor overall sequence.

## Temperature

Temperature rescales logits before softmax:

\[
P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\]

- Lower T sharpens the distribution and makes high-probability tokens more likely.
- Higher T flattens the distribution and increases randomness.
- T approaching zero behaves like greedy selection in common implementations.

Temperature changes sampling behavior; it does not make the underlying model more knowledgeable.

## Top-k and top-p sampling

- **Top-k:** Keep only the k highest-scoring next-token candidates, then renormalize.
- **Top-p (nucleus):** Keep the smallest candidate set whose cumulative probability reaches p, then renormalize.

Top-k uses a fixed candidate count. Top-p adapts the candidate set size to the shape of the distribution. They can be combined, but tuning both without a clear evaluation goal makes behavior hard to interpret.

## Beam search

Beam search keeps several partial sequences and expands them at each step. It is useful for some structured generation tasks such as transcription, but can produce generic or repetitive text for open-ended dialogue. Length normalization and stopping rules matter.

## Repetition controls

Repetition penalties, no-repeat n-gram constraints, and frequency or presence penalties modify the preference for tokens or sequences that appeared earlier. Strong penalties can damage grammar, names, code, and tasks where repetition is necessary.

## Hugging Face example

~~~python
# Assumes tokenizer and model have already been loaded.
inputs = tokenizer("Explain gradient descent simply.", return_tensors="pt")

sampled = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
)

print(tokenizer.decode(sampled[0], skip_special_tokens=True))
~~~

When do_sample is false, temperature and top-p sampling controls do not provide the expected sampling behavior. Always check the generation configuration and library version.

## Choose decoding for the task

| Task | Reasonable starting point |
|---|---|
| Deterministic extraction | Greedy or constrained decoding with schema validation |
| Open-ended brainstorming | Sampling with moderate temperature |
| Translation or transcription | Greedy or beam search, evaluated on task metrics |
| Code generation | Conservative sampling plus tests and validation |
| Factual question answering | Grounded context, conservative decoding, and source checks |

These are starting points, not guarantees. The right configuration depends on the model, prompt, and evaluation set.

## Reproducibility

Record:
- Model identifier and revision.
- Tokenizer revision.
- Prompt and system instructions.
- Temperature, top-k, top-p, beam count, penalties, and token limits.
- Random seed where supported.
- Stop sequences and structured-output constraints.

Sampling can be nondeterministic across hardware, kernels, and library versions even when a seed is set.

## Evaluation checklist

- Use task-specific metrics and human review for subjective tasks.
- Measure diversity separately from correctness.
- Track truncation, refusals, malformed outputs, and repeated text.
- Compare settings on the same prompts and examples.
- Do not select a decoding setup based on a handful of cherry-picked generations.

## Mini-project

Create a fixed set of 30 prompts. Compare greedy, temperature sampling, top-k, and top-p. Record output diversity, task success, repetition, latency, and token count. Explain which configuration is best for each prompt type.

## References

- Hugging Face generation strategies: https://huggingface.co/docs/transformers/generation_strategies
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2020): https://arxiv.org/abs/1904.09751
