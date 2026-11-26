# Speculative Decoding for Large Language Models

Autoregressive language models normally generate one token at a time. Each token requires another forward step through the target model, which can make generation slow when memory bandwidth or model computation is the bottleneck.

Speculative decoding uses a smaller draft model to propose several tokens and a larger target model to verify them together. When the draft is good enough, several tokens can be accepted per target-model pass.

## The two-model workflow

1. The draft model generates a short candidate continuation.
2. The target model evaluates the proposed continuation in a parallel verification step.
3. A verification rule accepts a prefix of the draft tokens.
4. At the first rejected token, the algorithm samples or selects a correction according to the chosen decoding method.
5. The process repeats from the new prefix.

The draft model is not trusted to determine the final distribution. It proposes candidates; the target model remains responsible for the output distribution.

## Why it can be faster

A target model can often score multiple candidate positions in one forward pass more efficiently than generating each position separately. If many draft tokens are accepted, the system produces more output tokens per target-model pass.

Speedup depends on:
- how closely the draft distribution matches the target distribution;
- average accepted tokens per round;
- draft and target model sizes;
- batch size and sequence length;
- memory bandwidth, kernel efficiency, and serving overhead.

A draft model that is too weak may cause frequent rejection and add overhead instead of reducing latency.

## Distribution-preserving verification

For a draft distribution q and target distribution p at a proposed token, speculative sampling can accept a draft token with probability:

    min(1, p(token) / q(token))

If a token is rejected, the algorithm samples from an appropriately constructed residual distribution so that the final sequence follows the target distribution. This correction is important: simply accepting every plausible draft token would change the target model's sampling behavior.

Greedy speculative decoding uses a related verification procedure to preserve the target model's greedy output.

## A conceptual example

Suppose the draft model proposes:

    "The capital of France is Paris"

The target model scores the candidate positions. If the early tokens are accepted, the system can advance several positions at once. If a token is rejected, generation resumes from the verified prefix with a target-consistent correction.

The example is illustrative: acceptance is based on token distributions, not a human judgment that a sentence sounds reasonable.

## When to use it

Speculative decoding is attractive when:
- the target model is expensive to decode;
- a smaller draft model is available;
- the draft model has similar token preferences for the target workload;
- the serving stack supports efficient multi-token verification.

It may help less for very large batches, workloads dominated by prefill rather than decode, or tasks where draft acceptance is low.

## Engineering and evaluation

Benchmark against ordinary target-model decoding with the same prompts, output settings, hardware, and concurrency.

Measure:
- output tokens per second;
- time to first token and inter-token latency;
- accepted draft tokens per round;
- target and draft GPU utilization;
- peak memory and total memory bandwidth;
- output equivalence or distributional correctness for the selected algorithm.

Do not compare only a single short prompt. Include varied prompt lengths, output lengths, batch sizes, and decoding temperatures.

## Common pitfalls

- Treating the draft model's output as the final answer without verification.
- Assuming every speculative algorithm preserves the exact target distribution.
- Ignoring the memory cost of loading both models.
- Reporting speedup without measuring acceptance rate and latency percentiles.
- Comparing different decoding settings or output quality.
- Assuming a smaller model is automatically a useful draft model.

## Practice

Use a small causal language model as the target and a smaller related model as the draft. Compare ordinary decoding with assisted generation on a fixed prompt set. Record accepted tokens, time to first token, inter-token latency, and memory use.

## References

- [Leviathan et al.: Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chen et al.: Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Hugging Face: Assisted Generation](https://huggingface.co/docs/transformers/generation_strategies)

**Key takeaway:** Speculative decoding accelerates autoregressive generation by letting a draft model propose and a target model verify. The verification algorithm, acceptance rate, and serving hardware determine whether it is both fast and faithful.