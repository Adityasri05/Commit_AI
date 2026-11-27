# Rotary Positional Embeddings and ALiBi

Self-attention compares token representations, but attention alone does not tell a Transformer where tokens occur in a sequence. Positional methods inject information about order and distance into the attention computation.

Two important approaches are Rotary Positional Embeddings (RoPE) and Attention with Linear Biases (ALiBi). They encode position differently and have different implementation and context-length trade-offs.

## Why position matters

Consider the sentences:

- "The student helped the teacher."
- "The teacher helped the student."

The same words appear, but their order changes the meaning. A sequence model needs access to token positions or relative order to distinguish such cases.

## Rotary Positional Embeddings (RoPE)

RoPE applies a position-dependent rotation to pairs of dimensions in the query and key vectors.

For a token at position m, each two-dimensional pair is rotated by an angle proportional to m and a frequency assigned to that pair. In compact form:

- q_m' = R_m q_m
- k_n' = R_n k_n

The attention score uses the rotated vectors. Because rotations compose relative to their angle difference, the dot product carries information about the relative displacement between positions m and n.

Important properties:

- Position is incorporated into queries and keys rather than simply added to token embeddings.
- Different dimension pairs use different frequencies, allowing multiple positional scales.
- RoPE is widely used in decoder-only LLMs.
- Long-context extensions often modify frequency scaling or interpolation; they should be validated rather than assumed to generalize automatically.

RoPE is not a guarantee of unlimited context extrapolation. Training length, frequency choices, scaling method, and attention implementation all matter.

## ALiBi: Attention with Linear Biases

ALiBi adds a distance-dependent bias directly to attention logits. For a causal model, a head can receive a penalty proportional to how far a key position is behind the query position.

For head h and query/key positions i and j, a simplified causal bias is:

bias_h(i, j) = -slope_h * (i - j), for j <= i

The slope differs across heads. The attention score becomes:

score_h(i, j) = q_i dot k_j / sqrt(d_k) + bias_h(i, j)

The causal mask still prevents access to future tokens. The linear bias encourages heads to favor nearby context to different degrees, while allowing attention to more distant positions.

ALiBi does not add a separate positional embedding vector. It changes attention scores, so the bias must be integrated consistently with masking and any optimized attention kernel.

## Comparison

| Aspect | RoPE | ALiBi |
|---|---|---|
| Where position enters | Rotates queries and keys | Adds a bias to attention logits |
| Main signal | Relative phase/rotation | Distance penalty |
| Per-head behavior | Frequencies across dimensions | Head-specific slopes |
| Long-context behavior | Often needs scaling strategies | Can extrapolate in some settings, but still requires testing |
| Implementation concern | Correct rotation and frequency handling | Correct bias and causal-mask broadcasting |

## Implementation considerations

When implementing RoPE:

1. Confirm whether the model rotates all head dimensions or a configured subset.
2. Match the model's frequency base and position convention.
3. Handle odd dimensions and tensor layout according to the architecture.
4. Apply the same positional convention during training, prefill, and cached decoding.
5. Test full-sequence attention against token-by-token decoding.

When implementing ALiBi:

1. Generate the intended slope for each attention head.
2. Align query and key position indices, including cached offsets.
3. Combine the bias with causal and padding masks correctly.
4. Check numerical stability and memory use for long sequences.

## Common mistakes

- Adding RoPE to token embeddings instead of applying it to the intended query/key tensors.
- Applying the wrong position offset when using a KV cache.
- Assuming that a longer configured context automatically means reliable long-context reasoning.
- Mixing a model checkpoint with a different positional convention.
- Broadcasting an ALiBi tensor along the wrong batch, head, query, or key dimension.
- Comparing methods without controlling training data, context length, and compute.

## Practice

Implement a small causal attention layer with RoPE and another with ALiBi. Verify that future tokens are masked, compare outputs for a short sequence, and test cached decoding against full-prefix decoding. Then evaluate retrieval of information placed near the beginning, middle, and end of longer contexts.

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409)
- [Hugging Face Transformers: Internal positional embedding utilities](https://huggingface.co/docs/transformers)

**Key takeaway:** RoPE encodes position through rotations of queries and keys; ALiBi encodes it through distance-based attention biases. Correct offsets, masks, and long-context evaluation are essential for both.
