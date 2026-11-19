# Fine-Tuning vs Retrieval-Augmented Generation

Fine-tuning and RAG are two different ways to adapt an LLM application. They solve overlapping but not identical problems.

## Fine-tuning

Fine-tuning updates model weights using examples for a target behavior or domain. It can teach a consistent format, style, task pattern, or specialized response behavior. It does not automatically provide reliable access to changing facts.

## Retrieval-Augmented Generation

RAG retrieves relevant external documents at inference time and places them in the model's context. It is useful when answers must reflect private, current, or traceable source material. The model weights need not change when the knowledge base is updated, though the index and documents do.

## Comparison

| Question | Fine-tuning | RAG |
|---|---|---|
| Changes model weights? | Yes | Usually no |
| Updates facts quickly? | Requires another training cycle | Update source/index |
| Source citations? | Not inherent | Can be designed into retrieval |
| Learns output behavior? | Often useful | Prompt/context may help |
| Main failure risk | Overfitting, forgetting, poor data | Bad retrieval, stale or unsafe sources |

## Decision guide

- Start with prompting and a representative evaluation set.
- Choose RAG when the challenge is access to external knowledge, freshness, or source traceability.
- Consider fine-tuning when the challenge is repeatable task behavior that prompting cannot achieve reliably.
- Combine them only when evaluation shows each addresses a distinct requirement.

## Exercise

For a university assistant that must answer from the latest course handbook and consistently return a strict JSON schema, identify which requirement points toward RAG and which may benefit from constrained output or fine-tuning. Explain how you would evaluate both.

**Takeaway:** Choose based on the failure mode you need to solve, not because one technique is newer.