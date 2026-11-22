# Prompt Caching and LLM Cost-Latency Optimization

Many LLM applications repeatedly send the same large instructions, tool schemas, or reference prefix. Provider-supported prompt caching can reduce repeated processing for eligible input prefixes.

Caching behavior is provider- and model-specific. It may have minimum token thresholds, time-to-live rules, and restrictions on which request parts are cacheable. Always check the current API documentation.

## Design prompts for reusable prefixes

A practical ordering is:

1. Stable system instructions.
2. Stable tool definitions and output schemas.
3. Reusable policy or domain reference material.
4. Variable conversation state.
5. The current user request.

Keep the reusable prefix byte-for-byte or token-for-token stable where the provider requires exact prefix matches. Inserting a timestamp or request ID near the beginning can invalidate reuse for everything after it.

## Measure whether caching helps

Track:
- Cache-read and cache-write token counts, if exposed.
- Cache hit rate.
- Input and output tokens.
- Time to first token and total latency.
- Cost per successful task.
- Quality and safety regressions.

Do not assume that a cache hit means the application is cheaper or faster in every case. Verify actual billing and latency behavior for the chosen model.

## Prompt caching vs response caching

**Prompt caching** reuses processing for a repeated prompt prefix. The model still generates a fresh response.

**Response caching** reuses a previously generated answer. Exact response caching is appropriate only when the request and relevant context are equivalent. A semantic cache may treat similar questions as equivalent, which is risky when answers depend on user identity, permissions, current data, or small wording differences.

For response caches, include model and prompt version, tenant or authorization scope, relevant data version, and expiration policy in the cache key. Never share private outputs across users.

## Other efficiency techniques

- Limit unnecessary conversation history.
- Retrieve fewer, more relevant RAG chunks.
- Choose a smaller model for low-risk subtasks and validate quality.
- Use structured output only when downstream code needs it.
- Batch independent requests when latency permits.
- Stream responses for better perceived latency, while still handling incomplete output.
- Set token and tool-call budgets to prevent runaway loops.

## Common pitfalls

- Putting dynamic values before a stable prefix.
- Assuming caching is enabled for every model or request size.
- Reusing generated answers after source data has changed.
- Ignoring authorization boundaries in response caches.
- Optimizing token cost while degrading answer quality.
- Logging sensitive prompts just to calculate cache metrics.

## Practice

Take a repeated customer-support prompt. Separate stable instructions from per-request content, measure token usage before and after, and design a response-cache key that prevents cross-account leakage.

## References

- [OpenAI API: Prompt caching](https://platform.openai.com/docs/guides/prompt-caching)
- [OpenAI API: Production best practices](https://platform.openai.com/docs/guides/production-best-practices)

**Key takeaway:** Reuse stable prompt prefixes where supported, but treat response caching as a separate correctness and privacy problem.