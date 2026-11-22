# LLM Context Windows and Token Budgeting

An LLM context window limits how much tokenized input and output the model can handle in one request. It is not the same as a character limit, and the full budget is not available for user text alone.

## What consumes context?

Depending on the API and model, the request budget may include:
- System and developer instructions.
- Conversation history and the latest user message.
- Tool descriptions and tool-call results.
- Retrieved RAG passages.
- Structured output instructions.
- The generated response or reserved output tokens.

Some platforms count hidden formatting or multimodal content differently. Check the model's documentation and use its tokenizer or token-counting utility where available.

## A simple budgeting method

Let C be the model's usable context capacity, I the input tokens, and O the output reserve. Keep a safety margin M for formatting and platform-specific overhead:

    I + O + M <= C

For example, if a model supports 32,000 tokens, reserving 3,000 for output and 1,000 for overhead leaves at most 28,000 for the remaining input. This is an illustration, not a recommendation for every model.

## Prioritize context instead of truncating blindly

1. Keep essential system instructions and current user constraints.
2. Include the most relevant recent conversation turns.
3. Retrieve only passages that help answer the current question.
4. Deduplicate repeated material.
5. Summarize older history while preserving decisions, dates, and source references.
6. Truncate at meaningful boundaries such as message or document sections.
7. Tell the model when context is incomplete rather than implying it has seen everything.

A summary can lose nuance. Preserve exact values and provenance when they matter for calculations, compliance, or citations.

## RAG-specific budgeting

A retrieval pipeline should not automatically stuff the top 20 chunks into the prompt. Estimate the token size of each chunk, rank by relevance, remove duplicates, and reserve room for the answer. Consider diversity across sources and include metadata that helps the model cite the correct passage.

## Long context is not perfect memory

A model may miss details in a long prompt, especially when relevant evidence is buried among unrelated content. Test retrieval and answer quality at realistic context lengths. A larger window can increase cost and latency and does not eliminate the need for good context selection.

## Common mistakes

- Counting characters instead of tokens.
- Forgetting tool schemas and history in the input budget.
- Reserving no room for the answer.
- Truncating the beginning or end without considering message boundaries.
- Summarizing away critical constraints or source attribution.
- Treating retrieved content as trusted instructions.

## Practice

Build a small context packer that estimates token counts, reserves output space, ranks retrieved chunks, and stops before the input budget is exceeded. Test it with long documents and duplicated chunks.

## References

- [OpenAI: Tokenization](https://platform.openai.com/tokenizer)
- [OpenAI API: Conversation state](https://platform.openai.com/docs/guides/conversation-state)

**Key takeaway:** Context engineering is a selection problem. Fit the most useful evidence and constraints into a measured token budget while leaving room for the response.