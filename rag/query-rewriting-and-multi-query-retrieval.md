# Query Rewriting and Multi-Query Retrieval for RAG

A user's question is not always phrased in the way a document collection is indexed. Query rewriting transforms the question into a retrieval-friendly form while preserving its meaning and constraints.

It complements chunking, hybrid retrieval, and reranking; it does not replace them.

## Common strategies

### Conversational query resolution

A user may ask, "What about its memory use?" after discussing a specific model. Rewrite the follow-up into a standalone query that includes the resolved model name, while retaining the original question for traceability.

### Query expansion

Add useful synonyms, acronyms, or domain terms. This can improve recall when documents use different terminology, but overly broad expansions can introduce noise.

### Multi-query retrieval

Generate several complementary search queries, retrieve candidates for each, then merge and deduplicate the results. Reciprocal rank fusion is one method for combining ranked lists without requiring their raw scores to be directly comparable.

### Hypothetical document embeddings (HyDE)

A model drafts a hypothetical answer-like passage and embeds it for retrieval. The generated passage is a search aid, not evidence. It may contain invented facts, so only retrieved source documents should support the final answer.

## A controlled pipeline

1. Preserve the original user question.
2. Resolve references using conversation context when available.
3. Generate a small number of rewrites with a constrained schema.
4. Check that names, dates, negations, and user constraints remain intact.
5. Retrieve for the original and rewritten queries.
6. Merge, deduplicate, and rerank candidates.
7. Generate from retrieved source passages only.
8. Record which query variant retrieved each passage for debugging.

## Example

Original question: "Does it support streaming and what are the limits?"

If the prior turn clearly discussed a particular API, a standalone rewrite might be: "For the previously discussed API, what streaming response modes are supported and what documented limits apply?"

Do not guess the API if the conversation is ambiguous. Ask a clarifying question instead of inventing the referent.

## Evaluation

Build a labeled query set with conversational follow-ups, acronyms, misspellings, multi-part questions, and questions containing negation. Compare:
- Recall@k before and after rewriting.
- Precision@k and duplicate rate.
- Answer faithfulness and citation correctness.
- Latency, token cost, and failure rate.

Inspect rewrites that change the user's intent. A rewrite that retrieves more documents but drops "only", "not", a date range, or a product version may be harmful.

## Safety and privacy

Treat generated rewrites as untrusted model output. Do not allow them to alter access-control filters, tenant IDs, or document permissions. Authorization must be enforced by application code during retrieval.

## Practice

Create 15 follow-up questions over a small documentation set. Compare original-query retrieval with query resolution and three-query expansion. Record both recall gains and cases where a rewrite changes the meaning.

## References

- [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

**Key takeaway:** Rewriting should clarify and diversify retrieval without changing the user's intent or bypassing access controls.