# Hybrid Retrieval and Reranking in RAG

A RAG system can fail even when its language model is capable: the retriever may not return the passages needed to answer the question.

## Sparse and dense retrieval
- **Sparse retrieval** such as BM25 matches lexical terms and is effective for exact names, identifiers, and rare keywords.
- **Dense retrieval** uses embeddings to find semantic similarity, including paraphrases.
- **Hybrid retrieval** combines both candidate sets to balance lexical and semantic matching.

## Reranking
A reranker takes a smaller candidate set and scores query-document pairs more carefully. A common pattern is:
1. Retrieve candidates with sparse and dense search.
2. Merge candidates, often using reciprocal rank fusion.
3. Rerank the top candidates with a cross-encoder or suitable reranking model.
4. Send only relevant, deduplicated passages to the generator.

## Evaluation
Measure retrieval recall at k, mean reciprocal rank, or nDCG using a labeled query set. Also evaluate final answer faithfulness and citation support. Compare latency and cost as well as relevance.

## Practical cautions
- Preserve document IDs and source metadata through every stage.
- Avoid sending duplicate or near-duplicate chunks.
- Do not assume a higher reranker score means the passage supports the answer.
- Test exact-match and semantic queries separately.

## Exercise
Compare BM25, dense retrieval, and hybrid-plus-reranking on 20 representative questions. Inspect failures rather than relying only on an aggregate score.

**Key point:** Retrieval quality is measurable; improve it with a query set, not intuition alone.