# Evaluating Retrieval-Augmented Generation

RAG quality depends on both retrieval and answer generation. Evaluate the two stages separately as well as end to end.

## Retrieval metrics
- Recall@k: fraction of relevant passages found in the top k results.
- Precision@k: fraction of top k retrieved passages that are relevant.
- Mean reciprocal rank: rewards placing the first relevant result near the top.

Ground-truth relevance labels should be created or reviewed for representative queries.

## Answer metrics
- Faithfulness: are claims supported by retrieved context?
- Answer relevance: does the response address the question?
- Completeness: does it cover the necessary points?
- Citation correctness: do citations point to passages that support the associated claims?

## Build an evaluation set
Include easy, ambiguous, multi-hop, out-of-scope, and adversarial questions. Record expected source passages and acceptable answer elements.

## Error analysis
Classify failures as parsing, chunking, embedding, retrieval, reranking, prompt construction, generation, or citation errors. Fix the stage causing the problem rather than blindly changing the LLM.

## Exercise
Create 20 questions over a small document collection. Label relevant chunks, compute Recall@5, and manually review whether generated answers are grounded.