# Evaluating Embedding Models for Retrieval

An embedding model maps text into vectors so semantically related queries and passages can be compared. A model that produces fluent or visually meaningful vector neighborhoods is not necessarily the best model for a specific retrieval task.

Choose embeddings by measuring retrieval performance on representative queries and relevant-document labels.

## Build an evaluation dataset

A useful retrieval benchmark contains:

- realistic user queries, including short, long, ambiguous, and conversational forms;
- a corpus that resembles the documents used in production;
- relevance judgments linking each query to one or more useful passages;
- hard negatives that look similar but do not answer the query;
- metadata such as language, source type, and query category.

Use human review or carefully audited judgments for high-impact datasets. Weak or incomplete relevance labels can make a strong model look poor or a weak model look good.

## Retrieval metrics

- **Recall@k:** Does at least one relevant item appear in the first k results, or what fraction of all known relevant items is retrieved?
- **Precision@k:** What fraction of the top-k results are relevant?
- **MRR:** How early does the first relevant result appear?
- **nDCG@k:** Does the ranking place highly relevant results above less relevant ones?
- **MAP:** Summarizes precision across relevant ranks for queries with multiple relevant items.

Define the exact metric convention used by the benchmark. For example, some recall implementations divide by the total number of known relevant items, while a hit-rate version checks only whether any relevant item appears.

## Compare models fairly

Keep the retrieval pipeline fixed while comparing embedding models:

1. Use the same corpus snapshot and chunking strategy.
2. Use the same query set and relevance judgments.
3. Apply each model's documented query and document input format.
4. Use the same similarity metric where appropriate.
5. Rebuild the index for each model; do not mix vectors from different embedding spaces.
6. Report quality by query category, language, and document type.
7. Measure embedding latency, indexing throughput, vector dimension, memory, and serving cost.

Some models use different prefixes or instruction templates for queries and passages. Follow the model's intended format rather than assuming both inputs should be identical.

## Example evaluation skeleton

~~~python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "A guide to gradient descent and learning rates.",
    "How database indexes accelerate SQL queries.",
    "An introduction to model evaluation metrics.",
]
queries = ["How do I measure a classifier?"]

doc_vectors = model.encode(
    documents, normalize_embeddings=True
)
query_vectors = model.encode(
    queries, normalize_embeddings=True
)

scores = cosine_similarity(query_vectors, doc_vectors)
ranking = np.argsort(-scores[0])
print([(documents[i], float(scores[0, i])) for i in ranking])
~~~

This example demonstrates ranking only. A real benchmark must compare ranked results with relevance judgments and compute metrics such as Recall@k or nDCG@k.

## Beyond one embedding score

Embedding quality can vary by domain. Test:
- domain terminology and abbreviations;
- multilingual and code-switched queries;
- spelling mistakes and paraphrases;
- long documents and short queries;
- exact identifiers that may favor lexical search;
- out-of-domain inputs and adversarially similar passages.

A hybrid lexical-plus-vector retriever may outperform a stronger embedding model alone for identifiers, product codes, or exact phrases.

## Benchmark references

- **MTEB:** Broad benchmark tasks for text embeddings, useful for initial model discovery.
- **BEIR:** Heterogeneous information-retrieval datasets that help test generalization across retrieval domains.
- **Sentence Transformers:** Tools for encoding, training, and evaluating sentence embeddings.

Public leaderboard scores are a shortlist signal, not a substitute for an in-domain evaluation set.

## Common pitfalls

- Choosing a model from a general leaderboard without testing the actual query distribution.
- Comparing models with different chunking or preprocessing pipelines.
- Forgetting that model-specific query/document prompts can matter.
- Using only cosine similarity without checking normalization and model guidance.
- Treating a high similarity score as a calibrated probability of relevance.
- Failing to version embeddings and rebuild indexes after changing models.

## Practice

Choose two embedding models and a small corpus of technical notes. Write 25 realistic questions and label relevant passages. Compare Recall@5, MRR, nDCG@10, p95 embedding latency, and index size. Inspect failure cases before selecting a model.

## References

- [MTEB: Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb)
- [BEIR: A Heterogeneous Benchmark for Information Retrieval](https://github.com/beir-cellar/beir)
- [Sentence Transformers documentation](https://sbert.net/)

**Key takeaway:** The best embedding model is the one that retrieves the right evidence for the target corpus and users at an acceptable cost—not simply the model with the highest general leaderboard score.