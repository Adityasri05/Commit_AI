# Embeddings and Vector Search

An embedding is a numeric vector representing an item such as a sentence, document, image, or product. Similar items may have nearby vectors in the embedding space, depending on the model and training objective.

## Similarity measures
- Cosine similarity compares vector direction.
- Dot product combines direction and magnitude.
- Euclidean distance measures straight-line distance.

The appropriate metric and normalization depend on the embedding model and index configuration.

## Retrieval workflow
1. Choose an embedding model suited to the content and language.
2. Embed documents and store vectors with IDs and metadata.
3. Embed a user query using the compatible model.
4. Search for nearest vectors.
5. Apply metadata filters or reranking if useful.
6. Return source text and provenance to the application.

## Practical cautions
- Query and document embeddings should be compatible.
- Chunking affects retrieval quality.
- Similarity is not proof that a passage answers the question.
- Keep source IDs and access controls with indexed content.
- Rebuild or version indexes when embedding models change.

## Exercise
Embed a small collection of course notes and compare the top five results for keyword search versus vector search. Record cases where each method succeeds or fails.