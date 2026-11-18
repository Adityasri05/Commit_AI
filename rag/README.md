# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines information retrieval with text generation. Instead of relying only on information stored in model parameters, an application retrieves relevant external content and supplies it to the model as context.

## Typical RAG pipeline
1. **Ingest:** Load documents from trusted sources.
2. **Chunk:** Split documents into manageable passages.
3. **Embed:** Convert passages into vector representations.
4. **Index:** Store vectors and metadata in a searchable index.
5. **Retrieve:** Find passages relevant to a user's question.
6. **Augment:** Add selected passages to the model prompt.
7. **Generate:** Produce an answer grounded in the retrieved context.
8. **Evaluate:** Check relevance, faithfulness, and answer quality.

## Why chunking matters
Chunks that are too large may contain irrelevant material; chunks that are too small may lose context. Chunk size and overlap should be tested against the document type and retrieval task.

## Retrieval approaches
- Dense retrieval uses embedding similarity.
- Sparse retrieval uses lexical matching, such as BM25.
- Hybrid retrieval combines both and may improve recall.

## Evaluation dimensions
- Retrieval quality: Did the system find useful passages?
- Groundedness: Are answer claims supported by the context?
- Relevance: Does the answer address the question?
- Citation quality: Can users trace claims to source passages?

## Common failure modes
Poor document parsing, unsuitable chunking, stale indexes, irrelevant retrieval, and prompts that do not constrain unsupported claims can all reduce quality.

RAG improves access to external knowledge, but it does not guarantee correctness. Source quality and evaluation remain essential.