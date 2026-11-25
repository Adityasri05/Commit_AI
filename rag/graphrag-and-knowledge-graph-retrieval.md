# GraphRAG and Knowledge-Graph Retrieval

Traditional RAG retrieves passages that resemble a question. GraphRAG adds an explicit graph of entities and relationships so a system can answer questions that require connecting facts across documents or summarizing a collection at a higher level.

## Build the knowledge graph

A common indexing workflow is:

1. Parse and chunk trusted source documents.
2. Extract entities such as people, organizations, products, and concepts.
3. Extract typed relationships between entities.
4. Resolve aliases and duplicate entity mentions.
5. Attach provenance: source document, chunk, and offsets where possible.
6. Build graph indexes and optionally cluster related entities into communities.
7. Generate community summaries with links back to supporting evidence.

The graph is a derived artifact. Entity extraction and relationship extraction can be wrong, so preserve provenance and make extraction quality measurable.

## Retrieval patterns

- **Local search:** Start from entities in the question, then traverse nearby relationships and retrieve supporting passages. Useful for questions about a particular person, product, or connected set of facts.
- **Global search:** Use community-level summaries to answer broad questions about themes or patterns across a corpus. A common approach retrieves relevant community reports and synthesizes them.
- **Hybrid retrieval:** Combine graph traversal with vector or lexical passage retrieval. The graph supplies structure; source passages supply detailed evidence.

## Example graph

Consider three source-backed facts:

    (Ada) --worked_on--> (Analytical Engine)
    (Babbage) --designed--> (Analytical Engine)
    (Lovelace) --wrote_notes_about--> (Analytical Engine)

A question about people connected to the Analytical Engine can begin at that entity and traverse relationships. The answer should still retrieve the source passages that justify each claim; graph edges alone are not automatically trustworthy evidence.

## Evaluation

Compare GraphRAG with a strong vector/hybrid RAG baseline using the same corpus and question set. Measure:
- entity and relationship extraction precision/recall on an annotated sample;
- retrieval recall and relevance;
- answer correctness and groundedness;
- citation coverage and provenance quality;
- indexing cost, graph size, query latency, and update time.

Include questions that need one fact, multiple hops, and corpus-wide synthesis. GraphRAG may not improve simple lookup questions enough to justify its extra indexing and maintenance cost.

## Operational concerns

- **Entity resolution:** Aliases can merge distinct people or split one entity into duplicates.
- **Graph freshness:** Document updates and deletions must propagate to graph edges and summaries.
- **Provenance:** Store source IDs and evidence spans for extracted facts.
- **Access control:** Apply document permissions during graph traversal and passage retrieval, not only after generation.
- **Prompt injection:** Treat source text and extracted descriptions as untrusted data.
- **Hallucinated edges:** Do not convert uncertain extraction into a confident fact without validation.

## Practice

Build a tiny graph from five documents about a fictional organization. Extract entities and relations, retain source IDs, and answer one-hop and two-hop questions. Compare the answers with vector-only retrieval and inspect every unsupported edge.

## References

- [Microsoft GraphRAG documentation](https://microsoft.github.io/graphrag/)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)

**Key takeaway:** GraphRAG adds relationships and corpus-level structure to retrieval, but reliable answers still need source provenance, permission checks, and measured extraction quality.