# Approximate Nearest-Neighbor Indexing for Retrieval

Vector retrieval compares an embedded query with stored document vectors. Exact nearest-neighbor search checks every candidate and can become expensive as a collection grows. Approximate Nearest-Neighbor (ANN) indexes reduce query cost by exploring only a promising subset of the vector space.

ANN is useful when low latency and large-scale retrieval matter, but it trades exactness for speed and resource efficiency.

## Exact search versus ANN

With exact search, the system ranks all vectors according to a distance or similarity function. This provides a reference result but can require work proportional to the number of vectors for each query.

An ANN index uses a structure that narrows the search:
- a proximity graph;
- a coarse partition into clusters;
- compressed vector codes;
- or a combination of these techniques.

Measure ANN quality against exact search on a representative sample. Do not assume that a fast result is relevant.

## Common index families

### HNSW

Hierarchical Navigable Small World indexes build a multi-layer proximity graph. Search starts at a sparse upper layer and moves toward closer neighbors in denser layers.

Important controls include:
- construction effort, which affects index quality and build time;
- graph connectivity, which affects memory and navigation;
- search effort, which affects latency and recall.

Higher search effort often improves recall but increases query cost.

### IVF

Inverted File indexes cluster vectors into coarse partitions. At query time, the system searches only a selected number of nearby partitions.

The number of clusters and the number of partitions probed trade recall for speed. Too few probes can miss relevant vectors in neighboring clusters.

### Product Quantization

Product Quantization divides a vector into subvectors and encodes each with compact codebooks. This reduces memory and can speed distance calculations, at the cost of approximate distances.

IVF and product quantization are often combined for large collections.

## Similarity and normalization

Common comparison functions include:
- Euclidean distance;
- inner product;
- cosine similarity.

Cosine similarity is the dot product of L2-normalized vectors. If a system uses inner product but vectors are not normalized, vector magnitude affects ranking. The index metric, embedding model, and query-time preprocessing must agree.

## Evaluate the index

Build an exact-search baseline and compare ANN results on the same query set.

Useful measurements:
- **Recall@k:** fraction of exact top-k neighbors recovered by ANN.
- **Latency:** median and p95 query time under realistic concurrency.
- **Memory:** index and metadata footprint.
- **Build cost:** time and resources required to construct or update the index.
- **Update behavior:** insert, delete, and rebuild cost.
- **Filtered recall:** quality when metadata or authorization filters are applied.

Recall is a retrieval-system metric, not proof that the retrieved passages answer the question. Evaluate end-to-end RAG quality separately.

## Practical tuning loop

1. Select a representative query set and corpus snapshot.
2. Compute exact neighbors for an evaluation subset.
3. Build candidate indexes with documented parameters.
4. Measure recall, p95 latency, memory, and build time.
5. Repeat under realistic filters and concurrency.
6. Choose an operating point that meets quality and service requirements.
7. Version the index with the embedding model and source corpus.
8. Re-evaluate after inserts, deletes, or embedding changes.

## Common pitfalls

- Comparing ANN recall against a different distance metric than production uses.
- Tuning on a tiny or unrepresentative query set.
- Ignoring metadata-filter behavior.
- Assuming index parameters transfer unchanged across embedding models.
- Forgetting that deleted or permission-revoked documents must stop appearing in results.
- Optimizing latency while allowing recall to fall below the application's needs.

## Practice

Create a few thousand vectors from a public text collection. Compare exact search, HNSW, and IVF on the same queries. Plot recall@10 against p95 latency and record index memory. Repeat after adding a metadata filter.

## References

- [FAISS documentation](https://faiss.ai/)
- [pgvector: Indexing and HNSW/IVFFlat](https://github.com/pgvector/pgvector)
- [hnswlib](https://github.com/nmslib/hnswlib)

**Key takeaway:** ANN indexing is a measurable quality-latency-memory trade-off. Keep an exact baseline, match the production metric, and evaluate filtering and index lifecycle as well as raw search speed.