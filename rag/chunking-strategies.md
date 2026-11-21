# Document Chunking Strategies for RAG

Retrieval-Augmented Generation (RAG) usually indexes smaller passages rather than entire documents. **Chunking** decides how source material is divided before embedding and retrieval. A poor chunk can separate a fact from its context; an oversized chunk can dilute the relevant passage and consume the model's context window.

There is no universally best chunk size. Choose a strategy based on document structure, query types, embedding model, and generation budget.

## Common strategies

### Fixed-size chunks

Split text by a target number of tokens, optionally with overlap. This is easy to implement and provides predictable sizes, but it can cut across headings, sentences, tables, or code blocks.

### Recursive or structure-aware splitting

Try natural boundaries in order—for example, document sections, paragraphs, sentences, and finally smaller token windows. Preserve headings with the content they describe. For code, split by functions or classes where possible.

### Semantic chunking

Use changes in topic or semantic similarity to identify boundaries. It may preserve coherent ideas better, but adds complexity and can create inconsistent chunk sizes.

### Parent-child retrieval

Index small child passages for precise matching, but retain a link to a larger parent section. Retrieve the child and provide the parent or a controlled surrounding context to the generator.

## Overlap: useful but not free

A small overlap can preserve context when a relevant statement falls near a boundary. Too much overlap creates duplicate results, increases index size, and may cause the same evidence to occupy several top-k positions.

Store stable document IDs, chunk IDs, parent IDs, source locations, and version metadata. Deduplicate near-identical chunks before building the final prompt.

## Practical workflow

1. Inspect representative source files: prose, headings, tables, code, PDFs, and scanned text may need different handling.
2. Extract text while preserving page, section, and access-control metadata.
3. Split on meaningful boundaries before falling back to token limits.
4. Record chunk length and the source span for every chunk.
5. Embed and index the chunks using a versioned embedding model.
6. Evaluate retrieval on a labeled query set.
7. Adjust chunking and retrieval together, then rerun the same evaluation.

## What to measure

- Recall@k: does a relevant passage appear among the first k results?
- Precision@k: how many retrieved passages are relevant?
- Mean reciprocal rank: how early is the first relevant passage?
- Context duplication: how much retrieved text repeats?
- Answer faithfulness and citation support: can the answer be traced to the retrieved source?
- Latency, index size, and token cost.

A larger chunk may improve context completeness but reduce retrieval precision. A smaller chunk may improve matching but omit definitions or qualifications. Evaluate both retrieval and final answers.

## Common mistakes

- Choosing a chunk size only because a tutorial used it.
- Splitting tables or code without preserving headers and structure.
- Dropping source metadata needed for citations or permissions.
- Allowing overlap to fill the prompt with repeated content.
- Changing chunking without rebuilding or versioning the index.
- Evaluating on a handful of hand-picked queries only.

## Practice

Create three indexes over the same small collection: fixed token windows, recursive paragraph-aware chunks, and parent-child chunks. Use at least 20 labeled questions. Compare Recall@5, duplicate context, citation quality, latency, and total tokens sent to the generator.

## References

- [LlamaIndex: Node parsers and text splitters](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [LangChain: Text splitters](https://docs.langchain.com/oss/python/integrations/splitters)

**Key takeaway:** Chunking is an empirical retrieval design choice. Preserve document structure and provenance, then compare strategies on representative questions.