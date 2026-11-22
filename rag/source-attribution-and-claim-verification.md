# Source Attribution and Claim Verification in RAG

A RAG answer is more useful when readers can trace important statements to the documents that support them. A list of links at the end is not enough if it is unclear which source supports which claim.

## Preserve provenance during ingestion

For every chunk, retain metadata such as:
- Stable document ID and document version.
- Source URI or repository path.
- Title, author or owner where appropriate.
- Page, section, or paragraph identifier.
- Chunk ID and offsets into the original text.
- Ingestion timestamp and access-control scope.

Do not discard source metadata when embeddings are stored or when candidates are reranked.

## Attach evidence to claims

A useful answer-generation process is:

1. Retrieve passages and retain their identifiers.
2. Ask the model to draft concise claims with evidence IDs.
3. Check that every cited ID exists in the retrieved set.
4. Verify that the passage actually supports the associated claim.
5. Remove or qualify unsupported claims.
6. Abstain or request more information when evidence is insufficient.

A citation can be syntactically valid but semantically irrelevant. For high-impact use cases, use deterministic checks where possible and human review for consequential claims.

## Example output contract

A structured internal response might contain:

    {
      "claims": [
        {
          "text": "The service supports streamed responses.",
          "evidence_ids": ["doc-12-chunk-4"]
        }
      ],
      "insufficient_evidence": false
    }

The application should resolve each evidence ID to trusted source metadata and render the citation. Do not let the model invent URLs or page numbers.

## Citation quality checks

- **Existence:** Does the cited document and chunk exist?
- **Relevance:** Is the passage about the claim?
- **Entailment:** Does it support the claim, contradict it, or say something weaker?
- **Coverage:** Are important factual claims cited?
- **Version:** Is the source current enough for the question?
- **Access:** Is the user allowed to see the cited source?

For numeric claims, preserve units, dates, and table context. For policy or technical documentation, cite the correct version because behavior can change between releases.

## Common mistakes

- Adding citations after generation without checking support.
- Citing a whole document when only one section is relevant.
- Losing page or chunk offsets during indexing.
- Treating a model-generated quotation as a verified quotation.
- Showing a source the user is not authorized to access.
- Failing to distinguish conflicting sources or outdated versions.

## Practice

Build a small RAG answer schema that returns claims and evidence IDs. Add tests for an unknown citation ID, an irrelevant passage, conflicting sources, and a question with no supporting evidence.

## References

- [RAGAS documentation](https://docs.ragas.io/)
- [ALCE: Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627)

**Key takeaway:** Citations are evidence links, not decoration. Preserve provenance and verify claim-to-source support before displaying them.