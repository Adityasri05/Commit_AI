# Access Control and Tenant Isolation in RAG

A Retrieval-Augmented Generation system can expose private information if it retrieves documents that the current user is not allowed to see. A prompt instruction such as "do not reveal confidential content" is not an access-control mechanism. Authorization must be enforced by the application and retrieval layer.

## Store authorization metadata

At ingestion time, associate every indexed chunk with stable metadata such as:

- Document and chunk identifiers.
- Tenant or workspace identifier.
- Allowed user or group scopes.
- Classification level.
- Source-system permissions or ACL version.
- Document version and deletion status.

Do not trust access labels supplied by the user in the query. Build the effective identity and group memberships from authenticated server-side context.

## Filter before retrieval

The retrieval service should apply the user's authorized scope as part of the search operation, before candidate passages are returned to the generation layer.

A safe high-level flow is:

1. Authenticate the request.
2. Resolve the user's tenant and current permissions using trusted identity data.
3. Construct a server-side authorization filter.
4. Search only within the authorized document scope.
5. Re-check authorization for each returned source against the authoritative policy when required.
6. Pass only approved passages to the model.
7. Apply the same authorization rules to citations, cached results, logs, and follow-up questions.

Metadata filtering in a vector database is useful, but it is not a substitute for application-level authorization. Verify that the database filter semantics, index design, and query path actually enforce the required boundary.

## Tenant isolation choices

- **Separate indexes or collections:** can provide a stronger operational boundary, but increase management overhead.
- **Shared index with mandatory tenant filters:** can be efficient, but every query path must apply the filter correctly.
- **Hybrid approach:** separate high-sensitivity tenants while using scoped filters for lower-risk workloads.

Choose based on threat model, scale, compliance requirements, and the guarantees of the storage engine.

## Revocation and caching

Permissions change. A document may be removed, a user may leave a group, or a tenant may revoke access.

- Propagate ACL changes and deletions to the index promptly.
- Version authorization metadata or re-check it at read time.
- Include user/tenant scope and policy version in cache keys.
- Invalidate cached answers when source permissions change.
- Do not reuse one user's retrieved context for another user.

## Avoid side-channel leaks

Even if the model never sees a restricted passage, the application can leak information through result counts, error messages, document titles, citation IDs, timing differences, or debug logs. Return only information the caller is authorized to know.

## Security tests

Create at least two test tenants with overlapping and private documents. Verify:
- A user can retrieve an authorized document.
- A user cannot retrieve another tenant's document by changing prompt text.
- A citation cannot be opened without authorization.
- Revoked access stops working after the documented propagation window.
- Cached answers are not shared across users with different permissions.
- Tool errors and logs do not expose restricted document content.

## Practice

Build a small RAG prototype with tenant_id and allowed_groups metadata. Add integration tests that attempt cross-tenant retrieval, direct citation access, and cache reuse after an ACL change.

## References

- [Pinecone: Filter by metadata](https://docs.pinecone.io/guides/search/filter-by-metadata)
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

**Key takeaway:** Authorization must constrain retrieval before private text reaches the model. Enforce scope in trusted application code and test every path that can return or cache a source.