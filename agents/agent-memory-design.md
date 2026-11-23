# Memory Design for LLM Agents

An LLM agent may need information across several steps or sessions. Memory provides a controlled way to retain useful state, but storing every conversation forever creates privacy, relevance, and security problems.

## Separate memory by purpose

- **Working or short-term state:** the current task, recent tool results, and intermediate decisions. It is usually scoped to one run or conversation.
- **Episodic memory:** a record of a specific past interaction or event, with time and source.
- **Semantic memory:** stable facts or preferences extracted from multiple observations, ideally with confidence and provenance.
- **Procedural memory:** reusable instructions or workflows that describe how to perform a task.

These are design categories, not mandatory database tables. A small agent may need only task state and a few explicitly saved preferences.

## A safe memory lifecycle

1. **Candidate extraction:** identify a possible fact, preference, or task outcome.
2. **Purpose check:** ask whether retaining it improves a defined future task.
3. **Permission and privacy check:** do not store secrets or sensitive personal details by default. Respect user consent and retention policy.
4. **Normalization:** store a compact, structured record rather than an unbounded transcript.
5. **Provenance:** keep source, timestamp, scope, and confidence so stale or uncertain facts can be handled.
6. **Retrieval:** filter by user, workspace, purpose, and permissions before ranking for relevance.
7. **Use:** provide only the minimum relevant memory to the model.
8. **Review and expiry:** support correction, deletion, expiration, and conflict resolution.

Example record fields:

    memory_id, subject_scope, fact, source_ref,
    created_at, last_confirmed_at, confidence,
    sensitivity, expires_at, policy_version

Avoid storing credentials, API keys, payment details, or private tokens in agent memory.

## Retrieval is a security boundary

A memory store is not automatically trustworthy. Stored text may contain stale claims or malicious instructions copied from a webpage or document.

- Treat retrieved memories as data, not higher-priority instructions.
- Keep user and tenant scopes enforced in application code.
- Do not let a prompt decide whether a user may access a memory.
- Prefer source-backed facts over unsupported summaries.
- Resolve contradictions by checking recency and provenance; ask the user when the conflict matters.
- Limit memory size and avoid injecting irrelevant history into every prompt.

## Example retrieval policy

For a request about a current project, retrieve records only from the authorized project scope, then rank by semantic relevance and recency. Exclude expired records and anything the user has asked not to retain. Pass the model a concise summary with source references, not the entire database result.

## How to evaluate memory

Create test cases for:
- Correct recall of a saved fact.
- Rejection of a fact from another user's scope.
- Avoidance of stale or superseded information.
- Correct handling of contradictory memories.
- Deletion and expiration behavior.
- Resistance to instructions embedded in retrieved text.
- Retrieval precision: how much injected context was actually useful.

Track both helpful recall and harmful over-retrieval. More memory is not automatically better.

## Practice

Build a small preference-memory prototype with explicit save, retrieve, update, and delete operations. Add tests proving that one user's records cannot be returned to another user and that expired records are not supplied to the model.

## References

- [LangGraph: Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

**Key takeaway:** Agent memory should be purpose-limited, scoped, source-aware, and deletable. Treat retrieval as both a relevance problem and an authorization boundary.