# Prompt Injection Risks in RAG Systems

Retrieved documents are data, not trusted instructions. A malicious or compromised document may contain text that attempts to override system behavior, reveal secrets, or trigger unsafe tool use.

## Threat examples
- A document says to ignore previous instructions.
- Retrieved content asks the model to expose hidden prompts or credentials.
- A page embeds instructions to call a tool or send data elsewhere.
- Poisoned content is crafted to influence answers across many queries.

## Defense in depth
1. Clearly separate trusted instructions from retrieved content.
2. Tell the model to treat retrieved passages as untrusted evidence, never as authority.
3. Restrict tools by allowlist and validate every tool argument server-side.
4. Keep secrets out of prompts and retrieval indexes.
5. Apply authorization filters before retrieval so users cannot access another user's documents.
6. Require confirmation for consequential external actions.
7. Log and test suspicious retrieval behavior.

## Testing ideas
Build a small adversarial corpus with instruction-like text, fake credentials, misleading citations, and cross-tenant references. Verify that the system answers from evidence without obeying embedded commands.

## Important limitation
Prompt wording alone cannot guarantee security. Enforce permissions and action boundaries in application code, and use least-privilege access for every connected tool.

**Key point:** RAG expands what a model can read; it must not expand what the model is authorized to do.