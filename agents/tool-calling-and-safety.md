# LLM Tool Calling: From Model Intent to Safe Actions

Tool calling lets an LLM request structured operations from an application, such as searching a database or performing a calculation. The model proposes a call; the surrounding application validates and executes it.

## Typical cycle

1. The application sends a user request and available tool schemas to the model.
2. The model returns a normal response or a structured tool-call request.
3. The application validates the tool name and arguments.
4. The application checks authorization and executes the operation.
5. The result is returned to the model as data.
6. The model produces a user-facing response.

A tool call is not proof that an action succeeded. The application must report the actual execution result.

## Safety controls

- Allowlist tool names and validate arguments against schemas.
- Apply least-privilege credentials and per-user authorization.
- Require explicit confirmation for consequential or irreversible actions.
- Set timeouts, rate limits, and resource budgets.
- Treat retrieved webpages, documents, and tool outputs as untrusted data, not instructions.
- Log enough for auditing while minimizing sensitive data.
- Make actions idempotent where possible and handle retries safely.

## Example schema concept

A calculator tool might accept exactly two numeric arguments, `a` and `b`, and return their sum. It should reject unexpected fields, strings where numbers are required, and values outside defined bounds. A payment or account-deletion tool needs stronger authorization and confirmation than a calculator.

## Exercise

Design a read-only course-search tool and a write-capable course-enrollment tool. Specify each schema, permissions, validation rules, and which action should require user confirmation.

**Takeaway:** The application—not the language model—must enforce permissions, validation, and action safety.