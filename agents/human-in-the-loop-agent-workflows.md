# Human-in-the-Loop Workflows for AI Agents

A human-in-the-loop (HITL) agent pauses at a defined decision boundary so a person can review, approve, reject, or edit a proposed action. This is important when an agent can affect money, access, customer communication, production systems, or other consequential resources.

Human review should be a designed workflow state, not an informal message asking someone to “check the result.”

## Where to place approval gates

Use approval gates when:
- an action is irreversible or difficult to undo;
- a tool can modify external records or permissions;
- the agent is uncertain or lacks required evidence;
- a policy requires a qualified reviewer;
- a high-impact decision needs an appeal or human judgment.

Read-only research usually needs a different level of oversight than sending a message or deleting a record.

## Model the workflow as explicit states

A simple workflow might use:

    CREATED
      -> RUNNING
      -> WAITING_FOR_REVIEW
      -> APPROVED or REJECTED
      -> EXECUTING
      -> COMPLETED or FAILED

The state machine should record the current task, proposed action, relevant evidence, reviewer identity, decision, and timestamps.

Do not allow a model to move itself from WAITING_FOR_REVIEW to APPROVED. The approval event must come from an authenticated reviewer through a controlled interface.

## Pause, persist, and resume

A long-running agent may need to wait minutes or days for review. Persist enough state to resume safely:

- task and workflow version;
- user request and validated context;
- proposed tool name and arguments;
- evidence shown to the reviewer;
- approval status and reviewer identity;
- idempotency key and execution status;
- retry count, deadline, and cancellation state.

Avoid storing unnecessary secrets or sensitive prompt content in the checkpoint. Apply retention and access controls to workflow state.

## Safe execution after approval

Approval is not a blank cheque. Before executing:

1. Confirm the approval is authentic and still valid.
2. Revalidate the proposed arguments and permissions.
3. Check that the target resource has not changed unexpectedly.
4. Use an idempotency key so retries do not duplicate the action.
5. Execute with a narrowly scoped credential.
6. Record the result and any external action identifier.

If the action fails halfway, define whether it can be retried, compensated, or must be escalated.

## Example: a reviewed email workflow

~~~python
def prepare_email(task):
    draft = agent.create_email(task)
    validate_recipients(draft)
    validate_attachments(draft)
    checkpoint = save_pending_review(
        draft=draft,
        task_id=task.id,
        status="WAITING_FOR_REVIEW",
    )
    return checkpoint.id

def resume_after_review(review_id, reviewer):
    review = load_review(review_id)
    require_authorized_reviewer(reviewer, review)
    require_approved(review)
    validate_recipients(review.draft)
    send_email(
        review.draft,
        idempotency_key=review.idempotency_key,
    )
    mark_completed(review.id)
~~~

This is illustrative pseudocode, not a drop-in library API. The approval record should be immutable or auditable, and the send operation should be protected against duplicate retries.

## Reliability patterns

- **Durable execution:** Save progress so a worker restart does not lose the task.
- **Idempotency:** Repeating a request should not repeat a side effect.
- **Timeouts:** Expire stale approvals and ask for a fresh review when necessary.
- **Cancellation:** Let authorized users cancel before execution.
- **Compensation:** Define a corrective action for operations that cannot be rolled back atomically.
- **Escalation:** Route ambiguous or high-risk cases to a qualified reviewer.
- **Auditability:** Preserve who approved what, which version was executed, and the final outcome.

An approval can become stale if the underlying resource changes. Re-check relevant state at execution time and return to review if the action's meaning has changed.

## Human factors

Reviewers need concise evidence, the exact proposed action, likely consequences, and meaningful alternatives. Avoid overwhelming them with raw model traces. Provide enough information to detect errors, but do not imply that human approval guarantees correctness.

Track reviewer workload, turnaround time, override rate, and errors discovered during review. These metrics can reveal confusing interfaces or poor agent behavior.

## Practice

Build a small agent that drafts a support-ticket response but cannot publish it without approval. Add rejection, edit-and-resubmit, timeout, cancellation, and duplicate-retry cases. Verify that no external side effect occurs before an authorized approval event.

## References

- [LangGraph: Interrupts and human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Temporal: Durable Execution](https://docs.temporal.io/)
- [OWASP: Human oversight and AI security guidance](https://genai.owasp.org/)

**Key takeaway:** Human-in-the-loop design is a stateful control mechanism. Safe workflows persist context, authenticate reviewers, revalidate actions, and make retries and side effects auditable.