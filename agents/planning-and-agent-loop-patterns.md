# Planning Patterns for LLM Agents

An LLM agent combines a model with tools, state, and a control loop. The loop determines how a task is broken down, when tools are called, how results are checked, and when execution stops.

A capable model does not automatically make a reliable agent. The application should choose a control pattern that matches the task's risk, complexity, and need for determinism.

## Three useful patterns

### 1. ReAct-style loop

The model alternates between reasoning-oriented decisions and tool actions, then observes tool results before deciding what to do next.

**Useful when:** the next action depends on information discovered during execution, such as searching a knowledge base or debugging an unfamiliar issue.

**Risks:** loops can wander, repeat calls, or spend too many tokens. Keep a step budget, tool allowlist, and explicit stopping conditions.

### 2. Plan-and-execute

A planning step creates a sequence of subtasks. An executor performs them, and a replanner may revise the plan when observations show that assumptions were wrong.

**Useful when:** a task has several stages that can be inspected independently, such as researching sources, extracting facts, and preparing a report.

**Risks:** an early plan can become stale. Treat it as a proposal, validate prerequisites, and replan when evidence changes.

### 3. State-machine or workflow agent

The application defines explicit states and allowed transitions. The LLM may interpret input or choose among permitted transitions, while code controls the overall workflow.

**Useful when:** the process has compliance requirements, approvals, or irreversible actions, such as processing a refund or onboarding a user.

**Strength:** transitions and failure paths are easier to test and audit. It is less flexible than an open-ended loop, but often more dependable.

## A safe control loop

A robust agent runner should make each iteration explicit:

1. Check the user and task authorization.
2. Read the current task state and remaining budget.
3. Ask the model for a permitted next action or final response.
4. Validate the tool name and arguments against an allowlist and schema.
5. Apply authorization checks in application code.
6. Execute with timeouts and resource limits.
7. Store the result as untrusted observation data.
8. Update state and check stop conditions.
9. Return a result that reflects what actually happened.

Use a maximum number of steps, wall-clock timeout, token budget, and tool-call budget. Distinguish a completed task from a partial result or a tool failure.

## Design choices

- Keep deterministic business rules in code rather than asking the model to enforce them.
- Use read-only tools by default; require explicit approval for consequential writes.
- Make retryable operations idempotent or attach idempotency keys.
- Store compact state with provenance instead of repeatedly passing an unbounded transcript.
- Treat retrieved content and tool output as data, never as privileged instructions.
- Record tool inputs and outcomes while redacting secrets and unnecessary personal data.
- Test success, malformed arguments, empty results, timeouts, retries, and adversarial content.

## Practice

Build a research agent that must find three sources and summarize them. Compare a free-form ReAct loop with a state machine containing SEARCH, VALIDATE_SOURCES, SUMMARIZE, and DONE states. Add a maximum of six tool calls and test what happens when search returns no usable sources.

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [LangGraph: Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

**Key takeaway:** Use open-ended loops for adaptive exploration and explicit workflows for controlled processes. In both cases, the application owns permissions, budgets, and stopping conditions.