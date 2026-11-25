# Evaluating and Tracing LLM Agents

An agent can produce a plausible final answer while making unnecessary tool calls, using the wrong tool, or taking an unsafe action along the way. Agent evaluation therefore needs to inspect both the **outcome** and the **trajectory** that produced it.

## What is a trace?

A trace represents one end-to-end task. It contains spans for important steps such as model calls, tool invocations, retrieval, validation, and human approval. Each span should record timing, status, and safe metadata that helps reconstruct the workflow.

A trace can reveal:
- repeated calls with the same arguments;
- a tool selected for the wrong task;
- malformed arguments or failed authorization;
- retries that hide an unreliable dependency;
- excessive model calls, latency, or token usage;
- an agent that reports success even though a required action failed.

## Build an agent evaluation set

Each test case should include:
- user request and relevant context;
- allowed tools and expected permission boundaries;
- task-level success criteria;
- required evidence or output schema;
- known edge cases and prohibited actions.

Include ordinary tasks, ambiguous requests, empty tool results, timeouts, malformed tool responses, adversarial content, and requests that should be refused or escalated.

## Useful metrics

| Dimension | Example measurement |
|---|---|
| Task outcome | Exact success rate or rubric score |
| Tool selection | Correct tool choice per decision |
| Argument quality | Schema validity and semantic correctness |
| Efficiency | Steps, model calls, and tool calls per task |
| Reliability | Timeout, retry, and tool failure rates |
| Operations | End-to-end latency and cost per successful task |
| Safety | Unauthorized action attempts and policy violations |
| Recovery | Correct behavior after empty or failed tool results |

Do not optimize step count alone: a short trajectory can be wrong, while a longer one may be justified by verification. Report metrics by task category and risk level.

## A practical test harness

~~~python
def evaluate_case(agent, case):
    trace = agent.run(
        case["request"],
        allowed_tools=case["allowed_tools"],
        max_steps=case["max_steps"],
    )

    return {
        "task_passed": case["success_check"](trace.final_output),
        "tool_calls": len(trace.tool_calls),
        "unauthorized_calls": sum(
            not call.authorized for call in trace.tool_calls
        ),
        "steps": len(trace.steps),
        "trace_id": trace.id,
    }
~~~

This is an illustrative interface, not a library API. In a real harness, ensure success checks are deterministic where possible and never rely only on the agent's own self-assessment.

## Evaluation methods

- Use exact checks for schemas, required fields, permissions, and deterministic outcomes.
- Use human review for nuanced quality and high-impact cases.
- Use an LLM judge only with a written rubric and examples; compare judge scores against human labels.
- Preserve failed traces as regression cases.
- Re-run the same versioned dataset after model, prompt, tool, or orchestration changes.

## Privacy and observability

Traces can contain prompts, retrieved documents, tool arguments, and personal data. Redact secrets, minimize captured content, set retention limits, and restrict access. Prefer stable identifiers and structured metadata over storing entire sensitive payloads. Make sure observability does not bypass the same authorization rules as the application.

## Practice

Create ten tasks for a research agent. Include a case where search returns no results and one where a page contains an instruction to call an unauthorized tool. Score task success, tool selection, unauthorized attempts, latency, and cost. Inspect the traces for the reason behind each failure.

## References

- [OpenTelemetry documentation](https://opentelemetry.io/docs/)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [LangSmith: Evaluation](https://docs.smith.langchain.com/evaluation)

**Key takeaway:** Reliable agent evaluation measures the final result and the sequence of decisions, tool calls, failures, and safeguards that led to it.