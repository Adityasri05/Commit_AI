# Multi-Agent Coordination Patterns

A multi-agent system uses multiple model-driven workers or specialized agents to complete a task. The agents may have different prompts, tools, context, or responsibilities. Coordination determines who does what, how results are exchanged, and how the system detects completion or failure.

Using more agents does not automatically improve quality. Every handoff adds communication, orchestration, latency, and opportunities for disagreement.

## Common coordination patterns

### 1. Supervisor and specialists

A supervisor decomposes a request, assigns bounded subtasks to specialists, checks their results, and synthesizes the final response.

Example: a research supervisor delegates source discovery, technical analysis, and risk review to separate workers.

**Strength:** clear ownership and centralized quality control.

**Risk:** the supervisor can become a bottleneck or accept weak specialist output without verification.

### 2. Sequential handoff

One agent completes a stage and passes a structured result to the next agent.

Example: a document extraction agent produces normalized facts, a verification agent checks evidence, and a report agent formats the result.

**Strength:** natural for staged pipelines with explicit dependencies.

**Risk:** an early error can propagate through every later stage. Validate outputs at each boundary.

### 3. Parallel workers with aggregation

Independent workers receive separate tasks at the same time. An aggregator combines their results.

Example: research three independent vendors in parallel, then compare the findings.

**Strength:** reduces wall-clock time when tasks are truly independent.

**Risk:** duplicated work, inconsistent assumptions, rate limits, and conflicting claims. Parallelism does not remove the need for source verification.

### 4. Shared blackboard or shared state

Agents read and write to a common task state or workspace. Each agent contributes a result that other agents can inspect.

**Strength:** supports iterative collaboration and shared evidence.

**Risk:** concurrent writes can conflict, stale observations can be reused, and one agent may overwrite another's work. Use ownership, versioning, append-only events, or transactional updates.

### 5. Critic or review agent

A separate reviewer checks a draft against explicit criteria, such as completeness, source support, or policy constraints.

**Strength:** creates a deliberate verification stage.

**Risk:** a model reviewing another model can share the same blind spots. Critical claims still need deterministic checks or trusted external evidence.

## Define task contracts

Every handoff should specify:

- Task ID and parent task ID.
- Objective and expected output schema.
- Allowed tools and data scope.
- Input references and provenance.
- Deadline, token budget, and maximum tool calls.
- Completion criteria and error format.
- Whether the result is a draft, verified fact, or recommendation.

Do not pass an unbounded conversation transcript when a compact structured artifact is sufficient. Preserve the source references needed to audit the result.

## A simple orchestration sketch

```python
import asyncio
from dataclasses import dataclass

@dataclass
class TaskResult:
    task_id: str
    status: str
    summary: str
    evidence: list[str]

async def run_worker(worker, task, semaphore):
    async with semaphore:
        # worker must enforce its own tool allowlist and timeout.
        return await worker(task)

async def coordinate(tasks, workers):
    semaphore = asyncio.Semaphore(3)
    jobs = [
        run_worker(workers[task["role"]], task, semaphore)
        for task in tasks
    ]
    results = await asyncio.gather(*jobs, return_exceptions=True)

    successful = []
    failures = []
    for task, result in zip(tasks, results):
        if isinstance(result, Exception):
            failures.append({"task_id": task["id"], "error": type(result).__name__})
        else:
            successful.append(result)

    # The supervisor must validate evidence and completion criteria
    # before synthesizing a final answer.
    return successful, failures
```

This sketch limits concurrency and separates failures from successful results. Production code also needs cancellation, wall-clock deadlines, retries for safe operations, tracing, and explicit result validation. Do not blindly retry non-idempotent tool actions.

## Coordination safeguards

- Give each agent the minimum permissions required for its role.
- Keep authorization checks in application code, not in natural-language instructions.
- Set global budgets across the team, not just per-agent budgets.
- Use timeouts, cancellation, and a maximum handoff count.
- Make side-effecting operations idempotent or require approval.
- Treat worker output as untrusted data and validate it before downstream use.
- Preserve provenance when combining facts from multiple workers.
- Define what happens when a worker fails, disagrees, or returns incomplete evidence.

## When not to use multiple agents

A single model call, deterministic function, or ordinary workflow is often better when:

- The task is short and independent.
- There is no meaningful specialization.
- The output can be produced with a simple retrieval or database query.
- Latency or cost is tightly constrained.
- A regulated or high-impact process requires deterministic state transitions.

Start with the simplest architecture that meets the quality requirement. Add agents only when measured benefits justify the extra complexity.

## Evaluation

Measure end-to-end task success, factual support, handoff count, tool-call errors, latency, token usage, cost, and recovery from worker failures. Compare against a single-agent or deterministic baseline. Evaluate each role separately so a weak specialist does not hide behind a good final answer.

## Practice

Build a research workflow with a planner, two parallel source analysts, and a verifier. Require every analyst claim to include a source URL and quoted evidence span. Inject a worker timeout and a contradictory source, then test whether the supervisor reports uncertainty instead of silently merging the claims.

## References

- [AutoGen: Multi-agent conversation framework](https://microsoft.github.io/autogen/)
- [LangGraph: Multi-agent systems](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [A Survey on LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2402.01680)

**Key takeaway:** Multi-agent design is primarily a coordination and reliability problem. Define contracts, permissions, budgets, and verification before adding more workers.
