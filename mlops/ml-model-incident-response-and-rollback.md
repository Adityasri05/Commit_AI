# ML Model Incident Response and Rollback

A production ML incident occurs when a model or its surrounding system causes unacceptable impact: incorrect predictions, data corruption, unsafe actions, privacy exposure, or service degradation. A response playbook helps teams limit harm while preserving enough evidence to diagnose the cause.

This guide focuses on what to do during an incident. Routine monitoring and release strategies are related disciplines, but they do not replace a clear response process.

## Prepare before an incident

For every production model, document:

- service owner and escalation contacts;
- model, feature, data, and code versions;
- intended use and known limitations;
- service and model-quality objectives;
- safe fallback behavior;
- last known-good model artifact;
- rollback and kill-switch procedure;
- data retention and evidence-handling rules;
- dependencies and downstream consumers.

Test the rollback path in a non-production environment. An undocumented rollback that has never been exercised is not a reliable control.

## Triage the impact

Classify the incident by user impact, scope, duration, and risk. Examples include:

- **Availability:** model endpoint is unavailable or timing out.
- **Quality:** prediction errors or severe segment-level degradation.
- **Data integrity:** wrong schema, unit, join, or feature values.
- **Safety:** harmful or prohibited outputs or actions.
- **Privacy/security:** sensitive data exposure, unauthorized access, or compromised credentials.

Do not wait for a perfect root-cause diagnosis before reducing active harm.

## Response sequence

1. **Acknowledge and assign an incident lead.** Establish one person coordinating technical work and communication.
2. **Contain the impact.** Pause a risky feature, disable an automated action, reduce traffic, or switch to a safe fallback.
3. **Preserve evidence.** Record timestamps, model and data versions, request IDs, relevant metrics, deployment events, and sanitized examples.
4. **Check recent changes.** Review model releases, feature pipelines, schema changes, dependencies, configuration, and upstream data sources.
5. **Choose mitigation.** Roll back, route to a known-good model, serve a conservative rule-based fallback, or temporarily abstain.
6. **Verify recovery.** Check service health, prediction distributions, critical segments, and user-impact indicators.
7. **Communicate status.** State known impact, mitigation, uncertainty, and next update time without exposing private data.
8. **Resolve and review.** Confirm the system is stable, document the timeline, and track corrective actions.

## Choosing a fallback

A fallback should be safer than the failing behavior for the affected use case. Options include:

- last known-good model, if its data and dependencies remain valid;
- a simpler baseline with known limitations;
- a deterministic business rule reviewed in advance;
- abstention or human review for high-risk decisions;
- disabling the affected feature.

A fallback can also fail. Check its compatibility with current feature schemas and traffic before routing users to it.

## Rollback decision checklist

Before switching artifacts, confirm:

- the target model and preprocessing versions match;
- required feature definitions and schemas are still available;
- the artifact is authentic and was previously validated;
- downstream API contracts remain compatible;
- rollback will not re-enable a known security or privacy issue;
- monitoring and audit logging remain active.

Do not roll back blindly if the incident is caused by compromised data or credentials that also affect the previous version.

## Root-cause analysis

Separate symptoms from causes. Ask whether the failure originated in:

- source data or upstream schema;
- feature computation or point-in-time logic;
- training data or evaluation design;
- model artifact or configuration;
- inference code, dependencies, or infrastructure;
- routing, access control, or user interaction;
- monitoring gaps or missing ownership.

A post-incident review should be blameless and evidence-based. Record what happened, impact, detection time, response time, contributing conditions, what worked, and actions with owners and due dates.

## Practice

Simulate a feature pipeline that changes a currency field from rupees to paise without updating the model. Write a response plan that detects the unit mismatch, disables unsafe decisions, preserves sanitized evidence, selects a compatible fallback, verifies recovery, and prevents recurrence.

## References

- [Google SRE: Managing Incidents](https://sre.google/workbook/incident-response/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

**Key takeaway:** A good incident process prioritizes harm reduction, evidence preservation, safe fallback, verified recovery, and concrete prevention work—not just restarting the model service.