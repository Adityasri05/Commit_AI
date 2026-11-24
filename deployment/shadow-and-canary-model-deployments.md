# Shadow and Canary Deployments for ML Models

A model that performs well offline can behave differently with real traffic. Shadow and canary deployments reduce the risk of introducing a new model by exposing it to production-like requests in controlled ways.

They answer different questions:

- **Shadow deployment:** How does the candidate behave on real requests if it does not serve the user-facing response?
- **Canary deployment:** What happens when a small portion of real traffic receives the candidate's response?

## Shadow deployment

The production model remains responsible for user-facing responses. A copy of incoming requests is sent to the candidate model, and its predictions are logged for comparison.

Useful measurements:
- Prediction disagreement with the current model.
- Candidate latency and error rate.
- Output validity and missing-feature rate.
- Performance on later-labeled outcomes.
- Resource utilization and cost.

Shadow traffic must not trigger duplicate side effects. For example, a candidate model should not send a second payment instruction, email, or irreversible action merely because it received a copied request.

## Canary deployment

A small fraction of eligible traffic is routed to the candidate model. The candidate's outputs are actually served to those users.

A gradual rollout might move from a small percentage to larger fractions after each evaluation gate. The exact percentages depend on traffic volume, risk, and rollback speed; they are not universal defaults.

Monitor:
- User-visible task metrics.
- Latency percentiles and error rates.
- Safety, fairness, and policy metrics.
- Business or domain outcomes.
- Differences across important user groups.
- Cost per request and infrastructure saturation.

## Comparison

| Dimension | Shadow | Canary |
|---|---|---|
| Candidate response served? | No | Yes, for routed traffic |
| User impact | Intended to be none | Real but limited |
| Main value | Observe behavior on live inputs | Measure real outcomes |
| Main risk | Duplicate side effects, privacy, extra load | User harm or degraded service |
| Strong use | Early compatibility and behavior check | Controlled production validation |

Shadow testing does not establish user acceptance or real outcome quality because users do not receive the candidate output. Canary testing does, but it must be designed with guardrails.

## Rollout state machine

A practical release can use these stages:

1. Offline evaluation and approval.
2. Shadow traffic with side effects disabled.
3. Canary traffic with a small eligible cohort.
4. Gradual traffic expansion.
5. Full rollout.
6. Rollback or pause if a guardrail fails.

Each stage should have an owner, an observation window, explicit thresholds, and a rollback mechanism.

## Example routing sketch

~~~python
def choose_model(request, rollout):
    if rollout.shadow_enabled:
        # Run asynchronously; do not return candidate output to the user.
        schedule_shadow_prediction(request, rollout.candidate_model)

    if rollout.candidate_fraction > 0:
        bucket = stable_hash(request.user_id) % 10_000
        if bucket < int(rollout.candidate_fraction * 10_000):
            return predict(rollout.candidate_model, request)

    return predict(rollout.current_model, request)
~~~

This is illustrative only. Production routing needs privacy controls, stable cohort assignment, authenticated model references, request timeouts, and protections against shadow calls triggering side effects. Do not use an untrusted user-supplied ID as a security boundary.

## Guardrails and rollback

Define thresholds before rollout. Examples include:
- Error rate exceeds a defined limit.
- p95 latency rises beyond the service budget.
- Invalid output rate increases.
- A critical subgroup metric falls below an approved floor.
- A safety or policy violation is detected.
- Cost per request exceeds the deployment budget.

Use a fast, tested rollback path. Keep the previous model artifact and compatible preprocessing code available. Do not automatically roll back solely because two models disagree; disagreement is a signal to investigate, not proof that the candidate is worse.

## Experiment design

- Use stable cohorts to reduce noisy switching between models.
- Avoid comparing different user populations without adjustment.
- Check sample size and duration before making decisions.
- Monitor delayed labels and seasonality.
- Avoid exposing high-risk users to a candidate without appropriate approval.
- Record model version, routing decision, feature version, and request timestamp.

## Mini-project

Build a small prediction API with a current and candidate model. First mirror traffic in shadow mode and compare outputs without serving the candidate. Then route a controlled test cohort to the candidate. Simulate a latency or error-rate breach and demonstrate rollback.

## References

- Google Cloud architecture guidance on ML deployment: https://cloud.google.com/architecture
- Kubernetes deployment concepts: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- AWS SageMaker shadow testing documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html
