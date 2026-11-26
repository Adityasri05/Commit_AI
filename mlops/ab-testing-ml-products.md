# A/B Testing for Machine Learning Products

Offline model metrics estimate how a model behaves on a prepared dataset. An online experiment tests whether changing the model or product actually improves outcomes for real users under production conditions.

A/B testing is a form of randomized controlled experiment. It should be designed before traffic is split, not interpreted after looking at whichever metric moved.

## Define the experiment

Write down:

- **Hypothesis:** What user or business outcome should improve, and why?
- **Treatment:** The exact model, feature, prompt, or decision policy being changed.
- **Control:** The current production experience.
- **Randomization unit:** User, account, session, device, or another unit that avoids cross-group contamination.
- **Primary metric:** One outcome that represents the main hypothesis.
- **Guardrail metrics:** Measures that must not deteriorate, such as latency, error rate, safety incidents, or user complaints.
- **Eligibility and duration:** Which traffic is included and how long the test must run.

Log assignment and exposure separately. A user assigned to treatment may never actually receive the new model because of caching, fallback, or routing behavior.

## Randomization and interference

Randomization makes treatment groups comparable in expectation. The randomization unit matters:

- If one user can have several sessions, randomizing by session may expose the same user to both variants.
- If users affect one another, such as in a marketplace or social network, ordinary individual-level randomization may violate the no-interference assumption.
- For shared organizational accounts, account-level assignment may be more appropriate than user-level assignment.

Choose the unit that matches how the treatment can affect outcomes.

## Metrics and statistical power

Estimate the minimum effect worth detecting before launching. Use historical variance, baseline conversion, expected traffic, and a chosen error rate to estimate sample size and duration.

Report:
- effect size and confidence interval;
- sample sizes and exposure counts;
- the predeclared primary metric;
- guardrail results;
- relevant segment results, labeled as exploratory when not preplanned.

A small p-value does not tell whether an effect is practically important. A non-significant result does not prove the variants are equivalent.

## Essential experiment checks

### Sample ratio mismatch

Compare observed assignment counts with the intended traffic split. A large mismatch can indicate assignment bugs, logging loss, filtering differences, or exposure problems. Investigate before interpreting outcomes.

### Metric integrity

Check missing events, duplicate events, delayed conversion windows, bot traffic, and changes in metric definitions. Ensure the same measurement pipeline is used for both groups.

### Repeated peeking

Repeatedly checking a conventional fixed-horizon p-value and stopping as soon as it becomes significant inflates false-positive risk. Use a preplanned stopping rule, a valid sequential-testing method, or an explicitly designed Bayesian decision rule.

## A practical workflow

1. Validate both variants in offline tests and a safe pre-release environment.
2. Register the hypothesis, primary metric, guardrails, sample-size plan, and stopping rule.
3. Verify randomization and event logging with a small ramp.
4. Run the experiment without changing the analysis plan midstream.
5. Check sample ratio, exposure, missingness, latency, and safety continuously.
6. Analyze the predeclared outcomes after the stopping condition is met.
7. Roll out, iterate, or revert based on effect size, uncertainty, and guardrails.
8. Archive the code, model version, assignment logic, and analysis.

## ML-specific considerations

- Model updates can change scores for many users, so inspect segment-level outcomes and failure modes.
- Delayed labels may require a longer observation window than clicks or impressions.
- Feedback loops can make future training data depend on the treatment.
- A new model may increase engagement while worsening fairness, quality, or long-term retention.
- Keep model, feature, prompt, and routing versions attached to each exposure.

## Common pitfalls

- Using offline accuracy as the only success criterion.
- Selecting the primary metric after seeing the results.
- Ignoring sample ratio mismatch.
- Running many unplanned subgroup comparisons and reporting only the strongest.
- Stopping at the first favorable result without a valid sequential design.
- Launching a treatment that has not passed safety and reliability guardrails.

## Practice

Imagine a ranking model that is expected to increase useful clicks. Define the randomization unit, primary metric, two guardrails, exposure event, minimum worthwhile effect, and stopping rule. Explain how you would detect a sample ratio mismatch and handle delayed purchases.

## References

- [Microsoft: Online Controlled Experiments](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/)
- [Kohavi, Tang, and Xu: Trustworthy Online Controlled Experiments](https://experimentguide.com/)

**Key takeaway:** A/B testing measures product impact, not just model accuracy. Randomization, metric integrity, guardrails, and a predeclared analysis plan are essential for trustworthy conclusions.