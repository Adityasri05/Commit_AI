# Survival Analysis: Modeling Time Until an Event

Survival analysis studies the time until an event occurs. It is useful for equipment failure, customer churn, hospital readmission, and time-to-completion problems.

Unlike ordinary regression, the event time may not be known for every observation.

## Censoring

A person or device is **right-censored** when observation ends before the event is seen. For example, a subscription is still active when a study ends. We know the customer survived at least until the last observation time, but not the eventual churn date.

Censoring is not the same as a negative label. Dropping censored rows or treating their last observed time as the event time creates bias.

A basic survival dataset contains:
- A duration: time from a defined starting point to event or last observation.
- An event indicator: 1 if the event was observed, 0 if censored.
- Features known at the prediction time.

The time origin and event definition must be consistent.

## Survival function and Kaplan-Meier estimate

The survival function S(t) is the probability that the event time T is greater than t:

    S(t) = P(T > t)

The Kaplan-Meier estimator updates survival at each event time. If n_i individuals are at risk immediately before time t_i and d_i events occur at that time:

    S_hat(t) = product over t_i <= t of (1 - d_i / n_i)

Censored observations leave the risk set after their censoring time. They do not count as events.

The curve is non-increasing. It can be compared across groups, but group differences alone do not establish a causal effect.

## Cox proportional hazards model

The Cox model relates features to a hazard rate:

    h(t | x) = h_0(t) * exp(beta_transpose_x)

The baseline hazard h_0(t) is left unspecified. A positive coefficient increases the modeled hazard, while a negative coefficient decreases it, holding other features constant.

The proportional-hazards assumption means hazard ratios are constant over time. Check this assumption rather than treating it as automatically true.

## Evaluation and leakage

- The concordance index measures how often comparable pairs are ordered correctly by predicted risk.
- Time-dependent AUC and Brier scores can assess prediction at specified time horizons.
- Censoring-aware methods are needed for valid evaluation.
- Use time-aware splits when deployment predicts future outcomes.
- Exclude features that are only recorded after the event or after the prediction time.

Standard methods often assume censoring is independent of event time, possibly conditional on observed covariates. If that assumption is implausible, seek specialist methods and domain review.

## Practice

Use a public survival dataset. Plot a Kaplan-Meier curve, fit a Cox model, inspect proportional-hazards diagnostics, and explain how a censored row contributes to the risk set.

## References

- [scikit-survival documentation](https://scikit-survival.readthedocs.io/)
- [lifelines documentation](https://lifelines.readthedocs.io/)

**Key takeaway:** Survival analysis uses both event and censoring information. Define time carefully and use methods that respect incomplete follow-up.