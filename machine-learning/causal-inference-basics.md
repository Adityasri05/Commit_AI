# Causal Inference Basics for ML Practitioners

Prediction asks, "What outcome is likely given these features?" Causal inference asks, "What would change if we intervened?" A model can predict accurately without identifying the effect of an action.

For example, students who attend optional tutoring may have different exam outcomes from students who do not. A simple comparison does not prove tutoring caused the difference: motivation, prior grades, or available time may affect both attendance and results.

## Potential outcomes

Let:

- Y(1) be the outcome a unit would have under treatment.
- Y(0) be the outcome the same unit would have without treatment.

The individual treatment effect is Y(1) - Y(0), but only one potential outcome is observed for each unit. This is the fundamental missing-data problem of causal inference.

A common population target is the average treatment effect:

    ATE = E[Y(1) - Y(0)]

Estimating it requires assumptions and a design—not just a predictive model.

## Confounding and causal graphs

A confounder is a variable that influences both treatment assignment and outcome. If it is not handled appropriately, the observed association may mix the treatment effect with the confounder's effect.

Directed acyclic graphs (DAGs) help make assumptions explicit:
- Arrows represent hypothesized causal direction.
- A backdoor path can create a non-causal association between treatment and outcome.
- Adjusting for suitable pre-treatment confounders can block backdoor paths.
- Conditioning on a collider can create bias, even when the collider was not a confounder.

Do not automatically control for every available column. Variables measured after treatment may be mediators or colliders.

## Common identification strategies

- **Randomized controlled trials:** random assignment helps balance measured and unmeasured pre-treatment factors in expectation.
- **Regression adjustment:** controls for measured confounders when the model and assumptions are appropriate.
- **Matching / weighting:** constructs comparable groups using observed covariates; it cannot fix unmeasured confounding by itself.
- **Difference-in-differences:** compares changes over time between treated and comparison groups, relying on assumptions such as parallel trends.
- **Instrumental variables:** use a variable that changes treatment but affects the outcome only through treatment, under additional assumptions.

## Key assumptions to state

For a basic observational ATE analysis, discuss:
1. **Consistency:** the treatment and outcome are well-defined.
2. **Conditional exchangeability:** after conditioning on measured covariates, treatment assignment is independent of potential outcomes.
3. **Positivity:** every relevant covariate profile has a non-zero chance of receiving each treatment.

These assumptions are substantive. Statistical significance does not prove them.

## Practical workflow

1. Define the intervention, population, outcome, and time horizon.
2. Draw a causal DAG before selecting adjustment variables.
3. Choose an estimand such as ATE or average treatment effect on the treated.
4. Select a design and state its assumptions.
5. Check overlap, missingness, and sensitivity to unmeasured confounding.
6. Report uncertainty and limitations, not only a point estimate.

## Practice

Estimate the effect of a study-support program on exam scores. Draw a DAG containing prior achievement, motivation, program participation, and exam score. Explain which variables you would adjust for and why. Then discuss how random assignment would change the design.

## References

- [The Book of Why — Judea Pearl and Dana Mackenzie](https://www.penguinrandomhouse.com/books/550655/the-book-of-why-by-judea-pearl-and-dana-mackenzie/)
- [Causal Inference: What If — Hernán and Robins](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

**Key takeaway:** Causal claims require a clear intervention, an estimand, a design, and defensible assumptions. Predictive accuracy alone is not causal evidence.