# Differential Privacy: A First Look for ML Engineers

Differential privacy (DP) is a mathematical framework for limiting how much an analysis output can reveal about one individual record. It provides a formal privacy guarantee under a defined threat model; it is not a general promise that a dataset or model is anonymous.

## Neighboring datasets

A DP mechanism compares two datasets that differ in one protected unit, such as one person's records. The definition depends on whether adjacency means adding or removing one record (unbounded DP) or replacing one record (bounded DP).

A randomized mechanism M is (epsilon, delta)-differentially private if, for every pair of neighboring datasets D and D-prime and every possible output set S:

    P[M(D) in S] <= exp(epsilon) * P[M(D-prime) in S] + delta

Epsilon controls the strength of the privacy bound: smaller values generally provide stronger privacy but may reduce utility. Delta allows a small probability of exceeding the pure-DP bound and should be chosen carefully for the population and threat model.

## How noise helps

A DP mechanism often bounds how much one record can change a query result, then adds calibrated random noise. For a simple count query with sensitivity 1, the Laplace mechanism adds Laplace noise scaled according to sensitivity divided by epsilon.

The noise is not arbitrary. The privacy guarantee depends on the mechanism, sensitivity calculation, adjacency definition, and parameters.

## Differential privacy in model training

DP-SGD is a common approach for training neural networks with a privacy guarantee:

1. Compute per-example gradients.
2. Clip each gradient to a chosen norm bound.
3. Aggregate the clipped gradients.
4. Add calibrated Gaussian noise to the aggregate.
5. Update model parameters.
6. Track privacy loss across training steps with a privacy accountant.

Clipping limits one example's influence; noise provides the formal privacy mechanism. More steps, sampling choices, and composition affect the total privacy budget.

## Important limitations

- DP does not correct biased data or guarantee fairness.
- It does not protect secrets that are already public or leaked elsewhere.
- A privacy guarantee applies to a specified unit and neighboring relation.
- Repeated releases consume privacy budget; composition must be accounted for.
- Very strong privacy may reduce model utility, especially with small datasets.
- Secure storage, access control, and deletion policies are still necessary.

## How to report a DP experiment

Record the adjacency definition, epsilon, delta, sampling method, number of steps, clipping norm, noise multiplier, accountant, dataset size, and utility metrics. Never report epsilon without enough configuration to interpret the guarantee.

## Practice

Use a reputable DP library to train a small classifier with and without DP-SGD. Compare test performance and inspect the reported privacy accountant. Do not implement a privacy mechanism from scratch for real personal data.

## References

- [Opacus: Differentially Private Training](https://opacus.ai/)
- [NIST: Privacy-enhancing technologies](https://www.nist.gov/privacy-framework)
- [Differential Privacy: A Primer for a Non-technical Audience](https://privacytools.seas.harvard.edu/differential-privacy)

**Key takeaway:** Differential privacy bounds the influence of an individual's data under explicit assumptions. Treat epsilon, delta, adjacency, and composition as essential parts of the result.