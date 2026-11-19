# A Practical Fairness Audit Starter

A fairness audit asks whether an AI system creates unjustified differences in outcomes for people or groups. No single metric proves a system is fair; the right analysis depends on the decision, population, harms, and legal context.

## Define the decision

Document the intended purpose, affected people, consequential outcomes, relevant groups and intersections, label-collection process, and ways to challenge a decision. Collect sensitive group attributes only when lawful, justified, and appropriately protected.

## Compare outcomes carefully

For a binary decision, useful diagnostics can include:
- Selection or positive prediction rates by group.
- False-positive and false-negative rates by group.
- Precision, recall, and calibration by group.
- Error rates across intersections and relevant time periods.

Metrics can conflict. For example, when base rates differ, calibration and equalized error rates may not both be achievable in the same way. Report definitions, uncertainty, sample sizes, and context rather than declaring a system fair from one parity statistic.

## Audit workflow

1. Establish a baseline before deployment.
2. Validate that group and outcome labels are reliable and appropriately collected.
3. Evaluate overall and subgroup performance.
4. Investigate data coverage, measurement error, proxy features, and workflow effects.
5. Consult affected stakeholders and domain experts.
6. Document mitigations, residual risks, owners, and review dates.
7. Monitor after launch and provide an appeal path.

## Exercise

Imagine a scholarship ranking model. Which groups might need outcome analysis? What harms could arise from false negatives? Why might historical scholarship awards be a biased target label?

**Takeaway:** Fairness is a socio-technical property of the full decision process, not a score attached to a model alone.