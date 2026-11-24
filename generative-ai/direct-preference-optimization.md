# Direct Preference Optimization (DPO) for LLM Alignment

Direct Preference Optimization (DPO) is a method for tuning a language model using preference comparisons. Instead of first training a separate reward model and then running a reinforcement-learning optimization loop such as PPO, DPO derives a classification-style objective from a preference objective under a particular policy parameterization.

It is often used after supervised fine-tuning (SFT), when examples contain a prompt, a preferred response, and a less-preferred response.

## Preference data

A typical record contains:

- **Prompt:** The context shown to the model.
- **Chosen response:** The response preferred by a human or a trusted preference process.
- **Rejected response:** A less-preferred response to the same prompt.

The quality of these comparisons matters. If annotators disagree, instructions are ambiguous, or the chosen response is preferred only for style rather than task correctness, the model can learn the wrong behavior.

## Intuition

DPO increases the relative likelihood of the chosen response compared with the rejected response, while using a reference policy to constrain how far the tuned policy moves.

A common form of the objective is:

\[
L_{\mathrm{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(
\log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)\right)\right]
\]

Here:
- x is the prompt.
- y_w is the chosen response.
- y_l is the rejected response.
- pi_theta is the trainable policy.
- pi_ref is a fixed reference policy, commonly the SFT model.
- beta controls the strength of the reference-policy constraint.

The exact loss implementation includes token-level log-probability calculations and masking. The equation is a conceptual summary, not a drop-in training script.

## Typical workflow

1. Start from a capable pretrained model.
2. Perform supervised fine-tuning on high-quality demonstrations.
3. Collect preference pairs for representative prompts.
4. Clean, deduplicate, and audit the preference dataset.
5. Keep a reference policy fixed.
6. Train the policy with DPO.
7. Evaluate helpfulness, correctness, safety, refusal behavior, and regressions.
8. Compare against the SFT baseline before deployment.

## DPO compared with RLHF using PPO

| Aspect | DPO | Reward-model + PPO pipeline |
|---|---|---|
| Preference signal | Direct chosen/rejected pairs | Trained reward model |
| Separate reward model | Not required in the standard formulation | Usually required |
| Optimization | Preference classification-style objective | Policy optimization with reward and constraints |
| Engineering | Often simpler to implement | More moving parts and tuning |
| Limitations | Sensitive to preference data and assumptions | Sensitive to reward-model errors and RL instability |

DPO is still preference optimization. It does not eliminate data quality problems, evaluation needs, or alignment trade-offs.

## Practical data checks

- Ensure chosen and rejected responses answer the same prompt.
- Remove exact duplicates and malformed examples.
- Avoid pairs where the preference depends on hidden metadata.
- Inspect whether one annotator or source dominates the dataset.
- Keep a held-out preference set.
- Track response length, because models can learn undesirable length preferences.
- Include safety and factuality examples without allowing preference labels to replace objective correctness checks.

## Common pitfalls

- Training on contradictory or low-quality comparisons.
- Using a reference model that does not match the intended SFT starting point.
- Miscomputing sequence log probabilities or failing to mask padding.
- Choosing beta by convention rather than validation.
- Evaluating only preference win rate while ignoring factual errors and safety regressions.
- Assuming DPO guarantees truthful or harmless behavior.

## Mini-project

Use a small open preference dataset and a modest instruction model with a parameter-efficient fine-tuning setup. Compare SFT and DPO outputs on a held-out prompt set. Report preference win rate, task accuracy, refusal behavior, response length, and qualitative failure cases.

## References

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023): https://arxiv.org/abs/2305.18290
- Hugging Face TRL DPO Trainer: https://huggingface.co/docs/trl/dpo_trainer
