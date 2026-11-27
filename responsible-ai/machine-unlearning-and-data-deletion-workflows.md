# Machine Unlearning and Data-Deletion Workflows

Machine unlearning studies how to remove the influence of selected training data from a trained model. It matters when records are withdrawn, data was used incorrectly, a user requests deletion, or a dataset must be corrected.

Deleting a row from a database does not automatically remove the effect that row may already have had on model parameters, embeddings, caches, or derived artifacts.

## Deletion and unlearning are different

- **Data deletion:** remove or restrict access to the source record and applicable derived copies.
- **Machine unlearning:** update a trained model so that the specified training data has no influence, or influence below a defined and tested approximation.
- **Full retraining:** train a replacement model from an approved dataset that excludes the data. This is often the clearest reference baseline, but can be expensive.

The required outcome depends on the system, threat model, data lifecycle, and applicable policy or law. A technical unlearning metric alone does not establish legal compliance.

## Why unlearning is difficult

Training combines information from many examples into shared parameters. A record may influence gradients, normalization statistics, learned representations, checkpoints, and downstream models.

There is usually no simple way to identify one row's exact contribution after many optimization steps. The difficulty grows with large datasets, distributed training, fine-tuning chains, and reused artifacts.

## Families of approaches

### 1. Exact retraining

Rebuild the model from a verified dataset that excludes the requested records. This provides a useful reference for evaluating approximate methods, although randomness and nondeterministic training can still produce different parameters.

### 2. SISA training

SISA stands for Sharded, Isolated, Sliced, and Aggregated training. Training data is partitioned so that a deletion request can trigger retraining of only affected shards or slices, then aggregation of the updated models.

The design can reduce deletion cost, but it changes the training architecture and may affect utility. It is most effective when planned before the original training run.

### 3. Approximate parameter updates

Some methods attempt to adjust model parameters using gradients, influence estimates, or targeted optimization. They can be cheaper than full retraining, but their guarantees depend on assumptions and model class.

A small change in loss or output on one example is not proof that all information about that example has been removed.

### 4. Fine-tuning and generative-model unlearning

For generative models, unlearning may target memorized sequences, concepts, or behaviors. The evaluation must test both the targeted information and nearby legitimate capabilities. Suppressing one prompt is not necessarily the same as removing the underlying influence.

## A practical deletion workflow

1. **Verify the request and scope.** Identify the record, subject, dataset version, and applicable retention policy.
2. **Trace lineage.** Find raw files, normalized tables, feature snapshots, embeddings, indexes, checkpoints, fine-tunes, and deployed artifacts that depend on the data.
3. **Stop new use.** Prevent the record from entering future training, retrieval, or inference while the request is processed.
4. **Remove source and derived copies.** Apply the system's deletion policy to storage, caches, vector indexes, and downstream datasets.
5. **Choose the model action.** Retrain, use a validated unlearning method, or document why a model artifact is outside the requested scope.
6. **Validate the result.** Test targeted forgetting, model utility, and relevant privacy risks.
7. **Update deployment and lineage.** Replace affected model versions, record the new artifact, and retire superseded versions according to retention policy.
8. **Keep a minimal audit record.** Record completion evidence without retaining the sensitive data that was meant to be deleted.

Backups and immutable audit systems need explicit retention and expiration rules. Do not claim immediate deletion from every backup unless the storage design actually supports it.

## How to evaluate unlearning

Use several complementary checks:

- **Reference comparison:** compare against a model retrained without the target records.
- **Targeted behavior tests:** check whether the model still reveals or reproduces the specified information.
- **Membership-inference tests:** measure whether an attacker can distinguish whether target examples were used for training.
- **Utility retention:** evaluate performance on legitimate, non-target data.
- **Collateral damage:** test neighboring concepts, classes, and important subgroups.
- **Artifact coverage:** verify that stale checkpoints, adapters, caches, and retrieval indexes are not still serving the old information.

No single empirical attack or test proves perfect forgetting. State the threat model, method, evaluation set, and limitations.

## Common mistakes

- Deleting source rows but leaving embeddings or checkpoints untouched.
- Treating prompt refusal as proof of parameter-level unlearning.
- Reporting only success on the targeted prompt and ignoring collateral damage.
- Forgetting downstream fine-tuned models and shared feature stores.
- Claiming formal guarantees for an approximate update.
- Keeping sensitive request contents in audit logs indefinitely.
- Failing to prevent deleted records from re-entering through a future data refresh.

## Practice

Design a lineage diagram for a model trained from raw files, a feature table, and a vector index. Simulate a deletion request for one source record. List every dependent artifact, define a retraining or unlearning decision, and create a verification checklist covering targeted behavior, membership inference, utility, and backup retention.

## References

- [Machine Unlearning: A Survey](https://arxiv.org/abs/2209.02299)
- [Making AI Forget You: Data Deletion in Machine Learning](https://arxiv.org/abs/2307.01191)
- [SISA Training: Machine Unlearning via Sharded, Isolated, Sliced, and Aggregated Training](https://arxiv.org/abs/1912.03817)

**Key takeaway:** Reliable unlearning is an end-to-end data-lineage and model-lifecycle process. Distinguish source deletion from removal of learned influence, and describe the strength and limits of the evidence honestly.
