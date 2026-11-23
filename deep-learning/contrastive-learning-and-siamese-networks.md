# Contrastive Learning and Siamese Networks

Contrastive learning trains an encoder to place related examples near one another in an embedding space and unrelated examples farther apart. It is useful when labels are scarce or when the final task depends on similarity, such as image retrieval, duplicate detection, and semantic search.

## Positive and negative pairs

A training example often contains:
- An **anchor** input.
- A **positive** input that should have a similar representation.
- One or more **negative** inputs that should be less similar.

A Siamese network uses the same encoder weights for two input branches. Sharing weights ensures both inputs are mapped into the same representation space.

For normalized embeddings u and v, cosine similarity is:

    sim(u, v) = u · v

A contrastive objective rewards high similarity for positive pairs and lower similarity for negatives. The exact loss depends on how pairs and batches are constructed.

## InfoNCE intuition

For one anchor, suppose there is one positive and several negatives. A common objective increases the positive's similarity relative to all candidates:

    loss = -log( exp(sim(anchor, positive) / tau)
                 / sum_j exp(sim(anchor, candidate_j) / tau) )

Here tau is a temperature parameter. Lower temperatures sharpen the distribution and can make training more sensitive to similarity differences.

In-batch negatives reuse other examples in the batch as negative candidates. This is efficient, but it assumes those examples are truly unrelated to the anchor.

## PyTorch loss sketch

```python
import torch
import torch.nn.functional as F

# anchor_emb and positive_emb have shape [batch, dimension].
# Both are produced by the same encoder and normalized.
anchor_emb = F.normalize(anchor_emb, dim=-1)
positive_emb = F.normalize(positive_emb, dim=-1)

logits = anchor_emb @ positive_emb.T
logits = logits / 0.07
labels = torch.arange(logits.size(0), device=logits.device)

loss = F.cross_entropy(logits, labels)
```

This is a simplified in-batch contrastive objective. It assumes the matching positive for row i is column i. A real training pipeline must construct valid pairs, handle duplicate or related examples, and use a suitable batch sampler.

## Design decisions

- **Augmentations:** define what should remain invariant. Aggressive image crops may remove the object; text augmentations may change meaning.
- **Temperature:** tune it with the batch construction and objective.
- **Negative sampling:** hard negatives can help, but false negatives teach the model to separate examples that should be close.
- **Embedding normalization:** often makes cosine similarity and dot-product comparisons easier to interpret.
- **Evaluation:** use retrieval metrics such as Recall@k or task-specific verification accuracy, not training loss alone.

## Common failure modes

- Data leakage between training and evaluation pairs.
- False negatives caused by semantically related samples.
- Representation collapse or weak augmentations.
- Evaluating on random pairs when deployment queries come from a different distribution.

## Practice

Train an encoder on image augmentations or sentence pairs. Evaluate Recall@1 and Recall@5 on held-out identities or documents. Inspect nearest neighbors manually to find systematic errors.

## References

- [SimCLR: A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
- [Sentence Transformers: Loss Overview](https://www.sbert.net/docs/sentence_transformer/loss_overview.html)

**Key takeaway:** Contrastive learning shapes an embedding space through relationships between examples. Pair construction and evaluation data are as important as the loss function.