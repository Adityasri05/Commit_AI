# Learning to Rank: Models and Evaluation Metrics

Learning to rank (LTR) orders items for a query rather than predicting an isolated label. Search engines, recommendation feeds, and job listings are common examples: the goal is to place the most useful items near the top.

## Three training formulations

- **Pointwise:** Predict a relevance score or label for each query-item pair. It is simple to implement, but it does not directly model the relative order of two items.
- **Pairwise:** Learn which of two items should rank higher for the same query. Pairwise losses focus on ordering mistakes.
- **Listwise:** Optimize a whole ranked list or a metric-inspired objective. This can align training more closely with ranking quality, but may be harder to optimize.

A ranking dataset should retain its query or group identifier. Items belonging to one query form a list; do not accidentally treat every row as an unrelated example.

## Important ranking metrics

Let rel_i be the relevance grade of the item at rank i.

**Discounted cumulative gain (DCG)** rewards relevant items and discounts lower positions:

    DCG@k = sum from i=1 to k of (2^rel_i - 1) / log2(i + 1)

**Normalized DCG (NDCG)** divides DCG by the ideal DCG for that query. It lies between 0 and 1 when the ideal score is positive. NDCG is useful when relevance is graded, for example 0 = irrelevant, 1 = useful, 2 = highly useful.

**Mean reciprocal rank (MRR)** focuses on the first relevant result. For each query, take 1 divided by the rank of its first relevant item, then average across queries. It suits tasks where finding one good answer is enough.

**Mean average precision (MAP)** averages precision at the ranks where relevant items appear, then averages across queries. It is useful for binary relevance when retrieving several relevant items matters.

No metric is universally best. Choose one that reflects the product experience and define relevance labels consistently.

## Small example

Suppose a query has relevance labels [3, 0, 2] in the displayed order. The first and third results contribute to DCG; the second contributes zero. If the ideal order is [3, 2, 0], the ideal DCG is the normalizer. Swapping the first and third items changes the score substantially because high-relevance results should appear early.

## Evaluation protocol

1. Split by query or another meaningful group, not by arbitrary item rows.
2. Keep all items for a query in the same split.
3. Avoid placing near-duplicate queries or the same user session across train and test when that would leak information.
4. Report metrics at a product-relevant cutoff such as NDCG@5 or Recall@10.
5. Inspect individual queries and slices; an aggregate can hide poor performance for rare query types.
6. Compare against a simple baseline such as popularity, recency, or a hand-built score.

## Common pitfalls

- Using classification accuracy for a ranking problem.
- Mixing relevance labels with inconsistent meanings.
- Computing a ranking metric across unrelated queries.
- Ignoring position bias in click-derived labels: users are more likely to click items they see.
- Reporting only one metric without checking whether it matches the user goal.

## Practice

Create five search queries with five candidate documents each. Assign graded relevance labels, calculate NDCG@3 by hand for one query, and compare it with MRR. Explain why the two metrics may prefer different rankings.

## References

- [Microsoft: Learning to Rank](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)
- [scikit-learn ranking metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

**Key takeaway:** Ranking quality depends on the order within each query. Preserve query groups and select metrics that reflect where relevant results should appear.