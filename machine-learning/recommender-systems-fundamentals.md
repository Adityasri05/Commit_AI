# Recommender Systems: Collaborative Filtering and Matrix Factorization

A recommender system helps a user discover items from a large catalog: courses, videos, products, songs, or job listings. A typical production system separates **candidate generation** (find a manageable set of items) from **ranking** (order those candidates for the current context). Ranking is related to learning-to-rank, but recommendation also has to model user-item history, catalog coverage, and cold-start behavior.

## Main approaches

- **Popularity and recency:** Strong, cheap baselines. They can over-recommend already popular items.
- **Content-based filtering:** Match item metadata or text with a user's interests. It can recommend new items if their features are known.
- **Collaborative filtering:** Learn from patterns in user-item interactions. It can discover relationships that item metadata misses.
- **Hybrid systems:** Combine collaborative, content, contextual, and business signals.

## Explicit and implicit feedback

Explicit feedback includes ratings or likes with clear semantics. Implicit feedback includes clicks, views, purchases, or watch time. An interaction is not always positive preference: a click can be accidental, and no click can mean the user never saw the item.

For implicit data, models often distinguish observed positives from unobserved entries rather than treating every missing matrix cell as a definite negative.

## Matrix factorization

Let R be a user-item interaction matrix. Matrix factorization learns low-dimensional user vectors p_u and item vectors q_i:

    score(u, i) = p_u dot q_i + user_bias + item_bias

For explicit ratings, a common objective minimizes squared error on observed ratings plus regularization:

    sum_(u,i in observed) (r_ui - score(u,i))^2
      + lambda * (||p_u||^2 + ||q_i||^2)

For implicit feedback, weighted objectives can assign higher confidence to observed interactions. The confidence should reflect the event and exposure process, not an arbitrary assumption that every click means strong preference.

## Evaluation protocol

- Split interactions by time when predicting future behavior.
- Prevent future clicks or purchases from leaking into training.
- Evaluate candidate recall as well as ranking quality.
- Use Recall@K, NDCG@K, MAP, or hit rate according to the product goal.
- Compare with popularity and simple content-based baselines.
- Measure catalog coverage, novelty, diversity, and subgroup performance where relevant.

Offline ranking metrics do not fully predict user satisfaction. Position bias, exposure, feedback loops, and changing catalogs affect observed interactions. Online experiments need guardrails and careful randomization.

## Cold-start and feedback loops

New users have little history; new items have few interactions. Use onboarding preferences, item metadata, contextual signals, or exploration to address cold start. Monitor whether popularity feedback loops reduce exposure for less-known items.

## Practical checklist

1. Define the recommendation unit and user goal.
2. Document what each interaction event means.
3. Build a popularity baseline.
4. Use a time-aware split and preserve user/item IDs.
5. Compare collaborative and content-based candidates.
6. Evaluate ranking, coverage, diversity, and latency.
7. Monitor changes in exposure and feedback.

## Practice

Create a small user-item matrix from course clicks. Compare popularity recommendations with a simple matrix-factorization model. Use a temporal holdout and calculate Recall@10; discuss what a missing interaction means.

## References

- [Google: Recommendation Systems](https://developers.google.com/machine-learning/recommendation)
- [Implicit library documentation](https://benfred.github.io/implicit/)
- [Surprise: Python recommender systems](https://surpriselib.com/)

**Key takeaway:** Recommendation is more than predicting ratings. It is a retrieval-and-ranking system evaluated against exposure, time, catalog, and user-behavior constraints.