# Evaluating Clustering Without Ground-Truth Labels

Clustering is often unsupervised, so evaluation is less straightforward than classification. No single score proves that clusters are useful.

## Internal metrics
- **Silhouette coefficient:** Compares within-cluster cohesion with nearest-cluster separation; ranges from -1 to 1.
- **Davies-Bouldin index:** Lower values indicate clusters that are compact and separated.
- **Calinski-Harabasz index:** Higher values generally indicate dense, separated clusters.

These metrics favor particular geometries and can disagree. For example, silhouette may not suit non-convex clusters.

## External metrics
When trusted labels exist for evaluation only, adjusted Rand index and normalized mutual information can compare cluster assignments with labels. Do not use labels to tune an allegedly unsupervised workflow if the task forbids it.

## Beyond a score
- Inspect cluster sizes and outliers.
- Visualize with PCA or UMAP, remembering that 2D projections can distort distances.
- Check stability across random seeds and resampled datasets.
- Ask domain experts whether clusters are interpretable and actionable.

## Exercise
Run K-means for several values of k, record silhouette and Davies-Bouldin scores, inspect cluster sizes, and explain why the selected k is useful for the task.

**Key point:** A mathematically separated cluster is not automatically a meaningful business segment.