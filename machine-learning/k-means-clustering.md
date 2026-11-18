# K-Means Clustering

K-means is an unsupervised algorithm that partitions observations into K clusters by minimizing the within-cluster sum of squared distances to cluster centroids.

## Algorithm
1. Choose K initial centroids.
2. Assign each observation to its nearest centroid.
3. Recompute each centroid as the mean of assigned observations.
4. Repeat until assignments or centroids stop changing, or a stopping limit is reached.

## Choosing K
- Use domain knowledge where available.
- An elbow plot compares K with within-cluster sum of squares.
- Silhouette scores compare how well points fit their own cluster versus neighboring clusters.

Neither method guarantees meaningful clusters; inspect results in context.

## Practical considerations
- Scale features when their units or ranges differ substantially.
- Run multiple initializations because local optima depend on starting centroids.
- K-means works best for compact, roughly spherical clusters and can struggle with irregular shapes or strong outliers.

## Mini-example
For one-dimensional points {1, 2, 8, 9} with K=2, choose initial centroids 1 and 8. Assign points to the nearest centroid, then recompute means. The clusters are {1, 2} and {8, 9}, with centroids 1.5 and 8.5.

## Exercise
Run K-means on a two-feature dataset with K=2 and K=3. Plot assignments and explain which grouping is more useful for your task.