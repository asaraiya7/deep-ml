import numpy as np
from scipy.spatial.distance import cdist

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    points_np = np.array(points)
    initial_centroids = np.array(initial_centroids)
    final_centroids = np.zeros(initial_centroids.shape)

    assert len(initial_centroids) == k, "Num centroids do not match num clusters"

    for i in range(0, max_iterations):
        dist_matrix = cdist(points_np, initial_centroids, metric='euclidean')
        nearest_clust_indices = np.argmin(dist_matrix, axis=1)
        final_centroids = np.array([points_np[nearest_clust_indices == k].mean(axis=0) for k in range(len(initial_centroids))])

    return [tuple(row) for row in final_centroids.tolist()]