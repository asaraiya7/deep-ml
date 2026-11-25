import numpy as np
from scipy.spatial.distance import cdist


def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points = np.array(points)
    query_point = np.array(query_point).reshape(1, -1)
    nearest_dist = cdist(points, query_point, metric='euclidean').ravel()
    sorted_indices = np.argsort(nearest_dist, kind='stable')

    return [tuple(p) for p in points[sorted_indices[:k]].tolist()]