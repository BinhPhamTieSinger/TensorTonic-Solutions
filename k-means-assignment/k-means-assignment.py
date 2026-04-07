import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points, centroids = np.array(points), np.array(centroids)
    centroids_point = []
    for point in points:
        nearest_distance = np.inf
        nearest_centroid = np.inf
        for i, centroid in enumerate(centroids):
            distance = np.sum((point - centroid) ** 2)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_centroid = i
        centroids_point.append(nearest_centroid)
    return centroids_point
        