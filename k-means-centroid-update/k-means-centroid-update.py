import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points, assignments = np.array(points), np.array(assignments)
    centroids = []
    
    for i in range(k):
        mask = assignments == i
        if np.any(mask):
            centroid = np.mean(points[mask], axis=0)
        else:
            centroid = np.zeros(points.shape[1])
        centroids.append(centroid.tolist())
        
    return centroids