import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


def euclidean_distance(a, b):
    # Euclidean distance (l2 norm)
    # updated for 2-d; computes squared differences between a and b and gets the square root of their sums
    return np.sqrt(np.sum((a - b)**2))

# Step 1


def closestCentroid(x, centroids):
    assignments = []
    for i in x:
        # distance between one data point and centroids
        distance = []
        for j in centroids:
            distance.append(euclidean_distance(i, j))
            # assign each data point to the cluster with closest centroid
        assignments.append(np.argmin(distance))
    return np.array(assignments)


# Step 2
def updateCentroid(x, clusters, K):
    new_centroids = []
    for c in range(K):
        # Update the cluster centroid with the average of all points in this cluster
        # updated to account for possible empty clusters if no data points were assigned yet
        if (clusters == c).any():
            cluster_mean = x[clusters == c].mean(axis=0)
        else:
            # assigns zeros to the cluster mean if the cluster was empty
            cluster_mean = np.zeros(x.shape[1])
        new_centroids.append(cluster_mean)
    return new_centroids


# 2-d kmeans
def kmeans(x, K):
    # initialize the centroids of 2 clusters in the range of [0,20)
    centroids = 20 * np.random.rand(K, x.shape[1])
    print('Initialized centroids: {}'.format(centroids))
    for i in range(10):
        clusters = closestCentroid(x, centroids)
        centroids = updateCentroid(x, clusters, K)
        print('Iteration: {}, Centroids: {}'.format(i, centroids))

    # this section was added to show the final centroids along with the clusters
    plt.scatter(x[:, 0], x[:, 1], s=300)
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
    plt.show()


K = 2
x = np.array([[2, 4],
              [1.7, 2.8], [7, 8], [8.6, 8], [3.4, 1.5], [9, 11]])
kmeans(x, K)
