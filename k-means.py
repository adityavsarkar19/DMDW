import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sample random 2D data arrays x and y
y = -2 * np.random.rand(100,2)
x = 1 + 2 + np.random.rand(100,2)

# Algo for KMeans
# cateogrizing data into clusters based on centroids
def new_clusters(data, centroids, k):
    clusters = {}
    # initializing a brand new set of clusters using new centroids info
    for i in range(k):
        clusters[i] = []
    
    for data in data:
        # list containing distance of data from respective centroids
        distance = []
        # Producing a list of distances of data point from each of the centroids
        for i in range(k):
            distance.append(np.linalg.norm(data-centroids[i]))
        # Appending data point with min dist to the respective cluster (based on centroid)
        clusters[distance.index(min(distance))].append(data)

    return clusters

# Setting new centroid based on cluster formation
def new_centroids(data, clusters, k):
    centroids = {}
    for i in range(k):
        centroids[i] = np.mean(clusters[i], axis=0)
    return centroids

def KMeans(data, clusters, iterations=10, centroids={}):
    k = clusters
    # setting up first set of centroids using random datapoints
    for i in range(k):
        centroids[i] = data[np.random.randint(0, len(data))]
    
    for iters in range(iterations):
        clusters = new_clusters(data, centroids, k)
        centroids = new_centroids(data, clusters, k)

    # Visualization Commands

    plt.figure(figsize=(8,5))

    colors = ('red', 'green', 'blue', 'orange', 'brown', 'violet')
    
    for i in range(k):
        for data in clusters[i]:
            plt.scatter(data[0], data[1], c=colors[i])
        plt.scatter(centroids[i][0], centroids[i][1], c='black')
    plt.show()

if __name__=="__main__":
    KMeans(x, 6)