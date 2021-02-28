import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


# DATA LOADING

# sample random 2D data arrays x and y
x = 3 + np.random.rand(50,2)

# Urban Great Britan Car Crash location data
filepath = "../DMDW/datasets/urbanGB_crashLoc.csv"
df = pd.read_csv(filepath)
GB_array = np.array(df) # use GB_array in KMeans function


# K-MEANS ALGORITHM
# ------------------------------------------------------------------------
# Following the given steps to produce the K-Means clustering algorithm:

# 1. Select k centroids. These will be the center point for each segment.
# 2. Assign data points to nearest centroid.
# 3. Reassign centroid value to be the calculated mean value for each cluster.
# 4. Reassign data points to nearest centroid.
# 5. Repeat until data points stay in the same cluster.

# reference: https://www.jeremyjordan.me/grouping-data-points-with-k-means-clustering/
# ------------------------------------------------------------------------

#read CSV/TXT file
def ReadData(fileName):  
  
    # Read the file, splitting by lines  
    f = open(fileName, 'r');  
    lines = f.read().splitlines();  
    f.close();  
  
    items = [];  
  
    for i in range(1, len(lines)):  
        line = lines[i].split(',');  
        itemFeatures = [];  
  
        for j in range(len(line)-1):  
              
            # Convert feature value to float 
            v = float(line[j]);  
              
            # Add feature value to dictionary  
            itemFeatures.append(v);  
  
        items.append(itemFeatures);  
  
    shuffle(items);  
  
    return items; 

# euclidean distance calculation
def dist_bw(a, b):
    # Returns euclidean distance between two points
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# assigning data into clusters based on centroids
def new_clusters(data, centroids, k):
    clusters = {}
    # initializing a brand new set of clusters using new centroids info
    for i in range(k):
        clusters[i] = []
    
    for data in data:
        # list containing distances of data from respective centroids
        distances = []
        # Producing a list of distances of data point from each of the centroids
        for i in range(k):
            distances.append(dist_bw(data, centroids[i]))
        # Appending data point with min dist to the respective cluster (based on centroid)
        clusters[distances.index(min(distances))].append(data)

    return clusters

# Setting new centroids based on cluster formation
def new_centroids(data, clusters, k):
    centroids = {}
    # new centroids based on the mean of all points in the new cluster
    for i in range(k):
        centroids[i] = np.mean(clusters[i], axis=0)
    
    return centroids

# Visualization Commands
def visualize_kmeans(clusters, centroids, og_centroids, k):
    plt.figure(figsize=(6,6))
    colors = ('red', 'green', 'blue', 'orange', 'purple', 'darkslateblue', 'cyan')
    
    for i in range(k):
        for data in clusters[i]:
            plt.scatter(data[0], data[1], c=colors[i])
        plt.scatter(og_centroids[i][0], og_centroids[i][1], c='black', marker='o')
        plt.scatter(centroids[i][0], centroids[i][1], c='black', marker='X')
    plt.show()

# k-means iterations
def KMeans(data, n_clusters, centroids={}):
    k = n_clusters
    iterations = 0
    # setting up first set of centroids using random datapoints
    if centroids == {}:
        for i in range(k):
            centroids[i] = data[np.random.randint(0, len(data))]
    og_centroids   = centroids
    prev_centroids = centroids

    while True:
        clusters = new_clusters(data, centroids, k)
        centroids = new_centroids(data, clusters, k)
        # check to see if iterations need to stop
        bool_list = [(prev_centroids[i][0] == centroids[i][0]) and
                     (prev_centroids[i][1] == centroids[i][1]) for i in range(k)]
        if all(bool_list) == True:
            print(f"No Further Changes: {iterations} completed iterations\n")
            print(f"Initial Centroids: {og_centroids}\nFinal Centroids: {centroids}")
            break
        iterations += 1
        prev_centroids = centroids

    # visualization of clusters and centroids
    visualize_kmeans(clusters, centroids, og_centroids, k)


if __name__=="__main__":
    items = ReadData('test.txt')
    kmeans_array = []
    for i in items:
        kmeans_array.append([i[0], i[1]])

    KMeans(GB_array, 4)
