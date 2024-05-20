import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def k_means(data, k, max_iterations=100):
    iter=0
    min=100000
    for i in range(1):
        #centroids = data[np.random.choice(range(len(data)), k, replace=False)]
        centroids=[[2,10],[5,8],[1,2]]
        for _ in range(max_iterations):
            labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            iter+=1
            centroids = new_centroids
        centroid_distances = np.array([np.linalg.norm(data[labels == i] - centroids[i]) for i in range(k)])
        if min> np.linalg.norm(centroid_distances):
            centroids_final=centroids
            labels_final=labels
            min=np.linalg.norm(centroid_distances)
    
    return centroids_final, labels_final,iter

def plot_clusters(data, centroids, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.', cmap='tab20', s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='black', label='Centroids')
    plt.title('k-means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
data = np.array([[2,10], [2, 5], [8,4], [5, 8], [7,5], [6,4], [1,2], [4,9]])
# data=pd.read_csv('Data_5.csv')
# data=np.column_stack([data['x1'],data['x2']])
# data=pd.read_csv('Vertiport_candidates.csv')
# data=np.column_stack([data['Longitude (deg)'],data['Latitude (deg)']])
k =3

# kmeans_algorithm=KMeans(n_clusters=k,random_state=0).fit(data)
# labels=kmeans_algorithm.labels_
# centroids=kmeans_algorithm.cluster_centers_

centroids, labels,iter = k_means(data, k)
print('Total iterations')
print(iter)
# print("Final cluster centroids:")
# print(centroids)
# print("Labels for each data point:")
# print(labels)

plot_clusters(data, centroids, labels)
