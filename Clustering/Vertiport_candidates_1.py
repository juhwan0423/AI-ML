import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

def find_nearest(data,centroids):
   
   # centroids=np.array(centroids)
    points=[]
    for centroid in centroids:
        min=10000
        for ind_data in data:
            if min>np.linalg.norm(centroid-ind_data):
                min=np.linalg.norm(centroid-ind_data)
                point=ind_data
        points.append(point)
        
    return points

def k_means(data, k, max_iterations=100):
    
    min=100000
    for i in range(200):
        centroids = data[np.random.choice(range(len(data)), k, replace=False)]
        for _ in range(max_iterations):
            labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        centroid_distances = np.array([np.linalg.norm(data[labels == i] - centroids[i]) for i in range(k)])
        if min> np.linalg.norm(centroid_distances):
            centroids_final=centroids
            labels_final=labels
            min=np.linalg.norm(centroid_distances)
    
    return centroids_final, labels_final

def plot_clusters(data, centroids):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], marker='.', s=100,c='b')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='.', s=200, c='r', label='Stops')
    plt.title('k-means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


data=pd.read_csv('Vertiport_candidates.csv')
data=np.column_stack([data['Longitude (deg)'],data['Latitude (deg)']])
k =6

kmeans_algorithm=KMeans(n_clusters=k,random_state=0).fit(data)
labels=kmeans_algorithm.labels_
centroids=kmeans_algorithm.cluster_centers_

centroids_all, labels = k_means(data, k)

# print("Final cluster centroids:")
# print(centroids)
# print("Labels for each data point:")
# print(labels)
centroids_final=[]
cluster_counts=Counter(labels)
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster + 1}: {count}개의 데이터 포인트")
    if count==4056:
        centroid,_=k_means(data[labels==cluster],4)
        # points=find_nearest(data[labels==cluster],centroid)
        centroids_final+=centroid.tolist()
    if count==1635:
        centroid,_=k_means(data[labels==cluster],4)
        #points=find_nearest(data[labels==cluster],centroid)
        centroids_final+=centroid.tolist()
    if count==883 or count==506:
        centroid,_=k_means(data[labels==cluster],3)
        #points=find_nearest(data[labels==cluster],centroid)
        centroids_final+=centroid.tolist()
    if count==593 :
        centroid,_=k_means(data[labels==cluster],2)
        #points=find_nearest(data[labels==cluster],centroid)
        centroids_final+=centroid.tolist()
    if count==101:
        centroid,_=k_means(data[labels==cluster],1)
        #points=find_nearest(data[labels==cluster],centroid)
        centroids_final+=centroid.tolist()
centroids=np.array(centroids_final)
points=find_nearest(data,centroids)
points=np.array(points)   

plt.figure(1)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], marker='.', s=100,c='b',label='Given')
plt.scatter(points[:, 0], points[:, 1], marker='.', s=200, c='r', label='Candidates')
plt.title('k-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(2)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.', cmap='tab20', s=100)
plt.scatter(centroids_all[:, 0], centroids_all[:, 1], marker='x', s=200, c='black', label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(3)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.', cmap='tab20', s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='black', label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()