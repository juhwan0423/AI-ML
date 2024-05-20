import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# 주어진 centroid와 가장 가까운 데이터를 찾기.
def find_nearest(data,centroids):
    points=[]
    for centroid in centroids:
        min=10000
        for ind_data in data:
            if min>np.linalg.norm(centroid-ind_data):
                min=np.linalg.norm(centroid-ind_data)
                point=ind_data
        points.append(point)
        
    return points

# k_means 알고리즘
def k_means(data, k, max_iterations=120):
    
    min=100000
    for i in range(150):
        centroids = data[np.random.choice(range(len(data)), k, replace=False)]
        for _ in range(max_iterations):
            labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        centroid_distances = np.array([np.linalg.norm(data[labels == i] - centroids[i]) for i in range(k)])
        
        #얻어진 K_means의 결과가 잘 되었는지 판단하기.
        if min> np.linalg.norm(centroid_distances):
            centroids_final=centroids
            labels_final=labels
            min=np.linalg.norm(centroid_distances)
    
    return centroids_final, labels_final,min

def elbow(data,k,centroids):
    sum_dis=0
    for i in range(k):
        for data_ind in data[labels==i]:
            sum_dis+=np.linalg.norm(data_ind-centroids[i])
    return sum_dis
data=pd.read_csv('Vertiport_candidates.csv')
data=np.column_stack([data['Longitude (deg)'],data['Latitude (deg)']])
index=[]
distances=[]

#적절한 K 찾아 보기.(elbow 방법사용)
for k in range(1,13):
    # centroids, labels = k_means(data, k)
    # distance=elbow(data,k,centroids)
    _,_,distance=k_means(data,k)
    distances.append(distance)
    index.append(k)
plt.figure(figsize=(8, 6))
plt.plot(index,distances)
plt.xlabel('K')
plt.ylabel('WWE')
plt.title('Proper K')
plt.legend()
plt.grid(True)
plt.show()
k=8
centroids_all, labels,_ = k_means(data, k)
centroids_final=[]
add_index=0

#각 클러스터의 데이터 개수 세기.
cluster_counts=Counter(labels)

#각 클러스터의 데이터 개수에 따라 새로 클러스터링 하기.
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster + 1}: {count}개의 데이터 포인트")
    if count>2000:
        centroid,_,_=k_means(data[labels==cluster],3)
        centroids_final+=centroid.tolist()
    elif count>900:
        centroid,_,_=k_means(data[labels==cluster],3)
        centroids_final+=centroid.tolist()
        add_index=cluster
    elif count>400:
        centroid,_,_=k_means(data[labels==cluster],2)       
        centroids_final+=centroid.tolist()
    else :
        centroid,_,_=k_means(data[labels==cluster],1)
        centroids_final+=centroid.tolist()
for data_ind in data[labels==add_index] :
    if data_ind[0]>128.2 and data_ind[1]<38.01:
        centroids_final.append(data_ind)
  
centroids=np.array(centroids_final)
points=find_nearest(data,centroids)
points=np.array(points)   

#k=8로 전체 데이터 클러스터링 했을 때 시각화
plt.figure(1)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.', cmap='tab20', s=100)
plt.scatter(centroids_all[:, 0], centroids_all[:, 1], marker='x', s=200, c='black', label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('Longitude(deg)')
plt.ylabel('Latitude(deg)')
plt.legend()
plt.grid(True)
plt.show()

#클러스터링 데이터 개수에 따라 재 클러스터링 했을 때 시각화
plt.figure(1)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.', cmap='tab20', s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='black', label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('Longitude(deg)')
plt.ylabel('Latitude(deg)')
plt.legend()
plt.grid(True)
plt.show()

#주어진 데이터에서 실제 후보지들을 선택했을 때 시각화
plt.figure(3)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], marker='.', s=100,c='b',label='Given')
plt.scatter(points[:, 0], points[:, 1], marker='.', s=200, c='r', label='Candidates')
plt.title('k-means Clustering')
plt.xlabel('Longitude(deg)')
plt.ylabel('Latitude(deg)')
plt.legend()
plt.grid(True)
plt.show()
