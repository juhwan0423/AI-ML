import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd


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
        if min> np.linalg.norm(centroid_distances):
            centroids_final=centroids
            labels_final=labels
            min=np.linalg.norm(centroid_distances)
    
    return centroids_final, labels_final

data=pd.read_csv('South_Korea_territory.csv')
data=np.column_stack([data['Longitude (deg)'],data['Latitude (deg)']])
# Convex Hull 알고리즘을 사용하여 경계를 찾음
hull = ConvexHull(data)

# 경계 내부에 균일하게 분포된 점 생성
num_points =1000
min_x, min_y = np.min(data, axis=0)
max_x, max_y = np.max(data, axis=0)

random_points = np.random.uniform(low=[min_x, min_y], high=[max_x, max_y], size=(num_points, 2))


# 경계 내부에 있는 점을 필터링하여 선택
inner_points = []

for point in random_points:
    inside = True
    for equation in hull.equations:
        if equation[:-1] @ point > -equation[-1]:
            inside = False
            break
    if inside:
        
        if point[1]>35 and point[1]<36.8 and point[0]<126.5:
            continue
            #point[0]=126.8+0.2*(0.5-np.random.random())
        if point[1]>37 and point[1]<37.8 and point[0]<126.5: 
            continue
            #point[0]=126.8+0.1*(0.5-np.random.random())
        inner_points.append(point)

inner_points = np.array(inner_points)
#data_array=np.vstack((data,inner_points))
data_array=inner_points

k=10
kmeans_cluster_center,kmeans_labels=k_means(data_array,k)

# 각 클러스터의 x 및 y의 최소 최대값을 저장할 리스트 초기화
cluster_centers = []

# 각 클러스터의 중심점을 기준으로 최소 최대값 구하기
for i in range(k):
    cluster_points = data_array[kmeans_labels == i]  # 클러스터 내의 점들 선택
    min_x = np.min(cluster_points[:, 0])
    max_x = np.max(cluster_points[:, 0])
    min_y = np.min(cluster_points[:, 1])
    max_y = np.max(cluster_points[:, 1])
    center_x=(min_x+max_x)/2
    center_y=(min_y+max_y)/2
    center=[center_x,center_y]
    cluster_centers.append(center)
cluster_centers=np.array(cluster_centers)    

# 시각화
plt.figure(1)
plt.figure(figsize=(8, 8))
plt.plot(data[:, 0], data[:, 1], '.', label='Given Data')
plt.plot(data[hull.vertices, 0], data[hull.vertices, 1], 'r--', lw=2, label='Convex Hull Inside')
plt.plot(inner_points[:, 0], inner_points[:, 1], 'g.', label='Inside Uniform points')
plt.xlabel('Longitude(deg) ')
plt.ylabel('Latitude(deg) ')
plt.title('Points within Shape')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

plt.figure(2)
plt.figure(figsize=(8, 8))
plt.scatter(data_array[:,0],data_array[:,1],c=kmeans_labels, marker='.',cmap='tab20',s=50)
plt.xlabel('Longitude(deg)',fontsize=10)
plt.ylabel('Latitude(deg)',fontsize=10)
plt.title('K_means Clustering')
plt.grid(True,alpha=0.3,linestyle='--')
plt.axis('equal')

plt.figure(3)
plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], c='b',marker='.',s=10, label='Given Data')
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],c='r',marker='.',s=100,label='Points')
plt.xlabel('Longitude(deg)',fontsize=10)
plt.ylabel('Latitude(deg)',fontsize=10)
plt.title('Candiates with Centroids')
plt.grid(True,alpha=0.3,linestyle='--')
plt.axis('equal')
plt.legend()

plt.figure(4)
plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], c='b',marker='.',s=10, label='Given Data')
plt.scatter(kmeans_cluster_center[:,0],kmeans_cluster_center[:,1],c='r',marker='.',s=100,label='Points')
plt.xlabel('Longitude(deg)',fontsize=10)
plt.ylabel('Latitude(deg)',fontsize=10)
plt.title('Candidates with centers')
plt.grid(True,alpha=0.3,linestyle='--')
plt.axis('equal')
plt.legend()