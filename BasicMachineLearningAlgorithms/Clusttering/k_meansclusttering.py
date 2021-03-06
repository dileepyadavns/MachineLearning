
#K-Means Clusttering

#Importing The Libraries

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt

#Importing the Dataset
df=pd.read_csv("Datasets/Mall_Customers.csv")
x=df.iloc[:,[3,4]].values

#Finding the Largest K value that is optimal for the Model
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  K_means=KMeans(n_clusters=i, init='k-means++', random_state=42)
  K_means.fit(x)
  wcss.append(K_means.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Training the dataset for k_means Clusterring
k_means=KMeans(n_clusters=5, init='k-means++',random_state=42)
y_kmeans=k_means.fit_predict(x)

#Visualizing the K-Means Clusttering
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],color='red',s=100,label='cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],c='blue',s=100,label='cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],c='pink',s=100,label='cluster3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],c='green',s=100,label='cluster4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],c='brown',s=100,label='cluster5')
plt.title("KMeans Clusttering")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()
