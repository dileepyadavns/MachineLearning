
#Hierarchial Clusterring
#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing The DataSet

df=pd.read_csv("Datasets/Mall_Customers.csv")
x=df.iloc[:,[3,4]].values

#Drawing the Dendagram to find the number of clusster
import scipy.cluster.hierarchy as sch
dendgrm=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#Training the data set for hierarchial clusterring
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity = 'euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

#Visualizing the Hierarchial Clusttering
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],c='red',s=100,label='cluster1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],c='brown',s=100,label='cluster2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],c='green',s=100,label='cluster3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],c='pink',s=100,label='cluster4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],c='blue',s=100,label='cluster5')
plt.title("Hierarchial Clusterring")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
