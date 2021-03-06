
#Support Vector Machine Kernel

#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Data set
df=pd.read_csv("Datasets/Social_Network_Ads.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#Splitiing the data frame into train and test data
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=0)

print(test_x)

#Scalling the dataset
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

print(test_x)

#training the data set for Spport vector Machine Kernel
from sklearn.svm import SVC
classifier=SVC(kernel="rbf")
classifier.fit(train_x,train_y)

#Predict the test set Results
y_pred=classifier.predict(test_x)
print(y_pred)

"""**Confusion Matrix**"""

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(test_y,y_pred)
print(cm)
print(accuracy_score(test_y,y_pred))

#Visualizing train set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(train_x), train_y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualizing Test Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(test_x), test_y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



