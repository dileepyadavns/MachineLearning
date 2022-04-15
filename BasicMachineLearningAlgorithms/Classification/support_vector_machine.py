
#Support Vector Machines

#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Data sets
df=pd.read_csv("Datasets/Social_Network_Ads.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#Splitting the datasets Into train and test sets
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature Scalling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

#Training the dataset for Support vector Machine 
from sklearn.svm import SVC

classifier=SVC(kernel='linear',random_state=0)
classifier.fit(train_x,train_y)

#Predicting test set results
y_pred=classifier.predict(test_x)
print(np.concatenate((y_pred.reshape(len(y_pred),1),test_y.reshape(len(test_y),1)),1))

#Predicting new Results
print(classifier.predict(sc.transform([[30,86000]])))

#Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(test_y,y_pred))
print(accuracy_score(test_y,y_pred))

#Visualizing the test results for SVM

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
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualizing Test set results in SVM
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(test_x),test_y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
