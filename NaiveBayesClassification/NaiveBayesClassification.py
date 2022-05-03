#Naive Bayes Classification

#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing data set
df=pd.read_csv("Social_Network_Ads.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x)
print(y)

#Splitting the data set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

#Training the data set fro naive bayes classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_x, train_y)

#predicting test set results
y_pred=classifier.predict(test_x)
print(y_pred)

#Printing the test set results and predicated results
print(np.concatenate((test_y.reshape(len(test_y),1),y_pred.reshape(len(y_pred),1)),1))

#Confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(test_y,y_pred)
print(cm)
print(accuracy_score(test_y,y_pred))

#Visualizing the training set results
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
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualizing the test set results
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
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
