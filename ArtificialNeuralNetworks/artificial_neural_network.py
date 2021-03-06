
# Imporing Libraries

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__ #to check version

#Importing Dataset

df=pd.read_csv("Churn_Modelling.csv")
x=df.iloc[:,3:-1].values   
y=df.iloc[:,-1].values

print(x)
print(y)

#Encoding Categorical Data

#Encoding Geneder Column

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

print(x)

#Encoding Country Column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
#passthrough keeps remaining columns as it is, 
x=np.array(ct.fit_transform(x))

print(x)

#training the Data Set

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25, random_state=0)

#Feature Scalling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

#Building the ANN

#Initializing ANN

ann=tf.keras.models.Sequential()

#Adding input first layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the second Outer layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding Output Layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Training the ANN

#Compiling the ANN

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the ANN on test set

ann.fit(train_x, train_y, batch_size = 32, epochs = 100)

#making predictions on single observation

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

y_pred = ann.predict(test_x)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), test_y.reshape(len(test_y),1)),1))

#confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(test_y, y_pred)
print(cm)
accuracy_score(test_y, y_pred)
