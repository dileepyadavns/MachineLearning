
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset

dataset=pd.read_csv("/content/Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
# here delimiter is assigned value '\t' which indicates that we are working with the tsv file.
# quoting parameter is used to ignore all the quotes presnt in the text file

# Cleaning the tex

import re
import nltk
nltk.download('stopwords')

#nltk library used to download all the ensemble of stopwards which dont help in the prediction.

#nltk.download used to download all stopswords
from nltk.corpus import stopwords

#the downloaded stopwords are stored in nltk.corpus we use above command to import the stopwords into our note book
from nltk.stem.porter import PorterStemmer
#nltk.stem.porter is used to import PorterStemmer which is used to apply stemming on our text
# if we dont apply stemming there will separete columns for each words which have similar meaning having different verb forms this makes our dataframe large.
corpus=[]
for i in range(0,1000):

  review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #Replacing the punctuation marks into empty charcter using sub function.
 
  review=review.lower()  
  review=review.split() 
  ps=PorterStemmer()
  all_words=stopwords.words('english')
  all_words.remove('not') #here we excluded "not" from stopwords as it is a negative word we want it.
  review= [ ps.stem(word) for word in review if word not in set(all_words)] 
  #applied stemming for each word which are not stopwords
  
  review=' '.join(review)

  corpus.append(review)

print(corpus)

"""# Creating Bag of Model"""

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) #max_features will set the maximum ammount features to be considered by doing vectorization.
x=cv.fit_transform(corpus).toarray() 
y=dataset.iloc[:,-1].values

print(len(x[0]))

# Splitting the Dataset into train set and test set
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2, random_state=0)

#training the datset on Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(train_x,train_y)

# Predicting the test case results
y_pred=classifier.predict(test_x)
print(y_pred)

print(np.concatenate((y_pred.reshape(len(y_pred),1) ,test_y.reshape(len(test_y),1)),1))

# Confusion Matrix
from sklearn.metrics import accuracy_score,confusion_matrix

cm=confusion_matrix(test_y,y_pred)
print(cm)
print(accuracy_score(test_y,y_pred))
