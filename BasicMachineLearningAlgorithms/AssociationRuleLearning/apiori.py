
pip install apyori #installing apyori

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset

df=pd.read_csv("Datasets/Market_Basket_Optimisation.csv",header=None)


trans=[]
for i in range(0,7501):
  row=[]
  for j in range(0,20):
    row.append(str(df.values[i,j]))
  trans.append(row)

print(trans)

#Training the Aprorio Model on Dataset

from apyori import apriori
rules=apriori(transactions=trans,min_support=0.003,min_confidence=0.2,min_length=2,max_length=2,min_lift=3)

#Visualizing the results

#Displaying the first results coming directly from the output of the apriori function

results=list(rules)
print(results)

#Organizing the data on The Table creating DataFrame

def inspect(results):
  new=[]
  for result in results:
    lh=tuple(result[2][0][0])[0]

    rh=tuple(result[2][0][1])[0]

    support=result[1]

    confidence=result[2][0][2]

    lift=result[2][0][3]
    new.append((lh,rh,support,confidence,lift))  

  return new
newDataFrame=pd.DataFrame(inspect(results),columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print(newDataFrame)

#Displaying the results sorted by descending lifts

newDataFrame.nlargest(n = 10, columns = 'Lift')
print(newDataFrame)
