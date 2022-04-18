#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading Dataset
df=pd.read_csv("Datasets/Ads_CTR_Optimisation.csv")
print(df)

#Implementing Upper confidence bound
import math #importing math module
N=10000
d=10
ad_selected=[]
no_times_adselected=[0] * d #will show list containing zero of length d
sum_of_rewards=[0] * d #will show list containing zero of length d
total_rewards=0

for n in range(0,N): #iterate over all the rows 
    ad=0
    max_ucb=0
    for i in range(0,d): #iterate over all the ads
        if (no_times_adselected[i] > 0 ):
            average_reward=sum_of_rewards[i] / no_times_adselected[i]
            delta_i=math.sqrt(3/2 * math.log(n+1) / no_times_adselected[i])
            upper_bound=average_reward+delta_i
        else:
            upper_bound=1e400
        if (upper_bound > max_ucb):
            max_ucb=upper_bound
            ad=i
    ad_selected.append(ad) #appending the add with max upper bound
    no_times_adselected[ad]=no_times_adselected[ad] + 1
    reward=df.values[n,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_rewards=total_rewards+reward

#Visualizing the results
print(ad_selected)
print(len(ad_selected))
plt.hist(ad_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()        

