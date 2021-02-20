
# coding: utf-8

# In[3]:


from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random


# In[4]:


df=pd.read_csv("C:\\Users\\Amit George\\downloads\\breast-cancer-wisconsin.datas.text.txt")
df.drop(['5','2.1','1.5','1.4','3','1.3','2','1.2','1.1','1','1000025'],1,inplace=True)
df


# In[5]:


df.drop(['id'],1,inplace=True)
df


# In[6]:


df.replace('?',-99999,inplace=True)
data=df.astype('float').values.tolist()
data


# In[7]:


def K_nearest_neighbors(train_set,data,k=5):
    if len(train_set)>=k:
        warnings.warn('the number of voting groups is greater than the k values')
    distances=[]
    list=[]
    for group in train_set:
        for features in train_set[group]:
            euclid_dist=np.linalg.norm(np.array(features)-np.array(data))
            distances.append([euclid_dist,group])
    for i in sorted(distances)[:k]:
            list.append(i[1])
            vote_result=Counter(list).most_common()[0][0]
            print(vote_result)    
    return vote_result               
#result= K_nearest_neighbors(dataset,new_features,k=3)  

#for i in dataset:
   # for ii in dataset[i]:
     #   plt.scatter(ii[0],ii[1], s=100 ,color=i)
#plt.scatter(new_features[0],new_features[1],color=result)
#plt.show()
         


# In[8]:


random.shuffle(data)


# In[9]:


test_size=0.2
train_set={2:[], 4:[]}
test_set={4:[], 2:[]}
training_set=data[:-int(test_size*len(data))]
testing_set=data[-int(test_size*len(data)):]


# In[10]:


for i in training_set:
    train_set[i[-1]].append(i[:-1])
for i in testing_set:
    test_set[i[-1]].append(i[:-1])


# In[11]:


total=0
correct=0
for group in test_set:
    for data in test_set[group]:
        votes = K_nearest_neighbors(train_set,data,k=5)
        if group==votes:
            correct+=1
        total+=1
print('accuracy',correct/total)
        

