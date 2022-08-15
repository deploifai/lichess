#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# In[110]:


train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

train_winner = train['winner']
test_winner = test['winner']

drop_cols = ['Unnamed: 0', 'winner', 'opening_eco']
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)


# In[105]:


encoder = LabelEncoder()
train_winner = encoder.fit_transform(train_winner)
test_winner = encoder.fit_transform(test_winner)

# train['opening_eco'] = encoder.fit_transform(train['opening_eco'])
# test['opening_eco'] = encoder.fit_transform(test['opening_eco'])


# In[111]:


models = [BernoulliNB(), ComplementNB(), GaussianNB(), MultinomialNB(), KNeighborsClassifier()]
results = []
for model in models:
    model.fit(train, train_winner)
    predict = model.predict(test)
    accuracy = accuracy_score(predict, test_winner)
    results.append(accuracy)
print(results)


# In[ ]:




