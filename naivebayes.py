# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection

data = pd.read_csv('athlete_events.csv')
#Convert medals to medaled vs non-medaled
data['Medal']=data['Medal'].fillna('None')
data['Medal']=data['Medal'].map({'None':0,'Bronze':1,'Silver':1,'Gold':1})
data=data.dropna()

#Converts categories to values
data['Sex']=pd.Categorical(data['Sex']).codes
data['NOC']=pd.Categorical(data['NOC']).codes
data['Season']=pd.Categorical(data['Season']).codes
data['Sport']=pd.Categorical(data['Sport']).codes

#Print various info about the dataset
print(data.sample(n=5))
print(data.shape)
print(data.describe)
print(data.info)
print(data.Medal.value_counts())
print(data.head())
print(data.columns)

features=['ID','Name','Sex','Age','Height','Weight','Team','NOC','Games','Year','Season','City','Sport','Event']

#Select features we care about
featuresC=['Sex','NOC','Season','Sport']
featuresG=['Age','Height','Weight']

x=data[features]
y=data['Medal']


#This does a stratified split into 70% train and 30% test.
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, train_size=0.7, stratify=y)

#Bayes Stuff
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
naive_d=GaussianNB()
naive_d.fit(x_train[featuresG],y_train)
predictionsD=naive_d.predict(x_test[featuresG])

naive_c=CategoricalNB()
naive_c.fit(x_train[featuresC],y_train)
predictionsC=naive_c.predict(x_test[featuresC])

#Output Order for precision and recall is Non-medalists, Medalists
#F-measure computed by hand from these values
from sklearn.metrics import accuracy_score, precision_score, recall_score
print(accuracy_score(y_test,predictionsD))
dPrecision=precision_score(y_test,predictionsD,average=None)
print(dPrecision)
dRecall=recall_score(y_test,predictionsD,average=None)
print(dRecall)

print(accuracy_score(y_test,predictionsC))
cPrecision=precision_score(y_test,predictionsC,average=None)
print(cPrecision)
cRecall=recall_score(y_test,predictionsC,average=None)
print(cRecall)

predictionsCD=(predictionsC+predictionsD)/2
predictionsCDfloor=np.floor(predictionsCD)
print(accuracy_score(y_test,predictionsCDfloor))
print(precision_score(y_test,predictionsCDfloor,average=None))
print(recall_score(y_test,predictionsCDfloor,average=None))

predictionsCDceil=np.ceil(predictionsCD)
print(accuracy_score(y_test,predictionsCDceil))
print(precision_score(y_test,predictionsCDceil,average=None))
print(recall_score(y_test,predictionsCDceil,average=None))