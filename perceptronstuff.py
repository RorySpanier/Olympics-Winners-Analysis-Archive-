# -*- coding: utf-8 -*-
import pandas as pd
import sklearn as sk
import sklearn.model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score

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

features=['ID','Name','Sex','Age','Height','Weight','Team','NOC','Games','Year','Season','City','Sport','Event']

#Select features we care about
featuresC=['Sex','NOC','Season','Sport']
featuresD=['Age','Height','Weight','Year']

x=data[features]
y=data['Medal']

#Encode variables correctly for the classifier
needDummies=['NOC','Season','Sport']
dataDumb=pd.get_dummies(data,columns=needDummies)
dataDumb=dataDumb.drop(['ID','Name','Team','Games','City','Event'],axis=1)
dataDumb[featuresD]=(dataDumb[featuresD]-dataDumb[featuresD].min())/(dataDumb[featuresD].max()-dataDumb[featuresD].min())

print(dataDumb.columns)
print(dataDumb.head())

#Perceptron Stuff
from sklearn.neural_network import MLPClassifier
xp=dataDumb.drop('Medal',axis=1)
yp=dataDumb['Medal']

#This does a stratified split into 70% train and 30% test.
xp_train, xp_test, yp_train, yp_test = sk.model_selection.train_test_split(xp, yp, train_size=0.7, stratify=yp, random_state=500)

perc=MLPClassifier(hidden_layer_sizes=(25,25,25,25),max_iter=1000)
perc.fit(xp_train,yp_train)
pPredictions=perc.predict(xp_test)


#Output Order for precision and recall is Non-medalists, Medalists
#F-measure computed by hand from these values
print(accuracy_score(yp_test,pPredictions))
print(precision_score(yp_test,pPredictions,average=None))
print(recall_score(yp_test,pPredictions,average=None))