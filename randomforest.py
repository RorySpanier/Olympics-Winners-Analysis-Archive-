# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Read in data
data = pd.read_csv('athlete_events.csv')
#Convert medals to medaled vs non-medaled
data['Medal']=data['Medal'].fillna('None')
data['Medal']=data['Medal'].map({'None':0,'Bronze':1,'Silver':1,'Gold':1})
data=data.dropna()

data.drop(['ID','Name','Team','Games','City','Event'],axis=1,inplace=True)
featuresC=['Sex','NOC','Season','Sport']
featuresD=['Age','Height','Weight','Year']

#One-hot encode the data using pandas get_dummies, normalize discrete variables
data=pd.get_dummies(data,columns=featuresC,drop_first=True)
data[featuresD]=(data[featuresD]-data[featuresD].min())/(data[featuresD].max()-data[featuresD].min())

# Labels are the values we want to predict
labels = np.array(data['Medal'])

# Remove the labels from the features
# axis 1 refers to the columns
data = data.drop('Medal', axis = 1)

# Saving feature names for later use
feature_list = list(data.columns)

# Convert to numpy array
data = np.array(data)

#This does a stratified split into 70% train and 30% test.
train_features, test_features, train_labels, test_labels = train_test_split(data, labels, train_size = 0.7, stratify=labels)

# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100)

# Train the model on training data
rf.fit(train_features, train_labels);
rfPredictions=rf.predict(test_features)
#Round fuzzy predictions
rfPredictions=np.round(rfPredictions)

#Output Order for precision and recall is Non-medalists, Medalists
#F-measure computed by hand from these values
print(accuracy_score(test_labels,rfPredictions))
print(precision_score(test_labels,rfPredictions,average=None))
print(recall_score(test_labels,rfPredictions,average=None))
