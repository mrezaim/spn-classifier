'''
This program is a part of data prediction for Kaggle Santander Bank challenge

This code is for random forest model, it gets train on train set and predict on test set and outputs the CSV file for submit to Kaggle.
This code has the best adjusted parameters through several tuning and submissions. Score = 0.627 with threshold of 0.3.

This code is implemented using many great directions from https://machinelearningmastery.com/feature-selection-machine-learning-python/

'''

import numpy as np
import csv
import numpy
from scipy import stats
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

seed = 5
numpy.random.seed(seed)

# Read the CSV into a pandas data frame (df)
dftr = pd.read_csv('train.csv', delimiter=',')
train_data = np.array(dftr)
dfte = pd.read_csv('test.csv', delimiter=',')
test_data = np.array(dfte)

#Detecting and Removing outliers
train_data = np.array(train_data[:,1:(len(train_data[0]))], dtype=np.float)
train_data = train_data[(np.abs(stats.zscore(train_data)) < 3).all(axis=1)]

#read csv files to arrays and convert types
XX = train_data[:,1:(len(train_data[0]))]
YY = train_data[:,0]
TT = test_data[:,1:(len(test_data[0]))]
R = test_data[:,[0]]

T = np.array(TT, dtype=np.float)
X = np.array(XX, dtype=np.float)
Y = np.array(YY, dtype=np.int)

#Filter selected features
sel_f= np.array([ True,  True,  True, False, False, False,  True, False, False,True, False, False,  True,  True, False, False, False, False,True, False, False,  True,  True, False, False, False,  True, False, False, False, False, False,  True,  True,  True, False, False, False, False, False,  True, False, False,  True,  True, False, False, False, False, False, False,  True, False,  True, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False,  True,  True, False,  True, False,  True, True, False, False, False, False,  True, False, False,  True, False,  True,  True, False,  True,  True, False, False, False, True, False, False, False, False, False, False,  True,  True, True,  True,  True, False, False, False, False,  True, False, False, False,  True, False,  True,  True, False, False, False, False,  True, False, False,  True, False, False,  True, False, False, False, False, False,  True, False,  True, False, False, False,  True,  True,  True,  True, False, False, False, False, False,  True,  True, False,  True, False, False, False, False, True,  True,  True,  True,  True, False, False,  True,  True, False,  True,  True,  True, False, False,  True, False,  True, True, False, False, False,  True, False, False, False,  True, False,  True,  True, False, False, False, False, False,  True, True, False])

T = T[:,sel_f]
X = X[:,sel_f]

#Dividing the train data to train and validation sets for scaled and unscaled data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

# Feature Scaling
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  
T = sc.transform(T)

#RandomForest model 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
regressor.fit(X_train, y_train)
  
y_pred = regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

ynew = regressor.predict(T)
np.savetxt("submission14.csv", ynew,header="target")

