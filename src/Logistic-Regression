'''
This program is a part of data prediction for Kaggle Santander Bank challenge

This code is for Logistic Regression, it gets train on train set and predict on test set and outputs the CSV file for submit to Kaggle. 
73 selected features, z-score 4:
Score=0.58464
converged 10000 iteration Score=0.59031

All features:
Score=0.63067

This code is implemented using many great directions from https://machinelearningmastery.com/

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
train_data = train_data[(np.abs(stats.zscore(train_data)) < 4).all(axis=1)]

#read csv files to arrays and convert types
XX = train_data[:,1:(len(train_data[0]))]
YY = train_data[:,0]
TT = test_data[:,1:(len(test_data[0]))]
R = test_data[:,[0]]

T = np.array(TT, dtype=np.float)
X = np.array(XX, dtype=np.float)
Y = np.array(YY, dtype=np.int)


#Logistic Regression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs',max_iter=10000,multi_class='ovr').fit(X, Y)

clf.score(X, Y)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
ynew = clf.predict(T)
ynew = np.reshape(ynew, (-1, 1))
xx = np.append(R,ynew,axis=1)

np.savetxt("submission-LR.csv", (xx),"%s,%i",header="ID_code,target")
