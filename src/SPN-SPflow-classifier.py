'''
This program is a part of data prediction for Kaggle Santander Bank challenge

This code is for Sum Product Networks (SPNs), it gets train on train set and predict on test set and outputs the CSV file for submit to Kaggle.
score 0.67519
score 0.67537 by removing the outliers

The SPN libraries are from https://github.com/SPFlow/SPFlow
This code is implemented using many great directions from https://machinelearningmastery.com/feature-selection-machine-learning-python/

'''


import numpy as np
import csv
import numpy
from scipy import stats
from numpy import loadtxt
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
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

seed = 7
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
XX = train_data[:,0:(len(train_data[0]))]
X = np.array(XX, dtype=np.float)



from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context

t = [Categorical]
for i in range(200):
	t.append(Gaussian)

#Learning on train data
spn_classification = learn_classifier(X,
                       Context(parametric_types=t).add_domains(X),
                       learn_parametric, 0)


TT = test_data[:,1:(len(test_data[0]))]
R = test_data[:,[0]]

T = np.array(TT, dtype=np.float)
nan = np.array([[np.nan]]*200000)
T = np.append(nan,T,axis=1)
test_classification = T

#predicting on test data
from spn.algorithms.MPE import mpe
#print(mpe(spn_classification, test_classification))		
res = mpe(spn_classification, test_classification)

r = res[:,[0]]
r = np.array(r,dtype=int)
xx = np.append(R,r,axis=1)
np.count_nonzero(r == 1)	
np.savetxt("submission28-spn.csv", (xx),"%s,%i",header="ID_code,target")

