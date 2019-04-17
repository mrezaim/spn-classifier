'''
This program is a part of data prediction for Kaggle Santander Bank challenge

This code is for Neural network technique, it gets train on train set and predict on test set and outputs the CSV file for submit to Kaggle.
This code has the best adjusted parameters through several tuning and submissions. Score = 0.72533.

This code is implemented using many great directions from https://machinelearningmastery.com/feature-selection-machine-learning-python/

'''
import keras as K
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.layers import Dropout
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

# Normalization for NN models
target_count = dftr.target.value_counts()
class_weights = {0: round(target_count[1] / target_count[0],2),
				 1: round(5*target_count[0] / target_count[1],2)}
print(class_weights)

#read csv files to arrays and convert types
XX = train_data[:,1:(len(train_data[0]))]
YY = train_data[:,0]
TT = test_data[:,1:(len(test_data[0]))]
R = test_data[:,[0]]

T = np.array(TT, dtype=np.float)
X = np.array(XX, dtype=np.float)
Y = np.array(YY, dtype=np.int)

#Filter selected features
sel_f= np.array([ True,  True,  True,  True,  True,  True,  True, False,  True,
        True, False,  True,  True,  True, False,  True,  True, False,
        True, False,  True,  True,  True,  True,  True,  True,  True,
       False,  True, False, False,  True,  True,  True,  True,  True,
        True,  True, False, False,  True, False,  True,  True,  True,
       False, False, False, False,  True,  True, False,  True,  True,
       False,  True,  True,  True,  True,  True, False, False,  True,
        True,  True, False,  True,  True,  True, False, False,  True,
        True, False, False,  True,  True,  True,  True,  True,  True,
        True, False, False, False,  True,  True,  True,  True,  True,
       False,  True,  True,  True,  True,  True, False, False,  True,
        True, False, False, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
       False,  True,  True, False,  True,  True,  True, False,  True,
        True,  True,  True, False,  True,  True,  True,  True, False,
        True, False,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
       False,  True,  True,  True,  True, False,  True, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
       False,  True,  True,  True,  True, False,  True, False,  True,
        True,  True, False, False,  True, False,  True, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True, False])

T = T[:,sel_f]
X = X[:,sel_f]

#Dividing the train data to train and validation sets for scaled and unscaled data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Feature Scaling
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  
T = sc.transform(T)

# create model
model = Sequential()
model.add(Dense(int(X[0].size/2), input_dim=X[0].size, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(int(X[0].size/3), kernel_initializer='normal', activation='sigmoid'))

model.add(Dense(int(X[0].size/4), kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['binary_accuracy'])



earlyStopping = K.callbacks.EarlyStopping(monitor='binary_accuracy', min_delta=0, patience=100, verbose=1, mode='auto', restore_best_weights=True)

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=650, batch_size=10000, class_weight=class_weights, callbacks=[earlyStopping])
model.evaluate(X_test, y_test)[1]*100

y_pred = model.predict_classes(X_train)
conf_mat = confusion_matrix(y_true=y_train, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

y_pred = model.predict_classes(X_test)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

ynew = model.predict_classes(T)
xx = np.append(R,ynew,axis=1)
np.count_nonzero(ynew == 1)	
np.savetxt("submission23.csv", (xx),"%s,%i",header="ID_code,target")

