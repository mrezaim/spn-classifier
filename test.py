# import keras as K
# import tensorflow as tf
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.utils import to_categorical
# from keras import models
# from keras import layers
# from keras.layers import Dropout
# import csv

# from scipy import stats
# from numpy import loadtxt
# from urllib.request import urlopen
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import class_weight
# from sklearn.metrics import confusion_matrix
# from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# from sklearn.model_selection import train_test_split # Import train_test_split function
# from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import SelectFromModel

from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context

# configs
# seed = 5
# numpy.random.seed(seed)
use_gpu = False
train_path = '/home/saber/Dropbox/Projects/kaggle/santander/dataset/train.csv'
test_path = '/home/saber/Dropbox/Projects/kaggle/santander/dataset/test.csv'


# Read the CSV into a pandas data frame (df)
dftr = pd.read_csv(train_path, delimiter=',')
train_data = np.array(dftr)
dfte = pd.read_csv(test_path, delimiter=',')
test_data = np.array(dfte)

#read csv files to arrays and convert types
XX = train_data[:,1:(len(train_data[0]))]
X = np.array(XX, dtype=np.float)
#TODO: the first column as integer since is the binary class

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
T = np.append(T,nan,axis=1)
test_classification = T
#predicting on test data
from spn.algorithms.MPE import mpe
print(mpe(spn_classification, test_classification))