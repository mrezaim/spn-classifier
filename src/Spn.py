import numpy as np
from numpy import loadtxt
import csv
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
from spn.algorithms.MPE import mpe
#from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier

class Spn:
  def __init__(self, train, test,zscore,features):
    self.train = train
    self.test = test
    
    np.random.seed(5)

    # Read the CSV into a pandas data frame (df)
    dftr = pd.read_csv(train, delimiter=',')
    train_data = np.array(dftr)
    dfte = pd.read_csv(test, delimiter=',')
    test_data = np.array(dfte)

    train_data = train_data[(np.abs(stats.zscore(train_data)) < zscore).all(axis=1)]
    #read csv files to arrays and convert types
    XX = train_data[:,0:(len(train_data[0]))]
    X = np.array(XX, dtype=np.float)



    #Learning on train data
    spn_classification = learn_classifier(X,
                        Context(parametric_types=features).add_domains(X),
                        learn_parametric, 0)


    TT = test_data[:,1:(len(test_data[0]))]
    R = test_data[:,[0]]

    T = np.array(TT, dtype=np.float)
    nan = np.array([[np.nan]]*len(train))
    T = np.append(nan,T,axis=1)
    test_classification = T

    res = mpe(spn_classification, test_classification)

    r = res[:,[0]]
    r = np.array(r,dtype=int)
    xx = np.append(R,r,axis=1)
    #np.count_nonzero(r == 1)
    def saveresults(self,results):
        np.savetxt(self.results, (xx),"%s,%i",header="ID_code,target")
