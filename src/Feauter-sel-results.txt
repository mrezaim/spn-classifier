'''
This file  is a part of data prediction for Kaggle Santander Bank challenge

In this file the result of three feature selection methods are available beside the directions of the processes to generate them.
Each array has 200 values of True/False, which are showing which features had been selected corresponding to 200 features of Kaggle dataset, respectively.

''' 

'''
This features are selected using Recursive Feature Elimination (RFE) method by Logistic Regression, Feature_sel_RFE_and_Statistics.py. 150 featu been selected with rank 1.

'''
array([ True,  True,  True,  True,  True,  True,  True, False,  True,
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

Ranking of features:
array([ 1,  1,  1,  1,  1,  1,  1, 45,  1,  1, 51,  1,  1,  1, 21,  1,  1,
       49,  1, 27,  1,  1,  1,  1,  1,  1,  1, 33,  1,  9, 50,  1,  1,  1,
        1,  1,  1,  1, 40, 36,  1, 44,  1,  1,  1, 32, 19, 31,  3,  1,  1,
        6,  1,  1, 17,  1,  1,  1,  1,  1, 22, 38,  1,  1,  1, 10,  1,  1,
        1, 25, 11,  1,  1, 37, 24,  1,  1,  1,  1,  1,  1,  1,  7,  8, 20,
        1,  1,  1,  1,  1, 14,  1,  1,  1,  1,  1, 43, 28,  1,  1, 47, 18,
       13,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 46,  1,
        1, 35,  1,  1,  1, 16,  1,  1,  1,  1, 23,  1,  1,  1,  1,  2,  1,
       42,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        4,  1,  1,  1,  1, 39,  1, 41,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  5,  1,  1,  1,  1, 29,  1, 15,  1,  1,  1, 30, 34,  1, 48,  1,
       26,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 12])

'''
This features are selected using random forest model, Feature_sel_randomforest.py.
73 features had been selected.
'''
sel_f_RF = np.array([ True,  True,  True, False, False, False,  True, False, False,True, False, False,  True,  True, False, False, False, False,True, False, False,  True,  True, False, False, False,  True, False, False, False, False, False,  True,  True,  True, False, False, False, False, False,  True, False, False,  True,  True, False, False, False, False, False, False,  True, False,  True, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False,  True,  True, False,  True, False,  True, True, False, False, False, False,  True, False, False,  True, False,  True,  True, False,  True,  True, False, False, False, True, False, False, False, False, False, False,  True,  True, True,  True,  True, False, False, False, False,  True, False, False, False,  True, False,  True,  True, False, False, False, False,  True, False, False,  True, False, False,  True, False, False, False, False, False,  True, False,  True, False, False, False,  True,  True,  True,  True, False, False, False, False, False,  True,  True, False,  True, False, False, False, False, True,  True,  True,  True,  True, False, False,  True,  True, False,  True,  True,  True, False, False,  True, False,  True, True, False, False, False,  True, False, False, False,  True, False,  True,  True, False, False, False, False, False,  True, True, False])


'''
This features are selected based on their absolute correlation values to the target, the best correlation is 0.080917332, which is so low. The threshold for this selection was decided based on rule of thumb and is bigger or equal to 0.05. 
27 features had been selected. The following code and some simple excel processes were used for extracting these features(Univariate selection).
#Statistics
dftr = pd.read_csv('train.csv', delimiter=',')
dftr.describe().to_csv('describe.csv', sep=',')
dftr.corr().to_csv('corr.csv', sep=',')
dftr.cov().to_csv('cov.csv', sep=',')
'''
sel_f_corr = np.array([True, True, True, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False])
