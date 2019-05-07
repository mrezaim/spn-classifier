# SPN-Classifier [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Team Members
Mohammadreza Iman\
Saber Soleymani

## Getting Started

To setup the environment and run our dataset, or your own dataset for a binary classification task using Sum-Product Networks and SPFlow, follow the steps below. The optional prerequisities are just needed to run other classifiers mentioned in the Problem Statement.

## Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [SPFlow] (https://github.com/SPFlow/SPFlow)
- <em>(Optional)</em> [Keras](https://keras.io/#installation) - Open-source neural network library
- <em>(Optional)</em> [Tensorflow](https://www.tensorflow.org/) - API used as Backend for Keras

## Problem Statement

The problem focuses on the binary classification task using Sum-Product Networks (SPNs). However, we have provided implementations of other classifiers (i.e., Deep Neural Network, Logistic Regression, and Random Forest) mainly to compare with SPNs. The dataset we have tested our code on, is a dataset provided by Santander Bank that tries to predict customers' behavior regarding their future decision (yes or no) of a transaction. 

## Installation

### SPFlow
SPFlow is an extensible and customizable library for inferencing, learning, and manupulating Sum-Product Networks (SPNs). To use SPFlow libarary, you can install it by  running the ```pip install spflow --user``` command.

### Keras

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. We use it based on TensorFlow for our binary classification task. You can install keras using pip on command line ```pip install keras --user```.

### Tensorflow 

You can install Tensorflow using pip on command line, for CPU ```pip install tensorflow --user``` and for GPU ```pip install tensorflow-gpu --user```

## Dataset

Anonymized dataset containing numeric feature variables, the binary target column (classification), and a string ID_code column. There is no deatils about the feature, just tabular data. No correlation between data, the highest correlation is 0.08. There are about 200 features. Very unbalanced training set with less than 15% having 1.\
**Training dataset:** 200,000 samples consist of 201 features and the target\
**Test dataset:** 200,000 instances consist of same 201 features of the training set

## SPN Results 
|           Conditions           |Accuracy on Testing dataset|
|---------------------------|---------------------------|
|All features, No outlier removal |0.67519|
|All features, Z-Score < 3 |0.67537|
|**All features, Z-Score < 4** |**0.67538**|
|27 features, Z-Score < 4 |0.56130|
|73 features, Z-Score < 4 |0.63150|
|150 features, Z-Score < 4 |0.66761|

## Other Results 

|           Model           |Accuracy on Training dataset|Accuracy on Testing dataset|
|---------------------------|---------------------------|---------------------------|
|Regression|0.1799866|NA|
|Ridge regression|0.1799114|NA|
|LASSO regression|0.1799598|NA|
|Decision Tree|0.8396|0.5662|
|**Random Forest**|**0.7241**|**0.8050**|
|Support Vector Machine (Kernel:Sigmoid)|0.9005|0.5003|
|Support Vector Machine (Kernel:Poly)|0.9000|0.5000|
|Support Vector Machine (Kernel:Gaussian)|0.9101|0.5539|
|Sum-Product Network|0.6768|0.6753|
|Neural Network|0.7224|0.7253|
|Logistic Regression|NA|0.6306|
|Logistic Regression, Z-score < 4, 73 features|NA|0.5846|

## Acknowledgments
- This project was completed as a part of the Decision Making Under Uncertainty course taught by Professor Doshi, in Spring 2019 at the University of Georgia. 

### References
- For Sum-Product Networks this works is heavily based on https://github.com/SPFlow/SPFlow 
- For the Support Vector Machine codes we have used codes in ...
- For the Deep Neural Network implementation we have used codes in https://machinelearningmastery.com/feature-selection-machine-learning-python/
- https://machinelearningmastery.com/feature-selection-machine-learning-python/
- For the Logistic Regression we have used codes in https://machinelearningmastery.com/
