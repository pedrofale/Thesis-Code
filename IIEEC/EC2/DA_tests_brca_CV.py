# -*- coding: utf-8 -*-
"""
Applying a DA to reduce the dimensionality of the BRCA dataset
and using the obtained features in the subsequent classification problem

@author: pedro
"""

from __future__ import print_function

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import DA_tutorial_theano as DA

patterns = numpy.load('/home/pedro/IST/IIEEC/TCGA/brca-patterns.npy')
labels = numpy.load('/home/pedro/IST/IIEEC/TCGA/brca-labels.npy')

###################
# DATA PROCESSING #
###################

# Remove metastatic patterns
metastatic_patterns = numpy.where((labels == 2))[0]
patterns = numpy.delete(patterns, metastatic_patterns, axis=0)
labels = numpy.delete(labels, metastatic_patterns, axis=0)

# to use the cross-entropy loss function, the input data must belong
# to the interval [0,1] (on each dimension)
normalized_data = patterns / numpy.linalg.norm(patterns)
del patterns

########################
# TRAIN AND TEST SPLIT #
########################
from sklearn.model_selection import train_test_split
# train_test_split shuffles the data internally
x_train, x_test, y_train, y_test = train_test_split(normalized_data, labels, 
                                                    test_size=0.33, random_state=42)
del normalized_data, labels

# Balance data classes
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_train_smote, y_train_smote = sm.fit_sample(x_train, y_train)
numpy.save('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/x_train.npy', x_train)
del x_train

#########
# SETUP #
#########

batch_size = 200
hidden_units= 500
corr_level = 0.
learning_rate = 0.15
training_epochs = 10
visible_units = x_train_smote.shape[1]

#############################################
# STORE THE TEST DATA TO SAVE SCRIPT MEMORY #
#############################################
numpy.save('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/x_test.npy', x_test)
del x_test

floatX_x_train = x_train_smote.astype(theano.config.floatX)
del x_train_smote

symbolic_x_train = theano.shared(floatX_x_train)
del floatX_x_train

# compute number of minibatches for training, validation and testing
n_train_batches = symbolic_x_train.get_value(borrow=True).shape[0] // batch_size

# allocate symbolic variables for the data
index = T.lscalar()    # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images

######################
# BUILDING THE MODEL #
######################

rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = DA.dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    n_visible=visible_units,
    n_hidden=hidden_units
)

cost, updates = da.get_cost_updates(
    corruption_level=corr_level,
    learning_rate=learning_rate
)

train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: symbolic_x_train[index * batch_size: (index + 1) * batch_size]
    }
)

############
# TRAINING #
############

print("Training...")
# go through training epochs
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))

    print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

#############################
# TRANSFORM ALL DATA POINTS #
#############################
# This is done by using the original data as input to the encoder 
# part of the AE. The features for each data point appear at the output of the encoder 
#x = T.matrix('x')
#features = da.get_hidden_values(x)
#compute_features = theano.function([x], features)
#floatXdata = x_train[0,:].astype(theano.config.floatX).
#t_x_train_1 = compute_features(floatXdata)

####################################################
# TRAIN DECISION TREES ON THE RAW AND NEW FEATURES #
####################################################
from sklearn import tree
from sklearn.model_selection import cross_val_score

x_train = numpy.load('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/x_train.npy')
x_test = numpy.load('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/x_test.npy')
x_full = numpy.append(x_train, x_test, axis = 0)
del x_train, x_test
y_full = numpy.append(y_train, y_test.ravel(), axis = 0)
del y_train, y_test

clf_raw = tree.DecisionTreeClassifier()
raw_scores = cross_val_score(clf_raw, x_full, y_full, cv=10)

clf_da = tree.DecisionTreeClassifier()
x_full_transformed = da.get_hidden_values(x_full)
del x_full
da_scores = cross_val_score(clf_da, x_full_transformed.eval(), y_full, cv=10)

print("Raw accuracy: %0.3f (+/- %0.3f)" % (raw_scores.mean(), raw_scores.std() * 2))
print("DA accuracy: %0.3f (+/- %0.3f)" % (da_scores.mean(), da_scores.std() * 2))

###########
# RESULTS #
###########
# Raw accuracy: 0.993 (+/- 0.022)
# DA (500 units) accuracy: 0.918 (+/- 0.067)

### What's going on? ###
# It looks like the performance in the raw data is too good, which may
# be a sign of the model overfitting the data. This is probably due to
# the fact that we performed SMOTE in the whole dataset and only after
# did we separate the data into training and test sets. As said in
# https://stats.stackexchange.com/questions/60180/testing-classification-on-oversampled-imbalance-data,
# "copies of the same point may end up in both the training and test
# sets. This allows the classifier to cheat, because when trying to make
# predictions on the test set the classifier will already have seen the
# identical points in the training set."

### What should I do? ###
# Separate the data into training and test sets and perform SMOTE only
# on the training data. Train the DA with the oversampled training
# data. Then, to train the classifier, set as the whole dataset the 
# oversampled training data and the original test data. 