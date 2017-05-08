# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:35:22 2017

@author: pedro
"""

from __future__ import print_function

import numpy as np

# Load the data
patterns = np.load('/home/pedro/IST/IIEEC/TCGA/brca-patterns.npy')
labels = np.load('/home/pedro/IST/IIEEC/TCGA/brca-labels.npy')

# Remove metastatic patterns
metastatic_patterns_idx = np.where((labels == 2))[0]
patterns = np.delete(patterns, metastatic_patterns_idx, axis=0)
labels = np.delete(labels, metastatic_patterns_idx, axis=0)
del metastatic_patterns_idx

# to use the cross-entropy loss function, the input data must belong
# to the interval [0,1] (on each dimension)
normalized_data = patterns / np.linalg.norm(patterns)
del patterns

# Do K-Fold cross-validation
from sklearn.model_selection import KFold
k = 4
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Prepare oversampling via SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)

# Prepare the autoencoder
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import DA_tutorial_theano as DA
batch_size = 20
hidden_units= 500
corr_level = 0.3
learning_rate = 0.15
training_epochs = 10
visible_units = normalized_data.shape[1]

# Prepare PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=hidden_units)

# Prepare the classifiers
from sklearn import tree
from sklearn.metrics import f1_score
clf_pca = tree.DecisionTreeClassifier()
clf_da = tree.DecisionTreeClassifier()

# Arrays of scores for raw and DA-transformed data
scores_pca = np.zeros((1, k))
scores_da = np.zeros((1, k))

j = 0
fold_splits = kf.split(normalized_data, labels)
for train_index, test_index in fold_splits:
    np.save('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/x_train_' + str(j), normalized_data[train_index])
    np.save('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/y_train_' + str(j), labels[train_index])
    np.save('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/x_test_' + str(j), normalized_data[test_index])    
    np.save('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/y_test_' + str(j), labels[test_index])
    j = j + 1

del normalized_data

i = 0
for i in range(k):
    print('--- iteration no. %d ---' %(i+1))
    
    ### Separate into training and test sets ###
    x_train = np.load('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/x_train_' + str(i) + '.npy')
    y_train = np.load('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/y_train_' + str(i) + '.npy')
    
    print('Oversampling the training set...')
    ## Oversample the training set ###
    x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train.ravel())
    del x_train, y_train
#    ### Transform the data via PCA ### 
#    print('performing PCA...')
#    pca = pca.fit(x_train_sm)    
#    
#    ### Train a decision tree classifier on PCA features of oversampled data ###
#    print('training decision tree...')
#    clf_pca = clf_pca.fit(pca.transform(x_train_sm), y_train_sm)
##    
#    ### Evaluate the classifier with PCA features ###
#    prediction = clf_pca.predict(pca.transform(x_test))
#    scores_pca[0, i] = f1_score(prediction, y_test)
    
    print('building DA...')
    ### Train a DA ###
    floatX_x_train_sm = x_train_sm.astype(theano.config.floatX)
    np.save('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/x_train_sm.npy', x_train_sm)
    del x_train_sm
    symbolic_x_train_sm = theano.shared(floatX_x_train_sm, borrow=True)
    del floatX_x_train_sm
    
    # compute number of minibatches for training
    n_train_batches = symbolic_x_train_sm.get_value(borrow=True).shape[0] // batch_size
    
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')
    rng = np.random.RandomState(123)
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
            x: symbolic_x_train_sm[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    print('training DA...')
    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
    
        print('-training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))    
    
    del train_da
    print('training decision tree...')
    ### Train a decision tree classifier on DA features of oversampled data ###
    x_train_sm = np.load('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/x_train_sm.npy', mmap_mode='r')
    clf_da = clf_da.fit(da.get_hidden_values(x_train_sm).eval(), y_train_sm)
    del x_train_sm, y_train_sm
    
    ### Evaluate the classifier with DA features ###
    print('evaluating decision tree...')
    x_test = np.load('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/x_test_' + str(i) + '.npy')
    y_test = np.load('/home/pedro/IST/IIEEC/simulations/feature_extraction/Thesis-Scripts/data_folds/y_test_' + str(i) + '.npy')
    prediction = clf_da.predict(da.get_hidden_values(x_test).eval())
    del x_test
    scores_da[0, i] = f1_score(prediction, y_test)
    del y_test
    
    i = i + 1
    
print("PCA accuracy: %0.3f (+/- %0.3f)" % (scores_pca.mean(), scores_pca.std() * 2))
print("DA accuracy: %0.3f (+/- %0.3f)" % (scores_da.mean(), scores_da.std() * 2))
