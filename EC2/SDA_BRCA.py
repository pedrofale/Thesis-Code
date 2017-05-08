# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:35:22 2017

@author: pedro
"""

from __future__ import print_function
import timeit

import numpy as np
from utils import shared_dataset

# Load the data
patterns = np.load('/home/ubuntu/Thesis-Scripts/EC2/data/brca-patterns.npy')
labels = np.load('/home/ubuntu/Thesis-Scripts/EC2/data/brca-labels.npy')

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

# Prepare the SDA
from SDA_tutorial_theano import SdA
batch_size = 1
hidden_layers_sizes= [10000, 5000, 2000, 500] 
corruption_levels = [.3, .3, .3, .3]
pretrain_lr=0.001
finetune_lr=0.1
pretraining_epochs=15
training_epochs=1000
visible_units = normalized_data.shape[1]

# Prepare the classifiers
from sklearn import tree
from sklearn.metrics import f1_score
clf_pca = tree.DecisionTreeClassifier()
clf_da = tree.DecisionTreeClassifier()

# Arrays of scores
scores_da = np.zeros((1, k))

j = 0
fold_splits = kf.split(normalized_data, labels)
from sklearn.model_selection import train_test_split
for train_index, test_index in fold_splits:
    train_valid_data = normalized_data[train_index]
    train_valid_labels = labels[train_index]
    x_train, x_valid, y_train, y_valid = train_test_split(train_valid_data, 
                                                          train_valid_labels, test_size=0.2, random_state=42)
    #######################################
    # OVERSAMPLING THE DATA TRAINING DATA #
    #######################################
    print('Oversampling the training set...')
    ## Oversample the training set ###
    x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train.ravel())
    np.save('/home/ubuntu/temp_data/x_train_' + str(j), x_train)
    np.save('/home/ubuntu/temp_data/y_train_' + str(j), y_train)
    np.save('/home/ubuntu/temp_data/x_valid_' + str(j), x_valid)
    np.save('/home/ubuntu/temp_data/y_valid_' + str(j), y_valid)
    np.save('/home/ubuntu/temp_data/x_test_' + str(j), normalized_data[test_index])    
    np.save('/home/ubuntu/temp_data/y_test_' + str(j), labels[test_index])

    j = j + 1

del x_train, x_train_sm, x_valid, y_train, y_train_sm, y_valid, train_valid_data, train_valid_labels    

del normalized_data

i = 0
for i in range(k):
    print('--- iteration no. %d ---' %(i+1))
        
    x_train, y_train = shared_dataset(np.load('/home/ubuntu/temp_data/x_train_' + str(i) + '.npy'), 
                                              np.load('/home/ubuntu/temp_data/y_train_' + str(i) + '.npy'))
      
    ######################
    # BUILDING THE MODEL #
    ######################
    print('building SDA...')
    numpy_rng = np.random.RandomState(89677)
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=visible_units,
        hidden_layers_sizes=hidden_layers_sizes,
        n_outs=2
    )
    
    n_train_batches = x_train.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('getting the pretraining functions...')
    
    pretraining_fns = sda.pretraining_functions(train_set_x=x_train,
                                                batch_size=batch_size)

    print('pre-training the SDA...')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c, dtype='float64')))    
    
    ########################
    # FINETUNING THE MODEL #
    ########################
    # get the training, validation and testing function for the model
    print('getting the finetuning functions...')
    x_valid, y_valid = shared_dataset(np.load('/home/ubuntu/temp_data/x_valid_' + str(i) + '.npy'), 
                                              np.load('/home/ubuntu/temp_data/y_valid_' + str(i) + '.npy'))
    x_test, y_test = shared_dataset(np.load('/home/ubuntu/temp_data/x_test_' + str(i) + '.npy'), 
                                              np.load('/home/ubuntu/temp_data/y_test_' + str(i) + '.npy'))
    datasets = [(x_train, y_train.flatten()), (x_valid, y_valid.flatten()), (x_test, y_test.flatten())]    
    
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('finetunning the model...')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
    
    ##################################
    # USING FEATURES ON A CLASSIFIER #
    ##################################
    print('training decision tree...')
    ### Train a decision tree classifier on SDA features of oversampled data ###
    x_train_data = np.load('/home/ubuntu/temp_data/x_train_' + str(i) + '.npy', mmap_mode='r')
    y_train_data = np.load('/home/ubuntu/temp_data/y_train_' + str(i) + '.npy', mmap_mode='r')
    clf_da = clf_da.fit(sda.get_hidden_values(x_train_data).eval(), y_train_data)
    del x_train_data, y_train_data
    
    ### Evaluate the classifier with SDA features ###
    print('evaluating decision tree...')
    x_test = np.load('/home/ubuntu/temp_data/x_test_' + str(i) + '.npy')
    y_test = np.load('/home/ubuntu/temp_data/y_test_' + str(i) + '.npy')
    prediction = clf_da.predict(sda.get_hidden_values(x_test).eval())
    del x_test
    scores_da[0, i] = f1_score(prediction, y_test)
    del y_test
        
    i = i + 1
    
print("SDA accuracy: %0.5f (+/- %0.5f)" % (scores_da.mean(), scores_da.std() * 2))
