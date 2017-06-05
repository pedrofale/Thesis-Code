# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:35:22 2017

@author: pedro
"""

from __future__ import print_function
import timeit
import os

import numpy as np
from utils import shared_dataset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Remove cached data
print('Removing cached data...')
import os, shutil
folder = '/home/ubuntu/temp_data'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
print('done')

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
from sklearn.preprocessing import normalize
normalized_data = normalize(patterns)
del patterns

# Do K-Fold cross-validation
from sklearn.model_selection import KFold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Prepare oversampling via SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
sm = SMOTE(random_state=42)

# Prepare the SDA
from SDA_tutorial_theano import SdA
batch_size = 1
hidden_layers_sizes= [10] 
corruption_levels = [0.]
pretrain_lr=0.5
finetune_lr=0.5
pretraining_epochs=10
training_epochs=1000
visible_units = normalized_data.shape[1]
finetune = True
# Prepare the classifiers
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
clf_svm = SVC()
clf_dt = tree.DecisionTreeClassifier()

# Arrays of scores
acc_scores = np.zeros((2, k))
prc_scores = np.zeros((2, k))
f1_scores = np.zeros((2, k))

j = 0
fold_splits = kf.split(normalized_data, labels)
from sklearn.model_selection import train_test_split
for train_index, test_index in fold_splits:
    train_valid_data = normalized_data[train_index]
    train_valid_labels = labels[train_index]
    x_train, x_valid, y_train, y_valid = train_test_split(train_valid_data, 
                                                          train_valid_labels, test_size=0.2, random_state=42)
   
    np.save('/home/ubuntu/temp_data/x_train_' + str(j), x_train)
    np.save('/home/ubuntu/temp_data/y_train_' + str(j), y_train)
    #######################################
    # OVERSAMPLING THE DATA TRAINING DATA #
    #######################################
    print('Oversampling the training set...')
    ## Oversample the training set ###
    x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train.ravel())
    # shuffle the oversampled to make it more uniform
    x_train_sm, y_train_sm = shuffle(x_train_sm, y_train_sm, random_state=42)
    np.save('/home/ubuntu/temp_data/x_train_sm_' + str(j), x_train_sm)
    np.save('/home/ubuntu/temp_data/y_train_sm_' + str(j), np.reshape(y_train_sm, (y_train_sm.shape[0], 1)))
    np.save('/home/ubuntu/temp_data/x_valid_' + str(j), x_valid)
    np.save('/home/ubuntu/temp_data/y_valid_' + str(j), y_valid)
    np.save('/home/ubuntu/temp_data/x_test_' + str(j), normalized_data[test_index])    
    np.save('/home/ubuntu/temp_data/y_test_' + str(j), labels[test_index])

    j = j + 1

del x_train, x_train_sm, x_valid, y_train, y_train_sm, y_valid, train_valid_data, train_valid_labels    

del normalized_data

for j in range(k):
    print('--- iteration no. %d ---' %(j+1))
        
    x_train_sm, y_train_sm = shared_dataset(np.load('/home/ubuntu/temp_data/x_train_sm_' + str(j) + '.npy'), 
                                              np.load('/home/ubuntu/temp_data/y_train_sm_' + str(j) + '.npy'))
      
    ######################
    # BUILDING THE MODEL #
    ######################
    print('building SDA...')
    numpy_rng = np.random.RandomState(np.random.randint(0, 10000))
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=visible_units,
        hidden_layers_sizes=hidden_layers_sizes,
        n_outs=2
    )
    
    n_train_batches = x_train_sm.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('getting the pretraining functions...')
    
    pretraining_fns = sda.pretraining_functions(train_set_x=x_train_sm,
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
    x_train, y_train = shared_dataset(np.load('/home/ubuntu/temp_data/x_train_sm_' + str(j) + '.npy'), 
                                              np.load('/home/ubuntu/temp_data/y_train_sm_' + str(j) + '.npy'))
    x_valid, y_valid = shared_dataset(np.load('/home/ubuntu/temp_data/x_valid_' + str(j) + '.npy'), 
                                              np.load('/home/ubuntu/temp_data/y_valid_' + str(j) + '.npy'))
    x_test, y_test = shared_dataset(np.load('/home/ubuntu/temp_data/x_test_' + str(j) + '.npy'), 
                                              np.load('/home/ubuntu/temp_data/y_test_' + str(j) + '.npy'))

    if finetune == True:   
        datasets = [(x_train, y_train.flatten()), (x_valid, y_valid.flatten()), (x_test, y_test.flatten())]    
     
        train_fn, validate_model, test_model = sda.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )
        n_train_batches = x_train.eval().shape[0]
        n_train_batches //= batch_size

        print('finetunning the model...')
        valid_no_0 = len(np.where((y_valid.eval() == 0))[0])
        valid_no_1 = len(np.where((y_valid.eval() == 1))[0])
        valid_total = valid_no_0 + valid_no_1
        print('Bad results: %f or %f in validation set' % (valid_no_0/valid_total * 100.0, valid_no_1/valid_total * 100.0))
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
            i=0
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                i = i + 1
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = np.mean(validation_losses, dtype='float64')
                    print('epoch %i, minibatch %i/%i, validation error %f %%, training cost %f' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100., minibatch_avg_cost))

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

    del x_valid, y_valid
    ##################################
    # USING FEATURES ON A CLASSIFIER #
    ##################################
    print('training decision tree...')
    ### Train a decision tree classifier on SDA features of original data ###
    clf_dt = clf_dt.fit(sda.get_hidden_values(x_train).eval(), y_train.eval())
    # Grid search on the SVM-RBF parameters
    print('finding parameters for SVM-RBF...')
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(sda.get_hidden_values(x_train).eval(), y_train.eval().ravel())
    
    print("...the best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    clf_svm = SVC(C=grid.best_params_["C"], gamma=grid.best_params_["gamma"])
    print('training an SVM-RBF with optimal parameters...')
    clf_svm = clf_svm.fit(sda.get_hidden_values(x_train).eval(), y_train.eval().ravel())
    del x_train, y_train
    
    ### Evaluate the classifier with SDA features ###
    print('evaluating decision tree...')
    x_test = np.load('/home/ubuntu/temp_data/x_test_' + str(j) + '.npy')
    y_test = np.load('/home/ubuntu/temp_data/y_test_' + str(j) + '.npy')
    dt_prediction = clf_dt.predict(sda.get_hidden_values(x_test).eval())
    print('evaluating decision tree...')
    svm_prediction = clf_svm.predict(sda.get_hidden_values(x_test).eval())
    del x_test
    acc_scores[0, j] = accuracy_score(dt_prediction, y_test)
    acc_scores[1, j] = accuracy_score(svm_prediction, y_test)
    prc_scores[0, j] = precision_score(dt_prediction, y_test)
    prc_scores[1, j] = precision_score(svm_prediction, y_test)
    f1_scores[0, j] = f1_score(dt_prediction, y_test)
    f1_scores[1, j] = f1_score(svm_prediction, y_test)
    print("DT %0.5f acc, %0.5f prec, %0.5f f1" %(acc_scores[0,j],prc_scores[0,j],f1_scores[0,j]))
    print("SVM-RBF %0.5f acc, %0.5f prec, %0.5f f1" %(acc_scores[1,j],prc_scores[1,j],f1_scores[1,j]))
    del y_test

print("### All results: ###")
print("\tAcc\tPrec\tF1")
print("- DT:")
for i in range(k):
    print("It. 1: %0.5f\t%0.5f\t%0.5f" %(acc_scores[0,i],prc_scores[0,i],f1_scores[0,i]))
print()
print("\tAcc\tPrec\tF1")
print("- SVM-RBF:")
for i in range(k):
    print("It. 1: %0.5f\t%0.5f\t%0.5f" %(acc_scores[1,i],prc_scores[1,i],f1_scores[1,i]))
    
print("\n####################################################\n")        
    
print("SDA (%i layer(s), %0.2f corruption level) performance:" % (len(hidden_layers_sizes), corruption_levels[0]))
print("- Accuracy: DT %0.5f (+/- %0.5f) || SVM-RBF %0.5f (+/- %0.5f)" % (acc_scores[0,:].mean(), acc_scores[0,:].std() * 2, acc_scores[1,:].mean(), acc_scores[1,:].std() * 2))
print("- Precision: DT %0.5f (+/- %0.5f) || SVM-RBF %0.5f (+/- %0.5f)" % (prc_scores[0,:].mean(), prc_scores[0,:].std() * 2, prc_scores[1,:].mean(), prc_scores[1,:].std() * 2))
print("- F1: DT %0.5f (+/- %0.5f) || SVM-RBF %0.5f (+/- %0.5f) " % (f1_scores[0,:].mean(), f1_scores[0,:].std() * 2, f1_scores[1,:].mean(), f1_scores[1,:].std() * 2))

#import matplotlib.pyplot as plt
#plt.figure()
#plt.boxplot()
#plt.title("SVM-RBF")