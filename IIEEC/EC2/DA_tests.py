# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:01:03 2017

Testing the DA class (defined in DA_tutorial_theano.py) on synthetic
datasets

@author: pedro
"""
from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA

import DA_tutorial_theano as DA

#####################################################################
# Generate datasets: dataset1 and dataset2
#####################################################################
# dataset #1: 2D log10 curve
dataset1 = np.zeros((100, 2))
dataset1[:, 0] = np.linspace(1, 100, 100) 
dataset1[:, 1] = np.log(dataset1[:, 0])

# dataset #2: two concentric circles at (0,0) with radius 1.5 and 2
# N samples per circle
N = 100;
M = 2*N;
# theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, N)

# the radius of the circle
r_inner = np.sqrt(1.5)
r_outer = np.sqrt(2)

# create the circles
inner = np.zeros((N, 2))
inner[:, 0] = r_inner*np.cos(theta)
inner[:, 1] = r_inner*np.sin(theta)

outer = np.zeros((N, 2))
outer[:, 0] = r_outer*np.cos(theta)
outer[:, 1] = r_outer*np.sin(theta)

# concatenate the circle data
dataset2 = np.concatenate((inner, outer), axis=0)
#####################################################################
# Apply PCA
#####################################################################
pca = PCA(n_components = 1)

##############################
# dataset #1
##############################
pca_d1 = pca.fit_transform(dataset1)
pca_d1_moved = pca_d1[:,0] - min(pca_d1[:,0])

# plot result
fig = plt.figure()
plt.plot(dataset1[:, 0], dataset1[:, 1], '+', color = 'red')
plt.plot(pca_d1_moved, np.zeros((100, 1)), '+' ,color = 'black')
ax = fig.add_subplot(111)

for i in range(1, 100, 8):
    l = Line2D([pca_d1_moved[i], dataset1[i, 0]], [0, dataset1[i, 1]])
    ax.add_line(l)
    
ax.set_xlim(0, 100)
ax.set_ylim(0, 5)

plt.show()

##############################
# dataset #2
##############################
pca_d2 = pca.fit_transform(dataset2)

# plot the result
fig, ax = plt.subplots(1)
ax.plot(inner[:, 0], inner[:, 1], '.', color = 'red')
ax.plot(outer[:, 0], outer[:, 1], '.', color = 'blue')
plt.plot(pca_d2, np.zeros((2*N, 1)) - 3, '.' ,color = 'black')

inner_i_edges = np.array([0, N/2 - 1])
inner_i_zeros = np.array([N/4 - 1, 3*N/4 - 1])
outer_i_edges = np.array([N, 3*N/2 - 1])
outer_i_zeros = np.array([5*N/4 - 1, 7*N/4 - 1])

for i in inner_i_edges:
    l = Line2D([pca_d2[i], dataset2[i, 0]], [-3, dataset2[i, 1]], color = 'red', linestyle = '--')
    ax.add_line(l)
for i in inner_i_zeros:
    l = Line2D([pca_d2[i], dataset2[i, 0]], [-3, dataset2[i, 1]], color = 'red')
    ax.add_line(l)    
for i in outer_i_edges:
    l = Line2D([pca_d2[i], dataset2[i, 0]], [-3, dataset2[i, 1]], color = 'blue', linestyle = '--')
    ax.add_line(l)    
for i in outer_i_zeros:
    l = Line2D([pca_d2[i], dataset2[i, 0]], [-3, dataset2[i, 1]], color = 'blue')
    ax.add_line(l)      
    
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-3, 2)
ax.set_aspect(1)
plt.show()

#####################################################################
# Apply AE (=0% corruption)
#####################################################################
batch_size = 1
training_epochs = 25
hidden_units = 1
corruption_level = 0.
learning_rate = 0.01

##############################
# dataset #1
##############################

# to use the cross-entropy loss function, the input data must belong
# to the interval [0,1] (on each dimension)
normalized_d1 = dataset1 / np.linalg.norm(dataset1)

floatXd1 = normalized_d1.astype(theano.config.floatX)
symbolic_dataset1 = theano.shared(floatXd1)

# compute number of minibatches for training, validation and testing
n_train_batches = symbolic_dataset1.get_value(borrow=True).shape[0] // batch_size

# allocate symbolic variables for the data
x = T.matrix('x')
index = T.lscalar()    # index to a [mini]batch

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = DA.dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    n_visible=dataset1.shape[1],
    n_hidden=hidden_units
)

cost, updates = da.get_cost_updates(
    corruption_level=corruption_level,
    learning_rate=learning_rate
)

train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: symbolic_dataset1[index * batch_size: (index + 1) * batch_size]
    }
)
            
# go through training epochs
for epoch in range(training_epochs):
    # go through training set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))
        
    # the training cost after an epoch is the mean of the cost over all the mini-batches
    print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))

data = T.dmatrix('data')
features = da.get_hidden_values(x)
compute_features = theano.function([x], features)
encoded_data = compute_features(floatXd1)

# plot the result
fig = plt.figure()
plt.plot(normalized_d1[:, 0], normalized_d1[:, 1], '+', color = 'red')
plt.plot(encoded_data, np.zeros((100, 1)), '+' ,color = 'black')
ax = fig.add_subplot(111)

for i in range(1, 100, 8):
    l = Line2D([encoded_data[i, 0], normalized_d1[i, 0]], [0, normalized_d1[i, 1]])
    ax.add_line(l)

plt.show()
    
##############################
# dataset #2
##############################
# to use the cross-entropy loss function, the input data must belong
# to the interval [0,1] (on each dimension)
normalized_d2 = dataset2 / np.linalg.norm(dataset2)

floatXd2 = normalized_d2.astype(theano.config.floatX)
symbolic_dataset2 = theano.shared(floatXd2)

# compute number of minibatches for training, validation and testing
n_train_batches = symbolic_dataset2.get_value(borrow=True).shape[0] // batch_size

# allocate symbolic variables for the data
x = T.matrix('x')
index = T.lscalar()    # index to a [mini]batch

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = DA.dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    n_visible=dataset2.shape[1],
    n_hidden=hidden_units
)

cost, updates = da.get_cost_updates(
    corruption_level=corruption_level,
    learning_rate=learning_rate
)

train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: symbolic_dataset2[index * batch_size: (index + 1) * batch_size]
    }
)

# go through training epochs
for epoch in range(training_epochs):
    # go through training set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))

    print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))

data = T.dmatrix('data')
features = da.get_hidden_values(x)
compute_features = theano.function([x], features)
encoded_data = compute_features(floatXd2)

# center the encoded_data to facilitate interpretation
encoded_data = encoded_data - np.mean(encoded_data)
# plot the data and their projections
fig, ax = plt.subplots(1)
ax.plot(normalized_d2[:N, 0], normalized_d2[:N, 1], '.', color = 'red')
ax.plot(normalized_d2[N:, 0], normalized_d2[N:, 1], '.', color = 'blue')
plt.plot(encoded_data, np.zeros((M, 1)) - 0.20, '.' ,color = 'black')

inner_i_edges = np.array([M/8 - 1, 3*M/8 - 1])
inner_i_zeros = np.array([0, 2*M/8 - 1])
outer_i_edges = np.array([5*M/8 - 1, 7*M/8 - 1])
outer_i_zeros = np.array([M/2, 6*M/8 - 1])

for i in inner_i_edges:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'red', linestyle = '--')
    ax.add_line(l)
for i in inner_i_zeros:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'red')
    ax.add_line(l)    
for i in outer_i_edges:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'blue', linestyle = '-.')
    ax.add_line(l)    
for i in outer_i_zeros:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'blue')
    ax.add_line(l)      

ax.set_aspect(1)
plt.show()

#####################################################################
# Apply DAE (>0% corruption)
#####################################################################
batch_size = 10
training_epochs = 25
hidden_units = 1
corruption_level = 0.5
learning_rate = 0.1
##############################
# dataset #1
##############################

# to use the cross-entropy loss function, the input data must belong
# to the interval [0,1] (on each dimension)
normalized_d1 = dataset1 / np.linalg.norm(dataset1)

floatXd1 = normalized_d1.astype(theano.config.floatX)
symbolic_dataset1 = theano.shared(floatXd1)

# compute number of minibatches for training, validation and testing
n_train_batches = symbolic_dataset1.get_value(borrow=True).shape[0] // batch_size

# allocate symbolic variables for the data
x = T.matrix('x')
index = T.lscalar()    # index to a [mini]batch

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = DA.dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    n_visible=dataset1.shape[1],
    n_hidden=hidden_units
)

cost, updates = da.get_cost_updates(
    corruption_level=corruption_level,
    learning_rate=learning_rate
)

train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: symbolic_dataset1[index * batch_size: (index + 1) * batch_size]
    }
)
            
# go through training epochs
for epoch in range(training_epochs):
    # go through training set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))
        
    # the training cost after an epoch is the mean of the cost over all the mini-batches
    print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))

data = T.dmatrix('data')
features = da.get_hidden_values(x)
compute_features = theano.function([x], features)
encoded_data = compute_features(floatXd1)

# plot the result
fig = plt.figure()
plt.plot(normalized_d1[:, 0], normalized_d1[:, 1], '+', color = 'red')
plt.plot(encoded_data, np.zeros((100, 1)), '+' ,color = 'black')
ax = fig.add_subplot(111)

for i in range(1, 100, 8):
    l = Line2D([encoded_data[i, 0], normalized_d1[i, 0]], [0, normalized_d1[i, 1]])
    ax.add_line(l)

plt.show()
    
##############################
# dataset #2
##############################
# to use the cross-entropy loss function, the input data must belong
# to the interval [0,1] (on each dimension)
normalized_d2 = dataset2 / np.linalg.norm(dataset2)

floatXd2 = normalized_d2.astype(theano.config.floatX)
symbolic_dataset2 = theano.shared(floatXd2)

# compute number of minibatches for training, validation and testing
n_train_batches = symbolic_dataset2.get_value(borrow=True).shape[0] // batch_size

# allocate symbolic variables for the data
x = T.matrix('x')
index = T.lscalar()    # index to a [mini]batch

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = DA.dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    n_visible=dataset2.shape[1],
    n_hidden=hidden_units
)

cost, updates = da.get_cost_updates(
    corruption_level=corruption_level,
    learning_rate=learning_rate
)

train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: symbolic_dataset2[index * batch_size: (index + 1) * batch_size]
    }
)

# go through training epochs
for epoch in range(training_epochs):
    # go through training set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))

    print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))

data = T.dmatrix('data')
features = da.get_hidden_values(x)
compute_features = theano.function([x], features)
encoded_data = compute_features(floatXd2)

# center the encoded_data to facilitate interpretation
encoded_data = encoded_data - np.mean(encoded_data)
# plot the data and their projections
fig, ax = plt.subplots(1)
ax.plot(normalized_d2[:N, 0], normalized_d2[:N, 1], '.', color = 'red')
ax.plot(normalized_d2[N:, 0], normalized_d2[N:, 1], '.', color = 'blue')
plt.plot(encoded_data, np.zeros((M, 1)) - 0.20, '.' ,color = 'black')

inner_i_edges = np.array([M/8 - 1, 3*M/8 - 1])
inner_i_zeros = np.array([0, 2*M/8 - 1])
outer_i_edges = np.array([5*M/8 - 1, 7*M/8 - 1])
outer_i_zeros = np.array([M/2, 6*M/8 - 1])

for i in inner_i_edges:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'red', linestyle = '--')
    ax.add_line(l)
for i in inner_i_zeros:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'red')
    ax.add_line(l)    
for i in outer_i_edges:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'blue', linestyle = '-.')
    ax.add_line(l)    
for i in outer_i_zeros:
    l = Line2D([encoded_data[i], normalized_d2[i, 0]], [-0.20, normalized_d2[i, 1]], color = 'blue')
    ax.add_line(l)      

ax.set_aspect(1)
plt.show()