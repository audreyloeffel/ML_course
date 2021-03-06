# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def normalize(tX):
    nullValue = -999
    tX[tX==nullValue] = 0
    m = np.mean(tX, axis=0)
    tX_centered = tX - m

    tX_centered[tX_centered==0] = float('nan')
    tX_std= np.nanstd(tX_centered, axis=0)
    tX_centered[tX_centered==float('nan')] = 0
    normalized = tX_centered / tX_std
    return normalized

def polynomialBasis(tX, degree = 2, combi = False):
    tX_poly = tX
    
    for d in range(1, degree):
        tX_d = np.power(tX, d+1)
        tX_poly = np.hstack((tX_poly, tX_d))
   
    if combi :
        nb = tX.shape[1]
        for i in range(nb): 
            print("Iteration: ", i)
            for j in range(10, 20):
                if(i!=j):
                    coli = tX[:, i]
                    colj = tX[:, j]
                    tX_poly = np.hstack((tX_poly, (coli * colj).reshape(tX.shape[0], 1)))
       
    return tX_poly
        