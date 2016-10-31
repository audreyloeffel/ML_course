import numpy as np
import matplotlib.pyplot as plt
from regression import ridge_regression
from least_squares import least_squares
from proj1_helpers import *
from helpers import standardize

# Function used to preprocess the data
def prepare(x):
    """
    Prepare the data by standardizing and replacing unused 
    values (-999) by the mean of their columns such that they
    don't affect the computation then.
    """
    # Here we put the non sense values (-999) to mean 
    # such that then with the standardization they will be set to 0
    # And we count the number of -999 values to add this information to
    N = x.shape[0]
    novalues_len = np.zeros((x.shape[0], x.shape[1]))
    useless_features = []
    
    
    xt = np.copy(x.T)
    i = 0
    for xi in xt:
        xi[xi==-999] = np.nan
        nanidx = np.where(np.isnan(xi))
        number_noval = nanidx[0].shape[0]
        if number_noval >= N/2:
            useless_features.append(i)
        i = i + 1
    
    i = 0
    for xi in xt.T:
        nanidx = np.where(np.isnan(xi))
        novalues_len[i] = nanidx[0].shape[0]
        i = i + 1
        
    for xi in xt:
        xi[xi==-999] = np.nan
        m = np.nanmean(xi)
        nanidx = np.where(np.isnan(xi))
        xi[nanidx] = m
    
    tx = xt.T
    tx = np.delete(tx, useless_features, axis=1)
    tx = np.hstack((tx, novalues_len))
    
    tx, mean, std = standardize(tx)
    
    return tx

def build_poly(x, degree = 5):
    tX_poly = np.power(x, 0)
    tX_d = np.log2(np.abs(x))
    tX_poly = np.hstack((tX_poly, tX_d))
    tX_d = np.log10(np.square(x))
    tX_poly = np.hstack((tX_poly, tX_d))
    
    for d in range(0, degree):
        tX_d = np.power(x, d+1)
        tX_poly = np.hstack((tX_poly, tX_d))
       
    return tX_poly


# Import train data
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Preprocess data
tx = prepare(tX)
tx_poly = build_poly(tx)

# Train with ridge regression
lamb = 0.0000000001
weights, loss = ridge_regression(y, tx_poly, lamb)

# Import test data
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Preprocess data
tx_test = prepare(tX_test)
tx_test_poly = build_poly(tx_test)

# Create the output file
OUTPUT_PATH = '../output/out.csv'
y_pred = predict_labels(weights, tx_test_poly)

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)



