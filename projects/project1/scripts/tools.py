import numpy as np
from costs import *
from helpers import *
from scipy.special import expit # = sigmoid
import logging
import matplotlib.pyplot as plt

"""
train->
validate->
logistic(original->[0,1]->[-1,+1])->

"""


#TODO: remove low relative features
#TODO: 

def compute_loss(y, tx, w):

    e = y - np.dot(tx, w)
    logging.debug(y)
    logging.debug(tx)
    logging.debug(w)
    
    return np.dot(e.T, e) / y.size


def compute_rmse(y, tx, w):
    
    return np.sqrt(2 * compute_loss(y, tx, w))


"""LEARN"""


def compute_gradient(y, tx, w):
    
    return - np.dot(tx.T, y - np.dot(tx, w)) / y.size


def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradien
    loss = compute_loss(y, tx, w)
    
    return loss, w


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    
    w = initial_w
    for n_iter in range(max_iters):
        for mini_y, mini_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(mini_y, mini_tx, w)
            w = w - gamma * gradient
    loss = compute_loss(mini_y, mini_tx, w)
    
    return loss, w


def GD(y, tx, initial_w, grad_f, loss_f, max_iters=10000, threshold=1e-8, gamma=0.1):
    """ general perpose GD """

    w = initial_w
    losses = []
    for n_iter in range(max_iters):
        g = grad_f(y, tx, w)
        w -= gamma * g
        loss = loss_f(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return losses[-1], w


def SGD(y, tx, initial_w, grad_f, loss_f, batch_size=1, max_iters=10000, threshold=1e-8, gamma=0.1):
    """ general perpose SGD """
    
    w = initial_w
    losses = []
    for n_iter in range(max_iters):
        for mini_y, mini_tx in batch_iter(y, tx, batch_size):
            g = grad_f(mini_y, mini_tx, w)
            w = w - gamma * g
            loss = loss_f(y, tx, w)
            losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return losses[-1], w


def least_squares(y, tx):
    
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w) 
    
    return loss, w


def ridge_regression(y, tx, lamb):
    
    a = np.dot(tx.T, tx) + 2 * len(y) * lamb * np.eye(tx.shape[1])
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    #print("ridge")
    
    return loss, w



"""VALIDATE"""


def build_k_indices(y, k_fold=10, seed=42):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def split_data_k(x, y, k_indices):
    
    x_tr = []
    y_tr = []
    x_te = []
    y_te = []
     
    for i in range(len(y)):
        if (i in k_indices):
            x_te.append(x[i])
            y_te.append(y[i])
        else:
            x_tr.append(x[i])
            y_tr.append(y[i])
        
    return np.array(x_tr), np.array(y_tr), np.array(x_te), np.array(y_te)


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    

def cross_validation(y, x, k_indices, lambda_):
    """return the loss of ridge regression."""
    
    x_tr, y_tr, x_te, y_te = split_data_k(x, y, k_indices)
    
    loss, w = ridge_regression(y_tr, x_tr, lambda_)
    
    #print(w.shape)
    
    loss_tr = compute_rmse(y_tr, x_tr, w)
    loss_te = compute_rmse(y_te, x_te, w)
    
    return loss_tr, loss_te


def validate(y, x, k_fold=10, seed=42):
    
    #lambdas = np.logspace(-4, 2, 30)
    lambdas = np.logspace(-4, 2, 5)
    k_indices = build_k_indices(y, k_fold, seed)
    rmse_tr = []
    rmse_te = []
    
    for lamb in lambdas:
        l_tr_sum = 0
        l_te_sum = 0
        for k_th in range(k_fold):
            l_tr, l_te= cross_validation(y, x, k_indices[k_th], lamb)
            l_tr_sum += l_tr
            l_te_sum += l_te
            print("{k} / {f} fold, lambda={l}, training loss={tr:.3f}, testing loss={te:.3f}".format(k=k_th+1, f=k_fold, l=lamb, tr=l_tr, te=l_te))
            
        rmse_tr.append(l_tr_sum / k_fold);
        rmse_te.append(l_te_sum / k_fold);
        
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    

"""PREDICT"""


def loss_log_likelihood(y, tx, w):
    
    l = 0.0
    for n in range(y.shape[0]):
        l += np.log(1 + np.exp(np.dot(tx[n].T, w))) - (y[n] * np.dot(tx[n].T, w))

        
    return l[0] / y.shape[0]


def logistic_GD(y, tx, initial_w, max_iters=10000, gamma=0.001):
    
    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iter):
        w -= gamma * np.dot(tx.T, expit(np.dot(tx, w)) - y)
        loss = loss_log_likelihood(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return losses[-1], w


def reg_logistic_regression(y, tx, lamb, initial_w, max_iters=10000, gamma=0.001):
    
    """
    w = initial_w
    for mini_y, mini_tx in batch_iter(y, tx, batch_size):
        w -= gamma * np.dot(mini_tx.T, expit(np.dot(mini_tx, w)) - mini_y) + 2 * lamb * w
    """   
        
    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iters):
        w -= gamma * np.dot(tx.T, expit(np.dot(tx, w)) - y) + 2 * lamb * w
        loss = np.abs(np.sum(np.log(1 + np.exp(np.dot(tx, w))) - y * np.dot(tx, w))) / y.shape[0]
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss_log_likelihood(y, tx, w), w


def reg_logistic_regression_SGD(y, tx, lamb, initial_w, max_iters=10000, gamma=0.001):
    
    w = initial_w
    batch_size = 1
    for mini_y, mini_tx in batch_iter(y, tx, batch_size):
        w -= gamma * np.dot(mini_tx.T, expit(np.dot(mini_tx, w)) - mini_y) + 2 * lamb * w
     
    """   
    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iters):
        w -= gamma * np.dot(tx.T, expit(np.dot(tx, w)) - y) + 2 * lamb * w
        loss = loss_log_likelihood(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    """
    l = np.abs(np.sum(np.log(1 + np.exp(np.dot(tx, w))) - y * np.dot(tx, w)) + np.dot(lamb * w, w)) / y.shape[0]
    
    return l, w


def predict_log(data, w):
    y_pred = expit(np.dot(data, w))
    y_pred[np.where(y_pred < 0.5)] = -1
    y_pred[np.where(y_pred >= 0.5)] = 1
    
    return y_pred


#TODO: feed trained data into logistic