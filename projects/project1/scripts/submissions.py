# -*- coding: utf-8 -*-
import numpy as np

"""
	Costs functions
"""
def calculate_mse(e):
	"""Calculate the mse for vector e."""
	return 1/2*np.mean(e**2)


def calculate_mae(e):
	"""Calculate the mae for vector e."""
	return np.mean(np.abs(e))


def compute_loss(y, tx, w):
	"""Calculate the loss.

	You can calculate the loss using mse or mae.
	"""
	e = y - tx.dot(w)
	return calculate_mse(e)
	# return calculate_mae(e)


def calculate_nll(y, tx, w):
	"""calculate the negative log likelihood cost."""
	N = y.shape[0]
	precLim = 10
	xw = np.dot(tx, w)
	
	yxw = y * xw
	xw[xw<precLim] = np.log(1 + np.exp(xw[xw<precLim]))
	loss = xw - yxw

	return np.sum(loss, axis=0)

"""
	batch iter
"""
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

"""
	Least squares functions
"""

def compute_gradient(y, tx, w):
	"""Compute gradient for batch data."""
	N = y.shape[0]
	e = y - np.dot(tx, w)

	gradLw = -1/N * np.dot(tx.T, e)
	return gradLw


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	"""Gradient descent algorithm."""
	if len(initial_w.shape)==2:
		initial_w = initial_w.reshape((max(initial_w.shape)))
	if len(y.shape)==2:
		y = y.reshape((max(y.shape)))

	w = initial_w

	for n_iter in range(max_iters):
		gradient = compute_gradient(y, tx, w)
		w = w - gamma * gradient

	loss = compute_loss(y, tx, w)
        
	return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	"""Stochastic gradient descent algorithm."""
	if len(initial_w.shape)==2:
		initial_w = initial_w.reshape((max(initial_w.shape)))
	if len(y.shape)==2:
		y = y.reshape((max(y.shape)))

	batch_size = 5000
	w = initial_w

	for n_iter in range(max_iters):
		y_, tx_ = batch_iter(y, tx, batch_size).__next__()
		gradient = compute_gradient(y_, tx_, w)
		w = w - gamma * gradient
		if n_iter%3==0:
			gamma = gamma/1.2

	loss = compute_loss(y, tx, w)

	return w, loss


def least_squares(y, tx):
	"""calculate the least squares solution."""

	A = np.dot(tx.T, tx)
	b = np.dot(tx.T, y)

	w = np.linalg.solve(A, b)

	loss = compute_loss(y, tx, w)

	return w, loss


"""
	Ridge regression function
"""
def ridge_regression(y, tx, lambda_):
	"""implement ridge regression."""
	if len(y.shape)==2:
		y = y.reshape((max(y.shape)))

	N = tx.shape[0]
	M = tx.shape[1]
	
	A = np.dot(tx.T, tx) + 2*N*lambda_*np.identity(M)
	b = np.dot(tx.T, y)

	w = np.linalg.solve(A, b)
	loss = compute_loss(y, tx, w)

	return w, loss


"""
	Utilities for logistic regression
"""
def sigmoid(t):
	"""apply sigmoid function on t."""
	precLim = 10
	
	t[t<=-precLim] = 0
	t[t>-precLim] = 1/ (1 + np.exp(-t))

	return t

def calculate_gradient(y, tx, w):
	"""compute the gradient of loss."""

	ret = tx.T.dot(sigmoid(np.dot(tx, w)) - y)
	return ret


""" 
	Logistic regression
"""

def learning_by_gradient_descent(y, tx, w, gamma):
	"""
	Do one step of gradient descent using logistic regression.
	Return the loss and the updated w.
	"""
	grad = calculate_gradient(y, tx, w)

	w = w - gamma * grad
	return w

def logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma):
	"""
		Approximate the weights by doing @max_iters iterations 
		of gradient descent learning
	"""
	w = initial_w

	for iter in range(max_iters):
		w = learning_by_gradient_descent(y, tx, w, gamma)

	return w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
	"""
	Do the logistic regression using gradient descent 
	return w, loss
	"""
	if len(initial_w.shape)==2:
		initial_w = initial_w.reshape((max(initial_w.shape)))
	if len(y.shape)==2:
		y = y.reshape((max(y.shape)))

	w = logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma)
	
	loss = calculate_nll(y, tx, w)

	return w, loss


"""
	Regularized logistic regression
"""

def penalized_logistic_regression(y, tx, w, lambda_):
	"""return the gradient, and hessian."""
	gradient = calculate_gradient(y, tx, w) + 2.0 * lambda_ * w
	H = calculate_hessian(y, tx, w) + 2.0 * lambda_
	
	return gradient, H

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
	"""
	Do one step of gradient descent, using the penalized logistic regression.
	Return the loss and updated w.
	"""
	grad, H = penalized_logistic_regression(y, tx, w, lambda_)

	hgrad = np.linalg.inv(H).dot(grad)

	w = w - gamma * hgrad

	return w

def reg_logisitic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	if len(initial_w.shape)==2:
		initial_w = initial_w.reshape((max(initial_w.shape)))
	if len(y.shape)==2:
		y = y.reshape((max(y.shape)))

	w = initial_w
	
	for iter in range(max_iters):
		w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

	loss = calculate_nll(y, tx, w) + lambda_ * w.T.dot(w)

	return w, loss

