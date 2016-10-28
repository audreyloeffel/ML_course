# -*- coding: utf-8 -*-
import numpy as np
from helpers import batch_iter
from costs import compute_loss, calculate_nll

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
	w = initial_w

	for n_iter in range(max_iters):
		gradient = compute_gradient(y, tx, w)
		w = w - gamma * gradient

	loss = compute_loss(y, tx, w)
        
	return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	"""Stochastic gradient descent algorithm."""
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
	w = initial_w
	
	for iter in range(max_iters):
		w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
		loss = calculate_nll(y, tx, w)
		if iter%100==0:
			print("Iter", iter, "loss =", loss)

	loss = calculate_nll(y, tx, w) + lambda_ * w.T.dot(w)

	return w, loss

