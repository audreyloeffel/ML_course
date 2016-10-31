# -*- coding: utf-8 -*-
import numpy as np
from costs import calculate_nll, compute_loss

"""
	Ridge regression function
"""

def ridge_regression(y, tx, lambda_):
	"""implement ridge regression."""
	N = tx.shape[0]
	M = tx.shape[1]
	#w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx) + 2*N*lambda_*np.identity(M)), tx.T), y)
	
	A = np.dot(tx.T, tx) + 2*N*lambda_*np.identity(M)
	b = np.dot(tx.T, y)

	w = np.linalg.solve(A, b)
	loss = compute_loss(y, tx, w)
#	loss = calculate_nll(y, tx, w)

	return w, loss

"""
	Utilities for logistic regression
"""

"""
def apply_(v):
	if(v > 0) :
		return 1/ (1 + np.exp(-v))
	else:
		return np.exp(v) / (1 + np.exp(v))

sigmoid = np.vectorize(apply_)
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

def calculate_hessian(y, tx, w):
	"""return the hessian of the loss function."""	
	N = y.shape[0]
	xw = tx.dot(w)

	S = (sigmoid(xw) * (1-sigmoid(xw))).reshape((N, 1)) * tx

	return np.dot(tx.T, S)


"""
	Logistic regression learning phase
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
	w = initial_w

	for iter in range(max_iters):
		nw = learning_by_gradient_descent(y, tx, w, gamma)
		diff = np.linalg.norm(w - nw)
		if iter%499==0:
			print(diff)
		w = nw

	return w


def default_logistic_regression(y, tx, w):
	"""
	Return the gradient, and hessian for normal logistic regression.
	"""
	gradient = calculate_gradient(y, tx, w)
	H = calculate_hessian(y, tx, w)
	return gradient, H

def learning_by_newton_method(y, tx, w, gamma):
	"""
	Do one step on Newton's method.
	return the loss and updated w.
	"""

	grad, H = default_logistic_regression(y, tx, w)
	
	hgrad = np.linalg.inv(H).dot(grad)
	w = w - gamma * hgrad
	return w

def logistic_regression_newton_method(y, tx, initial_w, max_iters, gamma):
	w = initial_w

	for iter in range(max_iters):
		nw = learning_by_newton_method(y, tx, w, gamma)
		w = nw
	
	return w

""" 
	Logistic regression
"""

def logistic_regression(y, tx, initial_w, max_iters, gamma):
	"""
	Do the logistic regression using gradient descent 
	or Newton's technique, return loss, w
	"""

#	w = logistic_regression_newton_method(y, tx, initial_w, max_iters, gamma)
	w = logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma)
	
	loss = calculate_nll(y, tx, w)

	return w, loss

"""
	Penalized logistic regression learning
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

"""
	Penalized logistic regression
"""

def reg_logisitic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	w = initial_w
	
	for iter in range(max_iters):
		w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
		loss = calculate_nll(y, tx, w)
		if iter%100==0:
			print("Iter", iter, "loss =", loss)

	loss = calculate_nll(y, tx, w) + lambda_ * w.T.dot(w)

	return w, loss




