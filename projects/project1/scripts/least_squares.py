# -*- coding: utf-8 -*-
import numpy as np
from helpers import batch_iter
from costs import compute_loss, calculate_nll

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
#	loss = calculate_nll(y, tx, w)
        
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
#	loss = calculate_nll(y, tx, w)

	return w, loss


def least_squares(y, tx):
	"""calculate the least squares solution."""
	#w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)

	A = np.dot(tx.T, tx)
	b = np.dot(tx.T, y)

	w = np.linalg.solve(A, b)

	loss = compute_loss(y, tx, w)
#	loss = calculate_nll(y, tx, w)

	return w, loss

