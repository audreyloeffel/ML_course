# -*- coding: utf-8 -*-
""" File used for the cross validation """

import numpy as np
import matplotlib.pyplot as plt
from costs import *
from least_squares import *
from regression import *


def build_k_indices(y, k_fold, seed):
	"""build k indices for k-fold."""
	num_row = y.shape[0]
	interval = int(num_row / k_fold)
	np.random.seed(seed)
	indices = np.random.permutation(num_row)
	k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
	return np.array(k_indices)


def cross_validation(y, x, k_fold, k, lambda_, initial_w, gamma, method):
	"""return the loss of ridge regression."""
	seed = 56
	k_indices = build_k_indices(y, k_fold, seed)
	train_x = []
	train_y = []
	for i in range (k_fold):
		if i != k:
			train_x.append(x[k_indices[i]])
			train_y.append(y[k_indices[i]])

	train_tx = np.asarray([item for sublist in train_x for item in sublist])
	train_y = np.asarray([item for sublist in train_y for item in sublist])

	test_tx = x[k_indices[k]]
	test_y = y[k_indices[k]]

	if method == "least_squares_GD" :
		train_w, loss_tr = least_squares_GD(train_y, train_tx, initial_w, 5, gamma)
		loss_te = compute_loss(test_y, test_tx, train_w)
		return loss_tr, loss_te
	elif method == "least_squares_SGD" :
		train_w, loss_tr = least_squares_SGD(train_y, train_tx, initial_w, max_iters, gamma)
		loss_te = loss = compute_loss(test_y, test_tx, train_w)
		return loss_tr, loss_te
	elif method == "least_squares" :
		train_w, loss_tr = least_squares(train_y, train_tx)
		loss = compute_loss(test_y, test_tx, train_w)
		return loss_tr, loss_te
	elif method == "ridge_regression" :
		train_w, loss_tr = ridge_regression(train_y, train_tx, lambda_)
		loss_te = compute_loss(test_y, test_tx, train_w)
		return loss_tr, loss_te
    
	#TODO : add logistic regressions here  
    
	# Change here depending on what technique you want to cross validate
#	train_w, loss_tr = ridge_regression(train_y, train_tx, lambda_)
	train_w, loss_tr = reg_logisitic_regression(train_y, train_tx, lambda_, initial_w, 200, gamma)

	# The loss either MSE or NLL depending on the technique
#	loss_te = compute_loss(test_y, test_tx, train_w)
	loss_te = calculate_nll(test_y, test_tx, train_w)

	return loss_tr, loss_te


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


def cross_validation_demo(y, x, initial_w, gamma, k_fold=2):
	seed = 56
	# Cross validation on different lambdas, can also change for the gammas if wanted
	lambdas = np.logspace(-4, 0, 30)
	# split data in k fold
	k_indices = build_k_indices(y, k_fold, seed)
	# define lists to store the loss of training data and test data
	losses_tr = []
	losses_te = []
	i = 0
	n = len(lambdas)
	for lamb in lambdas:
		i = i+1
		print("Step", i, "over", n)
		loss_train = []
		loss_test = []

		for k in range (k_fold):
			loss_tr, loss_te = cross_validation(y, x, k_fold, k, lamb, initial_w, gamma, "ridge_regression")
			loss_train.append(loss_tr)
			loss_test.append(loss_te)

		# rmse of the mean mse
		losses_tr.append(np.sum(loss_train, axis=0)/k_fold)
		losses_te.append(np.sum(loss_test, axis=0)/k_fold)

	print(len(losses_tr))
	print(len(losses_te))
	print(len(lambdas))

	# mean of the rmse
	#rmse_tr.append(np.sum(np.sqrt(2 * loss_train), axis=0)/k_fold)
	#rmse_te.append(np.sum(np.sqrt(2 * loss_test), axis=0)/k_fold)

	cross_validation_visualization(lambdas, losses_tr, losses_te)


