import six.moves.cPickle as pickle
import numpy as np
import os

'''
This code was copied from the data_utils.py file written by Stanford's CS231n course staff.
'''


def load_CIFAR_file(filename):
	'''Load a single file of CIFAR'''
	with open(filename, 'rb') as f:
		datadict= pickle.load(f)#,encoding='latin1')
		X = datadict['data']
		Y = datadict['labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float32')
		Y = np.array(Y).astype('int32')
		return X, Y


def load_CIFAR10(directory):
	'''Load all of CIFAR'''
	xs = []
	ys = []
	for k in range(1,6):
		f = os.path.join(directory, "data_batch_%d" % k)
		X, Y = load_CIFAR_file(f)
		xs.append(X)
		ys.append(Y)
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	Xte, Yte = load_CIFAR_file(os.path.join(directory, 'test_batch'))
	return Xtr, Ytr, Xte, Yte
