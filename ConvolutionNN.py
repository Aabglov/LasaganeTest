import random
import numpy as np
from data_load import load_CIFAR10

from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet

cifar10_dir = "data/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

X_train_2d = np.dot(X_train[...,:3], [0.299, 0.587, 0.114]).reshape(-1,1,32,32).astype(np.float32)
X_test_2d = np.dot(X_test[...,:3], [0.299, 0.587, 0.114]).reshape(-1,1,32,32).astype(np.float32)

X_train_2d = (X_train_2d /255.0)-0.5
X_test_2d = (X_test_2d/255.0)-0.5

net2 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		("hidden4", layers.DenseLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, 1, 32, 32),
		conv1_num_filters = 16, conv1_filter_size = (3, 3), pool1_pool_size = (2,2),
		conv2_num_filters = 32, conv2_filter_size = (2, 2) , pool2_pool_size =  (2,2),
		conv3_num_filters = 64, conv3_filter_size = (2, 2), pool3_pool_size = (2,2),
		hidden4_num_units = 200,
		output_nonlinearity = softmax,
		output_num_units = 10,

		#optimization parameters:
		update = nesterov_momentum,
		update_learning_rate = 0.005,
		update_momentum = 0.9,
		regression = False,
		max_epochs = 5,
		verbose = 1,
		)

net2.fit(X_train_2d, y_train)

import six.moves.cPickle as pickle
with open('results/net2.pkl', 'wb+') as f:
    pickle.dump(net2, f, -1)

y_pred2 = net2.predict(X_test_2d)
print("The accuracy of this network is: %0.2f" % (y_pred2 == y_test).mean())
