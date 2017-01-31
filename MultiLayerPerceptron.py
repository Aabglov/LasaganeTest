import random
import numpy as np
from data_load import load_CIFAR10

from setup_GPU import setup_theano

setup_theano()

from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet

cifar10_dir = "data/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

X_train_flat = np.dot(X_train[...,:3], [0.299, 0.587, 0.114]).reshape(X_train.shape[0],-1).astype(np.float32)
X_test_flat = np.dot(X_test[...,:3], [0.299, 0.587, 0.114]).reshape(X_test.shape[0],-1).astype(np.float32)
X_train_flat = (X_train_flat/255.0)-0.5
X_test_flat = (X_test_flat/255.0)-0.5


net1 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		#layers parameters:
		input_shape = (None, 1024),
		hidden_num_units = 100,
		output_nonlinearity = softmax,
		output_num_units = 10,

		#optimization parameters:
		update = nesterov_momentum,
		update_learning_rate = 0.01,
		update_momentum = 0.9,

		regression = False,
		max_epochs = 20,
		verbose = 1,
		)

net1.fit(X_train, y_train)

y_pred1 = net1.predict(X_test_flat)
print "The accuracy of this network is: %0.2f" % (y_pred1 == y_test).mean()

