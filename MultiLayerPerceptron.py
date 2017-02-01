import random
import numpy as np
from data_load import load_SPAM

from lasagne import layers
from lasagne.updates import adagrad
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet

dir = "data/spam"
# "Hey, idiot, why aren't you using a test set?"
# "Well, friend, this isn't a real problem.  I'm just setting this up to test CPU vs GPU speed.
#   I don't care if it works or not, just that it runs."
X_train, y_train = load_SPAM(dir)

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)

net1 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('hidden2', layers.DenseLayer),
		('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
		#layers parameters:
		input_shape = (4601, 57),
		hidden_num_units = 5000,
		hidden2_num_units = 2000,
		hidden3_num_units = 1000,
		output_nonlinearity = softmax,
		output_num_units = 100,

		#optimization parameters:
		update = adagrad,
		update_learning_rate = 0.01,

		regression = False,
		max_epochs = 20,
		verbose = 1,
		)

net1.fit(X_train, y_train)

y_pred1 = net1.predict(X_train)
print("The accuracy of this network is: %0.2f" % (y_pred1 == y_train).mean())
