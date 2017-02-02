###############################################################
#                        TEST THEANO
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
import os
from data_load import load_SPAM
import time
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X = T.matrix('x')
Y = T.ivector('y')


def castData(data):
    return T.cast(data,dtype=theano.config.floatX)#theano.shared(floatX(data),borrow=True)

def floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

# RANDOM INIT
def init_weights(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)

def init_zeros(x,y,name):
    return theano.shared(floatX(np.zeros((x,y))),name=name,borrow=True)


######################################################################
# MODEL AND OPTIMIZER
######################################################################
# Recurrent Neural Network class

class SoftmaxLayer:
    def __init__(self,input_size,vocab_size):
        self.x = input_size
        self.y = vocab_size
        self.w = init_weights(input_size,vocab_size,'w')
        self.b = init_zeros(1,vocab_size,'b')
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
        # Used in Adagrad calculation
        self.mw = init_zeros(input_size,vocab_size,'mw')
        self.mb = init_zeros(1,vocab_size,'mb')
        self.memory_params = [self.mw,self.mb]

     # Expects saved output from last LSTM layer
    def forward_prop(self,F):
        pyx = (T.dot(F,self.w) + T.tile(self.b,(F.shape[0],1)))#+ self.b)
        pred = T.nnet.softmax(pyx).ravel()
        return castData(pred)

class RegularLayer:
    def __init__(self,input_size,output_size,name):
        self.x = input_size
        self.y = output_size
        self.w = init_weights(input_size,output_size,'{}_w'.format(name))
        self.b = init_weights(1,output_size,'{}_b'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
        # Used in Adagrad calculation
        self.mw = init_zeros(input_size,output_size,'m{}_w'.format(name))
        self.mb = init_zeros(1,output_size,'m{}_b'.format(name))
        self.memory_params = [self.mw,self.mb]

     # Expects saved output from last layer
    def forward_prop(self,F):
        pyx = T.nnet.sigmoid(T.dot(F,self.w) + T.tile(self.b,(F.shape[0],1)))
        return castData(pyx)

class TestNetwork:
    def __init__(self):
        # Hidden layers
        self.hidden_layer = RegularLayer(57,5000,'h')
        self.hidden_layer2 = RegularLayer(5000,2000,'h')
        self.hidden_layer3 = RegularLayer(2000,1000,'h')
        # Output Layer
        #   Just a standard softmax layer.
        self.output_layer = SoftmaxLayer(1000,100)

        # Update Parameters - Backprop
        self.update_params = self.hidden_layer.update_params + \
                             self.hidden_layer2.update_params + \
                             self.hidden_layer3.update_params + \
                             self.output_layer.update_params
        # Memory Parameters for Adagrad
        self.memory_params = self.hidden_layer.memory_params + \
                             self.hidden_layer2.memory_params + \
                             self.hidden_layer3.memory_params + \
                             self.output_layer.memory_params

    # Our cost function
    def calc_cost(self,X,Y):
        H = self.hidden_layer.forward_prop(X)
        H2 = self.hidden_layer2.forward_prop(H)
        H3 = self.hidden_layer3.forward_prop(H2)
        pred = self.output_layer.forward_prop(H3)
        cost = -T.log(pred[Y])
        return  castData(cost),pred

    # RMSprop is for NERDS
    def Adagrad(self,cost, params, mem, lr=0.01):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p,g,m in zip(params, grads, mem):
            g = T.clip(g,-5.,5)
            new_m = castData(m + (g * g))
            # Here's where the update list mentioned in
            # init comes into play.
            updates.append((m,new_m))
            new_p = castData( p - ((lr * g) / T.sqrt(new_m + 1e-8)) )
            updates.append((p, new_p))
        return updates


dir = "data/spam"
# "Hey, idiot, why aren't you using a test set?"
# "Well, friend, this isn't a real problem.  I'm just setting this up to test CPU vs GPU speed.
#   I don't care if it works or not, just that it runs."
X_train, y_train = load_SPAM(dir)

X_train = X_train.astype(np.float32)[:25]
y_train = y_train.astype(np.int32)[:25]

######################################################################
# FUNCTIONS AND VARIABLES
######################################################################
# Create our class
tn = TestNetwork()

params = tn.update_params
memory_params = tn.memory_params


outputs_info=[None,None]
#scan_costs,y_preds = theano.scan(fn=tn.calc_cost,
#                              outputs_info=outputs_info,
#                              sequences=[X,Y]
#                            )[0] # only need the results, not the updates

scan_costs,y_preds = tn.calc_cost(X,Y)

scan_cost = T.sum(scan_costs)

updates = tn.Adagrad(scan_cost,params,memory_params)
back_prop = theano.function(inputs=[X,Y], outputs=[scan_cost,y_preds], updates=updates, allow_input_downcast=True)

print("Model initialized, beginning training")


smooth_loss = 10.
n = 0
p = 0
while True:

    start = time.clock()
    loss,preds = back_prop(X_train,y_train)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    end = time.clock()
    #if not n % 100:
        #predictTest()
    print("Completed iteration:",n,"Cost: ",smooth_loss, "Duration: ",end-start)

    n += 1

print("Training complete")
