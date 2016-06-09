from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import random as rnd

import cPickle

import os
from scipy.io import loadmat


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
    
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    
    return tanh(dot(W.T, y)+b)


# Part 7
def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = tanh_layer(L0, W1, b1)
    output = softmax(L1)
    return L0, L1, output


    
def cost(y, y_):
    
    return -sum(y_*log(y))

        
# Part 7
def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network
    L0: hidden layer
    L1: The last layer
    y: the probability (or the prediction)
    y_: the one-hot-encoding array'''
    
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, ((1.0 - L1**2)*dCdL1).T)
    dCdb1 = (1.0 - L1**2)*dCdL1
    
    dCdL0 = dot(W1, ((1.0 - L1**2)*dCdL1))
    dCdW0 = dot(x, ((1.0 - L0**2)*dCdL0).T)
    dCdb0 = (1.0 - L0**2)*dCdL0
    
    return dCdW0, dCdb0, dCdW1, dCdb1

    
    

def compact_data(data_dict):
    ''' This function puts all data into one single list for use. '''
    
    train_list = []
    test_list = []
    
    for i in range(10): # Total of 10 digits
        
        temp_train = []
        temp_test = []
        
        train_key = "train{0}".format(i)
        test_key = "test{0}".format(i)
        
        for j in range(len(data_dict[train_key])):
            
            x = data_dict[train_key][j] / 255.0
            x = vstack((x, ))
            x = x.T
            
            temp_train.append(x)
            
        for k in range(len(data_dict[test_key])):
            
            x = data_dict[train_key][j] / 255.0
            x = vstack((x, ))
            x = x.T
            
            temp_test.append(x)
            
        train_list.append(temp_train)
        test_list.append(temp_test)
            
    return train_list, test_list
        
        
        

# Part 2
def linear_network(x, W, b):
    '''This computes the network specified in Project handout Part 2.
    x is a normalized and flattened digit data from MNIST with size of 1xN. 
    w is a NxM matrix, the weight of the network.
    b is the bias of 1xM.
    This outputs the softmax of o_i with o_i is the linear combination of x.'''
    
    o = np.dot(W.T, x) + b
    
    return softmax(o)



# Part 3
def gradient_ln(x, y, y_):
    '''This computes the gradient of linear network in part 2.'''
    
    dCdo = y - y_
    dCdW = dot(x, dCdo.T)
    dCdb = y - y_
    
    return dCdW, dCdb

                
                
# Part 5
def minibatch_grad_des(dC, network, y_list, init_b, init_W, train_data, test_data, alpha):
    ''' Computes mini batch gradient descent.
    C is the cost function to be minimized.
    dC is the gradient of C.
    alpha is the learning rate.
    y_list is the list of all arrays of one-hot-encoding for each digit.'''
    
    rnd.seed(827)
    
    err = 1e-5
    prev_W = init_W - 10*err
    prev_b = init_b - 10*err
    W = init_W.copy()
    b = init_b.copy()
    
    iteration = 1
    digits = list(range(10))
    success = []
    test_success = []
    train_success = []
    test_neg_prob = []
    train_neg_prob = []
    
    while norm(W - prev_W) > err:

        i = rnd.sample(digits, 1)[0]
        prev_cost = 0
        # A small batch of 50
        batch = rnd.sample(train_data[i], 50)
        

        # Calculate the gradient of the batch
        grad_W = zeros(init_W.shape)
        grad_b = zeros(init_b.shape)

        for x in batch:

            dW, db = dC(x, network(x, W, b), y_list[i])
            grad_W = grad_W + dW
            grad_b = grad_b + db

        prev_W = W.copy()
        prev_b = b.copy()
        
        W = W - alpha*(1/50.0)*grad_W
        b = b - alpha*(1/50.0)*grad_b
        
        # Calculate Successful rate
        test_rate = classification_rate(test_data, network, W, b, 10000.0)
        train_rate = classification_rate(train_data, network, W, b, 60000.0)
        
                          
        print "Iteration {0}:  {1}%".format(iteration, test_rate)
        test_success.append(test_rate)
        train_success.append(train_rate)
        
        # Calculate the total cost every 50 iterations
        if (iteration % 50) == 0:
            
            test_cost = total_cost(test_data, network, W, b, y_list, cost)
            train_cost = total_cost(train_data, network, W, b, y_list, cost)
            
            train_neg_prob.append(train_cost)
            test_neg_prob.append(test_cost)
        
        # Early stop at 700 iterations
        if iteration == 700:
            
            break
        
        iteration = iteration + 1
        
    return W, b, iteration, train_success, test_success, train_neg_prob, test_neg_prob
    

# Part 6
def minibatch_modified(dC, network, y_list, init_b0, init_W0, \
                       init_b1, init_W1, train_data, test_data, alpha):
    ''' Computes mini batch gradient descent with one hidden layer.
    C is the cost function to be minimized.
    dC is the gradient of C.
    alpha is the learning rate.
    y_list is the list of all arrays of one-hot-encoding for each digit.'''
    
    rnd.seed(287)
    
    err = 1e-6
    prev_W0 = init_W0 - 10*err
    prev_b0 = init_b0 - 10*err
    W0 = init_W0.copy()
    b0 = init_b0.copy()
    
    prev_W1 = init_W1 - 10*err
    prev_b1 = init_b1 - 10*err
    W1 = init_W1.copy()
    b1 = init_b1.copy()
    
    iteration = 1
    digits = list(range(10))
    success = []
    test_success = []
    train_success = []
    test_neg_prob = []
    train_neg_prob = []
    
    while norm(W0 - prev_W0) > err:

        i = rnd.sample(digits, 1)[0]
        prev_cost = 0
        # A small batch of 50
        batch = rnd.sample(train_data[i], 50)
        

        # Calculate the gradient of the batch
        grad_W0 = zeros(init_W0.shape)
        grad_b0 = zeros(init_b0.shape)
        grad_W1 = zeros(init_W1.shape)
        grad_b1 = zeros(init_b1.shape)

        for x in batch:

            h, o, y = network(x, W0, b0, W1, b1)
            dW0, db0, dW1, db1 = dC(W0, b0, W1, b1, x, h, o, y, y_list[i])
            grad_W0 = grad_W0 + dW0
            grad_b0 = grad_b0 + db0
            grad_W1 = grad_W1 + dW1
            grad_b1 = grad_b1 + db1

        prev_W0 = W0.copy()
        prev_b0 = b0.copy()
        prev_W1 = W1.copy()
        prev_b1 = b1.copy()
        
        W0 = W0 - alpha*(1/50.0)*grad_W0
        b0 = b0 - alpha*(1/50.0)*grad_b0
        W1 = W1 - alpha*(1/50.0)*grad_W1
        b1 = b1 - alpha*(1/50.0)*grad_b1
        
        

        print "Iteration {0}".format(iteration)
        
        # Calculate the total cost every 50 iterations
        if (iteration % 100) == 0:
            
            test_cost = total_cost(test_data, network, (W0, W1), (b0, b1), y_list, cost)
            train_cost = total_cost(train_data, network, (W0, W1), (b0, b1), y_list, cost)
            
            # Calculate Successful rate
            test_rate = classification_rate(test_data, network, (W0, W1), (b0, b1), 10000.0)
            train_rate = classification_rate(train_data, network, (W0, W1), (b0, b1), 60000.0)

            print "Iteration {0}:  {1}%".format(iteration, test_rate)
            test_success.append(test_rate)
            train_success.append(train_rate)
            
            train_neg_prob.append(train_cost)
            test_neg_prob.append(test_cost)
            
            print test_cost
        
        if iteration == 25000:
            
            break
        
        iteration = iteration + 1
        
    return W0, W1, b0, b1, iteration, train_success, test_success, train_neg_prob, test_neg_prob
    
    

def total_cost(data, network, W, b, y_list, cost_fn):
    """ This function calculate the total cost of the given data."""
    
    s = 0
    
    if network == forward:
        
        W0, W1 = W
        b0, b1 = b
        
        for i in range(len(data)):

            for j in range(len(data[i])):
                
                h, o, y = network(data[i][j], W0, b0, W1, b1)
                s = s + cost_fn(y, y_list[i])
                
        return s
        
        
    
    else:
        for i in range(len(data)):

            for j in range(len(data[i])):

                s = s + cost_fn(network(data[i][j], W, b), y_list[i])
                
        return s
            
    return s


def classification_rate(data, network, W, b, n):
    """ Calculate the successful rate of a given data."""
    
    correct = 0.0
    
    if network == forward:
        
        W0, W1 = W
        b0, b1 = b
            
        for i in range(len(data)):

            for j in range(len(data[i])):

                h, o, p = network(data[i][j], W0, b0, W1, b1)
                ans = argmax(p)

                if ans == i:

                    correct = correct + 1
                    
        return (correct / n) * 100.0

        
    else:

        for i in range(len(data)):

            for j in range(len(data[i])):

                p = network(data[i][j], W, b)
                ans = argmax(p)

                if ans == i:

                    correct = correct + 1
                    
        return (correct / n) * 100.0

                    
    return (correct / n) * 100.0
    


#-------------------------------------------------------------------------#
# Inital Setup

# Load the MNIST digit data
M = loadmat("mnist_all.mat")


# Compact data into one single list
train, test = compact_data(M)


# One-Hot-Encoding list
y_list = []

for i in range(10):

    y_ = zeros((10, 1))
    y_[i] = 1.0
    y_list.append(y_)

#-------------------------------------------------------------------------#


#-------------------------------------------------------------------------#
# Part 4: Finite difference of weights to check the gradient
temp = M["train0"][100] / 255.0
temp = vstack((temp, ))
temp = temp.T

W = random.random_sample((784, 10)) * 1e-5
b = random.random_sample((10, 1)) * 1e-5

p = linear_network(temp, W, b)
label = zeros((10, 1))
label[0, 0] = 1
gradw1, gradb1 = gradient_ln(temp, p, label)


for i in range(10):
    
    h = 1e-5
    err = zeros(W.shape)
    err[600, i] = err[600, i] + h
    p1 = linear_network(temp, W+err, b)
    p2 = linear_network(temp, W-err, b)
    dC = (cost(p1, label) - cost(p2, label)) / (2.0*h)

    print "dC = {}".format(dC)
    print "gradient = {}\n".format(gradw1[600, i])



#-------------------------------------------------------------------------#

# Part 5
random.seed(385)

W = random.random_sample((784, 10)) * 1e-5
b = random.random_sample((10, 1)) * 1e-5
#b = zeros((10, 1))

print "\n-----------Part 5-----------\n"

out = minibatch_grad_des(gradient_ln, linear_network, y_list, b, W, train, test, 0.01)
ln_W, ln_b, ln_iter, ln_train_rate, ln_test_rate, ln_train_cost, ln_test_cost = out

# Get correctly classified digits and incorrectly classified ones

ln_corr_digit = []
ln_incorr_digit = []
    
for i in range(len(test)):

    for j in range(len(test[i])):
            
        p = linear_network(test[i][j], ln_W, ln_b)
        ans = argmax(p)

        if ans == i:

            ln_corr_digit.append(test[i][j])

        else:

            ln_incorr_digit.append(test[i][j])

print "\n-----------Finish-----------\n"


#-------------------------------------------------------------------------#




#-------------------------------------------------------------------------#

# Part 6
random.seed(385)

W0 = random.random_sample((784, 300)) * 1e-5
W1 = random.random_sample((300, 10)) * 1e-5
b0 = random.random_sample((300, 1)) * 1e-5
b1 = random.random_sample((10, 1)) * 1e-5

print "\n-----------Part 6-----------\n"

out = minibatch_modified(deriv_multilayer, forward, y_list, b0, W0, \
                         b1, W1, train, test, 0.01)
t_W0, t_W1, t_b0, t_b1, t_iter, t_train_rate, t_test_rate, t_train_cost, t_test_cost = out

# Get correctly classified digits and incorrectly classified ones

t_corr_digit = []
t_incorr_digit = []

    
for i in range(len(test)):

    for j in range(len(test[i])):
            
        h, o, p = forward(test[i][j], t_W0, t_b0, t_W1, t_b1)
        ans = argmax(p)

        if ans == i:

            t_corr_digit.append(test[i][j])

        else:

            t_incorr_digit.append(test[i][j])

print "\n-----------Finish-----------\n"


#-------------------------------------------------------------------------#

#-------------------------------------------------------------------------#
# Part 8: Finite difference of weights to check the gradient
temp = M["train0"][250] / 255.0
temp = vstack((temp, ))
temp = temp.T

W0 = random.random_sample((784, 300)) * 1e-5
W1 = random.random_sample((300, 10)) * 1e-5
b0 = random.random_sample((300, 1)) * 1e-5
b1 = random.random_sample((10, 1)) * 1e-5

h, o, p = forward(temp, W0, b0, W1, b1)
label = zeros((10, 1))
label[2, 0] = 1
gradw0, gradb0, gradw1, gradb1 = deriv_multilayer(W0, b0, W1, b1, temp, h, o, p, label)


for i in range(10):
    
    h = 1e-5
    
    
    err1 = zeros(W1.shape)
    
    
    err1[200, i] = err1[200, i] + h

    h1, o1, p1 = forward(temp, W0, b0, W1+err1, b1)
    h2, o2, p2 = forward(temp, W0, b0, W1-err1, b1)
    
    
    dC1 = (cost(p1, label) - cost(p2, label)) / (2.0*h)

    
    
    print "dCdW1 = {}".format(dC1)
    print "gradient = {}\n".format(gradw1[200, i])
    

for i in range(300):
    
    err0 = zeros(W0.shape)
    
    err0[400, i] = err0[400, i] + h
    
    h1, o1, p1 = forward(temp, W0+err0, b0, W1, b1)
    h2, o2, p2 = forward(temp, W0-err0, b0, W1, b1)
    
    dC0 = (cost(p1, label) - cost(p2, label)) / (2.0*h)
    
    print "dCdW0 = {}".format(dC0)
    print "gradient = {}\n".format(gradw0[400, i])

    
#-------------------------------------------------------------------------#   


#Display 10 images for each digit for Part 1
figure(1)

for i in range(100):

    digit = i / 10
    key = "train{0}".format(digit)
    plt.subplot(10, 10, i+1)
    imshow(M[key][i].reshape((28,28)), cmap=cm.gray)
    plt.axis('off')

# Part 5: Learning Curves & Visualization
figure(2)
title("Negtive Probability for training set")
plot(list(range(1, (ln_iter/50)+1)), ln_train_cost)
xlabel("50 iterations")
ylabel("neg-probability")

figure(3)
title("Negtive Probability for test set")
plot(list(range(1, (ln_iter/50)+1)), ln_test_cost)
xlabel("50 iterations")
ylabel("neg-probability")

figure(4)
title("Classification Rate for training set")
plot(list(range(1, ln_iter+1)), ln_train_rate)
xlabel("iteration")
ylabel("%")

figure(5)
title("Classification Rate for test set")
plot(list(range(1, ln_iter+1)), ln_test_rate)
xlabel("iteration")
ylabel("%")

figure(6)
title("Correctly Classified digits")
digits = rnd.sample(ln_corr_digit, 20)

for i in range(20):
    
    d = digits[i].reshape((28, 28))
    plt.subplot(2, 10, i+1)
    imshow(d, cmap=cm.gray)
    plt.axis('off')
    
figure(7)
title("Incorrectly Classified digits")
digits = rnd.sample(ln_incorr_digit, 10)

for i in range(10):
    
    d = ln_incorr_digit[i].reshape((28, 28))
    plt.subplot(1, 10, i+1)
    imshow(d, cmap=cm.gray)
    plt.axis('off')


# Part 6: Visualization of weights
figure(8)
for i in range(10):

    w = ln_W[:, i].reshape((28, 28))
    plt.subplot(1, 10, i+1)
    imshow(w)
    plt.axis('off')

# Part 9: Learning Curves & Visualization
figure(9)
title("Negtive Probability for training set")
plot(list(range(1, (t_iter/100)+1)), t_train_cost)
xlabel("100 iterations")
ylabel("neg-probability")

figure(10)
title("Negtive Probability for test set")
plot(list(range(1, (t_iter/100)+1)), t_test_cost)
xlabel("100 iterations")
ylabel("neg-probability")

figure(11)
title("Classification Rate for training set")
plot(list(range(1, (t_iter/100)+1)), t_train_rate)
xlabel("100 iterations")
ylabel("%")

figure(12)
title("Classification Rate for test set")
plot(list(range(1, (t_iter/100)+1)), t_test_rate)
xlabel("100 iterations")
ylabel("%")

figure(13)
title("Correctly Classified digits")
digits = rnd.sample(t_corr_digit, 20)

for i in range(20):
    
    d = digits[i].reshape((28, 28))
    plt.subplot(2, 10, i+1)
    imshow(d, cmap=cm.gray)
    plt.axis('off')


# Part 10: Visualization of weights
figure(14)
for i in range(10):

    w = t_W0[:, i].reshape((28, 28))
    plt.subplot(1, 10, i+1)
    imshow(w)
    plt.axis('off')



show()


    
    