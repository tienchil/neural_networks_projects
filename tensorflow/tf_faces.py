from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

from myalexnet import main_part_2
import tensorflow as tf

# Get a random seed
seed = 1454263
random.seed(seed)

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    if len(rgb.shape) < 3:
        
        return rgb
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.0


def get_train_batch(data, N):

    n = N / 6

    batch_xs = zeros((1, 16384))
    batch_y_s = zeros((1, 6))
    
    for k in range(6):

        train_size = len(data[k])
        idx = random.permutation(train_size)
        
        for j in range(n):

            batch_xs = vstack((batch_xs, data[k][idx[j]]))
            one_hot = zeros((1, 6))
            one_hot[:, k] = 1
            batch_y_s = vstack((batch_y_s, one_hot))

    return batch_xs, batch_y_s

def get_train(data):

    batch_xs = zeros((1, 16384))
    batch_y_s = zeros((1, 6))
    
    for k in range(6):

        train_size = len(data[k])

        for j in range(train_size):

            batch_xs = vstack((batch_xs, data[k][j]))
            one_hot = zeros((1, 6))
            one_hot[:, k] = 1
            batch_y_s = vstack((batch_y_s, one_hot))

    return batch_xs, batch_y_s

def get_test(data):

    batch_xs = zeros((1, 16384))
    batch_y_s = zeros((1, 6))
    
    for k in range(6):

        test_size = len(data[k])

        for j in range(test_size):

            batch_xs = vstack((batch_xs, data[k][j]))
            one_hot = zeros((1, 6))
            one_hot[:, k] = 1
            batch_y_s = vstack((batch_y_s, one_hot))

    return batch_xs, batch_y_s

def get_train_batch_part_2(data, N):

    n = N / 6

    batch_xs = zeros((1, 227, 227, 3))
    batch_y_s = zeros((1, 6))
    
    for k in range(6):

        train_size = len(data[k])
        idx = random.permutation(train_size)
        
        for j in range(n):

            batch_xs = vstack((batch_xs, data[k][idx[j]]))
            one_hot = zeros((1, 6))
            one_hot[:, k] = 1
            batch_y_s = vstack((batch_y_s, one_hot))

    return batch_xs, batch_y_s


def get_train_part_2(data, N):

    batch_xs = zeros((1, 227, 227, 3))
    batch_y_s = zeros((1, 6))
    
    for k in range(6):

        train_size = N
        idx = random.permutation(train_size)

        for j in range(train_size):

            batch_xs = vstack((batch_xs, data[k][idx[j]]))
            one_hot = zeros((1, 6))
            one_hot[:, k] = 1
            batch_y_s = vstack((batch_y_s, one_hot))

    return batch_xs, batch_y_s

def get_test_part_2(data):

    batch_xs = zeros((1, 227, 227, 3))
    batch_y_s = zeros((1, 6))
    
    for k in range(6):

        test_size = len(data[k])

        for j in range(test_size):

            batch_xs = vstack((batch_xs, data[k][j]))
            one_hot = zeros((1, 6))
            one_hot[:, k] = 1
            batch_y_s = vstack((batch_y_s, one_hot))

    return batch_xs, batch_y_s



# Extracting images
act = ["butler", "radcliffe", "vartan", "bracco", "gilpin", "harmon"]
actor = ["butler", "radcliffe", "vartan"]
actress = ["bracco", "gilpin", "harmon"]



all_data = []
y_data = []

# Please add the following directory if not exists
male_files = os.listdir("./male/")
female_files = os.listdir("./female/")

print "Processing Data..."

i = 0

# Get images of male faces
for name in actor:

    faces = [face for face in male_files if face[:len(name)] == name]
    temp = []

    for filename in faces:

        img = imread("./male/" + filename) 
        img = imresize(img, (64, 64))

        temp.append(img.flatten())

    y = zeros((1, 6))
    y[:, i] = 1
    y_data.append(y)

    all_data.append(temp)

    i = i + 1


# Get images of female faces
for name in actress:

    faces = [face for face in female_files if face[:len(name)] == name]
    temp = []

    for filename in faces:

        img = imread("./female/" + filename) 
        img = imresize(img, (64, 64))

        temp.append(img.flatten() / 255.0)

    y = zeros((1, 6))
    y[:, i] = 1
    y_data.append(y)

    all_data.append(temp)

    i = i + 1

print "Done\n"



# Divide images into training set, test set, and validation set
train = []
test = []
validation = []


for i in range(len(all_data)):

    N = len(all_data[i])
    face = all_data[i]
    one_hot = y_data[i]

    train_temp = []
    test_temp = []
    valid_temp = []

    # Get indices randomly
    index = random.permutation(len(face))

    print "Getting training set..."

    for j in range(70):

        train_temp.append(face[index[j]])

    print "Done.\n"

    index = index[:70]

    M = len(index) / 2

    print "Getting test and validation set..."

    for j in range(M):

        test_temp.append(face[index[j]])
        valid_temp.append(face[index[M+j]])

    print "Done\n"

    train.append(train_temp)
    test.append(test_temp)
    validation.append(valid_temp)



# Set up the neural network with one hidden layer
x = tf.placeholder(tf.float32, [None, 16384])


# nhid = 300
nhid = 800
W0 = tf.Variable(tf.random_normal([16384, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])



# Perform Gradient Descent
lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.GradientDescentOptimizer(0.000005).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Arrays for learning curves
iterations = 5000
g_test = zeros(iterations / 100)
g_train = zeros(iterations / 100)
g_valid = zeros(iterations / 100)

test_x, test_y = get_test(test)
valid_x, valid_y = get_test(validation)
train_xs, train_ys = get_train(train)


print "Strat training...\n"

for i in range(iterations):

  batch_xs, batch_ys = get_train_batch(train, 48)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  
  if i % 100 == 0:

    test_rate = sess.run(accuracy, feed_dict={x: test_x, y_: test_y}) * 100
    valid_rate = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y}) * 100
    train_rate = sess.run(accuracy, feed_dict={x: train_xs, y_: train_ys}) * 100

    g_test[i/100] = test_rate
    g_train[i/100] = train_rate
    g_valid[i/100] = valid_rate

    print "iteration ", i
    print "Test:", test_rate, "%"
    print "validation:", valid_rate, "%"
    print "Train:", train_rate, "%"

    print "-----------------------------------"

test_x, test_y = get_test(test)
test_rate = sess.run(accuracy, feed_dict={x: test_x, y_: test_y}) * 100
print "Final classification rate"
print "test = ", test_rate, "%"


try:
    np.save('test_rate', g_test)
    np.save('train_rate', g_train)
    np.save('valid_rate', g_valid)
    weight0 = sess.run(W0)
    weight1 = sess.run(W1)
    bias0 = sess.run(b0)
    bias1 = sess.run(b1)

    # np.save('300_W0', weight0)
    # np.save('300_W1', weight1)
    # np.save('300_b0', bias0)
    # np.save('300_b1', bias1)

    np.save('800_W0', weight0)
    np.save('800_W1', weight1)
    np.save('800_b0', bias0)
    np.save('800_b1', bias1)


except:

    print "Nothing has been saved"

print "Part 1 End"

#Part 2

print "Part 2 starts..."

# Getting inputs
i = 0

data_part_2 = []
y_part_2 = []

print "Processing data..."
# Get images of male faces
for name in actor:

    faces = [face for face in male_files if face[:len(name)] == name]
    temp = []

    for filename in faces:

        img = imread("./male/" + filename) 
        img = imresize(img, (227, 227))
        img = img[:, :, :3].reshape((1, 227, 227, 3))/255.0

        temp.append(img)

    y = zeros((1, 6))
    y[:, i] = 1
    y_part_2.append(y)

    data_part_2.append(temp)

    print ((i+1) / 6.0)*100, "%"

    i = i + 1


# Get images of female faces
for name in actress:

    faces = [face for face in female_files if face[:len(name)] == name]
    temp = []

    for filename in faces:

        img = imread("./female/" + filename) 
        img = imresize(img, (227, 227))
        img = img[:, :, :3].reshape((1, 227, 227, 3))/255.0

        temp.append(img)

    y = zeros((1, 6))
    y[:, i] = 1
    y_part_2.append(y)

    data_part_2.append(temp)

    print ((i+1) / 6.0)*100, "%"

    i = i + 1

print "Done\n"

train_part_2 = []
test_part_2 = []
validation_part_2 = []


for i in range(len(data_part_2)):

    N = len(data_part_2[i])
    face = data_part_2[i]
    one_hot = y_part_2[i]

    train_temp = []
    test_temp = []
    valid_temp = []

    # Get indices randomly
    index = random.permutation(len(face))

    print "Getting training set..."

    for j in range(70):

        train_temp.append(face[index[j]])

    print "Done.\n"

    index = index[:70]

    M = len(index) / 2

    print "Getting test and validation set..."

    for j in range(M):

        test_temp.append(face[index[j]])
        valid_temp.append(face[index[M+j]])

    print "Done\n"

    train_part_2.append(train_temp)
    test_part_2.append(test_temp)
    validation_part_2.append(valid_temp)

# Get training set, test set, and validation set

print "Start Training..."

# Set up the neural network with one hidden layer
x = tf.placeholder(tf.float32, [None, 13*13*384])


nhid = 300
# nhid = 800
W0 = tf.Variable(tf.random_normal([13*13*384, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.GradientDescentOptimizer(0.000005).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

iterations = 5000

test_x, test_y = get_test_part_2(test_part_2)
valid_x, valid_y = get_test_part_2(validation_part_2)
train_xs, train_ys = get_train_part_2(train_part_2, 70)

for i in range(iterations):

  batch_xs, batch_ys = get_train_batch_part_2(train_part_2, 48)

  features = zeros((49, 13*13*384))
  out = main_part_2(batch_xs, 49)

  for j in range(49):

    features[j, :] = out[j, :, :, :].flatten()

  sess.run(train_step, feed_dict={x: features, y_: batch_ys})



  if i % 10 == 0:

    M = shape(test_x)[0]
    features = zeros((M, 13*13*384))
    out = main_part_2(test_x, M)

    for j in range(M):

       features[j, :] = out[j, :, :, :].flatten()

    test_rate = sess.run(accuracy, feed_dict={x: features, y_: test_y}) * 100

    M = shape(valid_x)[0]
    features = zeros((M, 13*13*384))
    out = main_part_2(valid_x, M)

    for j in range(M):

       features[j, :] = out[j, :, :, :].flatten()

    valid_rate = sess.run(accuracy, feed_dict={x: features, y_: valid_y}) * 100
    # train_rate = sess.run(accuracy, feed_dict={x: train_xs, y_: train_ys}) * 100

    # g_test[i/100] = test_rate
    # g_train[i/100] = train_rate
    # g_valid[i/100] = valid_rate

    print "iteration ", i
    print "Test:", test_rate, "%"
    print "validation:", valid_rate, "%"
    # print "Train:", train_rate, "%"

    print "-----------------------------------"

print "Done\n"

print "Graphing..."
# Generate graphs
# x = arange(1, 5001, 100)

# Part 1
# figure(1)
# title("Training Set Leraning Curves")
# xlabel("iterations")
# ylabel("%")
# plot(x, g_train, label="training set")
# savefig("figure_1.png")

# figure(2)
# title("Test Set Leraning Curves")
# xlabel("iterations")
# ylabel("%")
# plot(x, g_test, label="test set")
# savefig("figure_2.png")

# figure(3)
# title("Validation Set Leraning Curves")
# xlabel("iterations")
# ylabel("%")
# plot(x, g_valid, label="validation set")
# savefig("figure_3.png")

# show()