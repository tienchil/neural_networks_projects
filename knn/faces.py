from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
import re
from scipy.ndimage import filters

"""

Use of k-nearest-neighbour to perform face recognition tasks.

"""


# A simple class to store the data with the corresponding labels.
class Face:
    
    def __init__(self, data, label, filename, gender):
        
        self.v = data
        self.label = label
        self.filename = filename
        self.gender = gender
        
    def __repr__(self):
        
        return self.filename + ":" + str(self.v.shape) + "," + self.gender



def label_data(path, filename, gender):
    """ Resize the img to 32x32, and label the image in the 
        last element of array. """
    
    # Resize and Flatten the image
    img = imread(path+filename, flatten=True)
    img = imresize(img, (32, 32))
    v = img.flatten()
    
    # Find the label
    label = re.split('(\d+)', filename)[0]
    
    # Create a new object and return it
    new_face = Face(v, label, filename, gender)
    
    return new_face


def create_flatten(files, path, labels, gender):
    """ Organize flattened image data with labels into a list."""
    
    data = []
    
    for filename in files:
        
        img = label_data(path, filename, gender)
        
        # Check if the files are correct
        if (img.label in labels):
            
            data.append(img)
        
    return data

def get_set(n, data, labels, seed):
    """ Generate a set for data with associated labels. 
        n is the number of data chosen per label. """
    
    # Change the seed if you want a different result
    random.seed(seed)
    dataset = []
    
    for l in labels:
        
        temp = [face for face in data if face.label == l]
        dataset = dataset + random.sample(temp, n)
        
    
    return dataset
    

def L2_distance(v1, v2):
    """ Calculate L2 distance between v1 and v2."""
    
    A = v1.astype(float)
    B = v2.astype(float)
    
    return np.sqrt(np.sum((A - B)**2))


def knn(k, train_set, test_set, labels):
    """ Perform k-nearest-neighbour on test set, and
        validation set with associated labels. 
        
        This uses L2 distance with k numbers of neighbour
        to classify the test cases and validation cases.
        
        Returns a list of lists of nearest neighbours with 
        the last element to be the classification result."""
    
    dist = np.zeros(len(train_set), dtype=float)
    prediction = []
    
    for test in test_set:
        
        i = 0
        neighbour = [] # k nearest neighbours
        
        # Calculate all distances and store them to an array
        for train in train_set:
            
            dist[i] = L2_distance(test.v, train.v)
            
            i = i + 1
        
        # Find k closest neighbour
        for j in range(k):
            
            index = argmin(dist)
            neighbour.append(train_set[index])
            dist[index] = np.inf
            

        
        # Determine the classofication
        
        total = []
        
        for l in labels:
            
            s = 0
            
            for n in neighbour:
                
                if l == n.label:
                    
                    s = s + 1
                    
            total.append(s)
                    
        result = labels[total.index(max(total))]
        neighbour.append(result)
        prediction.append(neighbour)
        
    
    return prediction


def knn_gender(k, train_set, test_set, labels):
    """ knn for gender classification."""
    
    dist = np.zeros(len(train_set), dtype=float)
    prediction = []
    
    for test in test_set:
        
        i = 0
        neighbour = [] # k nearest neighbours
        
        # Calculate all distances and store them to an array
        for train in train_set:
            
            dist[i] = L2_distance(test.v, train.v)
            
            i = i + 1
        
        # Find k closest neighbour
        for j in range(k):
            
            index = argmin(dist)
            neighbour.append(train_set[index])
            dist[index] = np.inf
            

        
        # Determine the classofication
        
        total = []
        
        for l in labels:
            
            s = 0
            
            for n in neighbour:
                
                if l == n.gender:
                    
                    s = s + 1
                    
            total.append(s)
                    
        result = labels[total.index(max(total))]
        neighbour.append(result)
        prediction.append(neighbour)
        
    
    return prediction



def reduce_data(set1, set2):
    """ Delete the same data in set1 from set2."""
    
    new = []
    
    for i in range(len(set1)):
        
        if not (set1[i] in set2):
            
            new.append(set1[i])   
    
    return new


def get_success(data, knn_data):
    """ Calculate the performance of knn result."""
    
    success = 0.0
    
    for i in range(len(data)):
        
        if knn_data[i][-1] == data[i].label:

            success = success + 1.0
            
        elif knn_data[i][-1] == data[i].gender:

            success = success + 1.0
            
    rate = success / len(test)

    return rate*100.0

    
if __name__ == "__main__":
    
    # Change the path to the directory of training sets accordingly
    PATH_1 = "./male/"
    PATH_2 = "./female/"
    
    PATHS = [PATH_1, PATH_2]
    
    # Actor names
    actor = list(set([a.split("\t")[0].lower() \
                      for a in open("subset_actors.txt").readlines()]))
    
    actor = [s.split()[1] for s in actor]
    
    actress = list(set([a.split("\t")[0].lower() \
                        for a in open("subset_actresses.txt").readlines()]))
    
    actress = [s.split()[1] for s in actress]
    
    all_ppl = actor + actress
    
    # This is for part 1 to part 5
    act = ['butler', 'radcliffe', 'vartan', 'bracco', 'gilpin', 'harmon']
    rest = [person for person in all_ppl if not(person in act)]
    
    # Find all files in the directories
    # This assumes that the specified directory contains only
    # necessary data and no other directories.
    male_files = [im for im in os.listdir(PATH_1)]
    female_files = [im for im in os.listdir(PATH_2)]
    
    # Organize the data
    male_data = create_flatten(male_files, PATH_1, actor, "male")
    female_data = create_flatten(female_files, PATH_2, actress, "female")
    all_data = male_data + female_data
    
    # Generate the data to training set, test set, and
    # validation set, by using random.seed()
    train = get_set(100, all_data, act, 752)
    
    test_data = reduce_data(all_data, train) # Reduce the data to avoid overlapping
    test = get_set(10, test_data, act, 1325)
    
    
    valid_data = reduce_data(test_data, test) # Reduce the data to avoid overlapping
    validation = get_set(10, valid_data, act, 9231)
    
    
    # Run knn
    k_val = np.arange(1, 51)
    test_rate = np.zeros(len(k_val))
    valid_rate = np.zeros(len(k_val))
    neighbour = []

    for k in range(1, 51):
    
        test_re = knn(k, train, test, act)
        neighbour.append(test_re)
        valid = knn(k, train, validation, act)
        test_rate[k-1] = get_success(test, test_re)
        valid_rate[k-1] = get_success(validation, valid)
        
        print "Test Performance = {0} % for k = {1}".format(test_rate[k-1], k)
        print "Validation Performance = {0} % for k = {1}\n".format(valid_rate[k-1], k)
    
    for i in range(len(neighbour)):
        
        if neighbour[i][-1] != test[i].label:
            
            for j in range(5):
                
                print test[i].filename + ":" + neighbour[i][j].filename
    
    

   print "------------------"
   print "Gender Classification Starts"
   
   gtest_rate = np.zeros(len(k_val))
   gvalid_rate = np.zeros(len(k_val))

   for k in range(1, 51):
   
       test_re = knn_gender(k, train, test, ["male", "female"])
       valid = knn_gender(k, train, validation, ["male", "female"])
       gtest_rate[k-1] = get_success(test, test_re)
       gvalid_rate[k-1] = get_success(validation, valid)
       
       print "Test Performance = {0} % for k = {1}".format(gtest_rate[k-1], k)
       print "Validation Performance = {0} % for k = {1}\n".format(gvalid_rate[k-1], k)
   
   
   print "--------Part 6----------"
   # Part 6

   test = get_set(10, all_data, rest, 132)
   
   valid_data = reduce_data(all_data, test) # Reduce the data to avoid overlapping
   validation = get_set(10, valid_data, rest, 9231)
   
   train = reduce_data(valid_data, validation) # Reduce the data to avoid overlapping
   
   six_test_rate = np.zeros(len(k_val))
   six_valid_rate = np.zeros(len(k_val))

   for k in range(1, 51):
   
       test_re = knn_gender(k, train, test, ["male", "female"])
       valid = knn_gender(k, train, validation, ["male", "female"])
       six_test_rate[k-1] = get_success(test, test_re)
       six_valid_rate[k-1] = get_success(validation, valid)
       
       print "Test Performance = {0} % for k = {1}".format(six_test_rate[k-1], k)
       print "Validation Performance = {0} % for k = {1}\n".format(six_valid_rate[k-1], k)
   
   
   # Generate figures
   
   figure(1)
   title("Face Recognition Performance of various k on validation set")
   plot(k_val, valid_rate)
   xlabel("k")
   ylabel("%")
   
   figure(2)
   title("Face Recognition Performance of various k on test set")
   plot(k_val, test_rate)
   xlabel("k")
   ylabel("%")

   figure(3)
   title("Gender Classification Performance of various k on test set")
   plot(k_val, gvalid_rate)
   xlabel("k")
   ylabel("%")
   
   figure(4)
   title("Gender Classification Performance of various k on test set")
   plot(k_val, gtest_rate)
   xlabel("k")
   ylabel("%")
   
   figure(5)
   title("Gender Classification Performance of various k on test set")
   plot(k_val, six_valid_rate)
   xlabel("k")
   ylabel("%")
   
   figure(6)
   title("Gender Classification Performance of various k on test set")
   plot(k_val, six_test_rate)
   xlabel("k")
   ylabel("%")
   
   
   
   show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    