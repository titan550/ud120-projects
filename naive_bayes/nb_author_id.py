#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
import numpy as np
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

start_time = time()
clf.fit(features_train, labels_train)
finish_time = time()-start_time

pred = clf.predict(features_test)
accuracy = (labels_test == pred).sum()/float(len(labels_test))

print "Algorith accuracy is: ", accuracy, "%"
print "Training time is", finish_time, "s"

#########################################################


