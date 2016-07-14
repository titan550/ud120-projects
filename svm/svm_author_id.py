#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


clf = SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train,labels_train)
t1 = time()

pred = clf.predict(features_test)
ChrisCount = (pred == 1).sum()
print ChrisCount
# accuracy = (labels_test == pred).sum()/float(len(labels_test))
# print "SVC Accuracy: " , accuracy
# print "SVC Training Time:" , round(t1-t0,3), "s"

# clf = GaussianNB()
# t0 = time()
# clf.fit(features_train, labels_train)
# t1 = time()
# pred = clf.predict(features_test)
# accuracy = (labels_test == pred).sum()/float(len(labels_test))
# print "NB Accuracy: ", accuracy
# print "NB Training Time:" , round(t1-t0,3), "s"

########## Reducing set to 1% of the original size
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

# clf = SVC(kernel='rbf', C=10000)
# t0 = time()
# clf.fit(features_train,labels_train)
# t1 = time()

# predictions = clf.predict((features_test[10],features_test[26],features_test[50]))

# print predictions
# pred = clf.predict(features_test)

# accuracy = (labels_test == pred).sum()/float(len(labels_test))
# print "Reduced SVC Accuracy: " , accuracy
# print "Reduced SVC Training Time:" , round(t1-t0,3), "s"

# clf = GaussianNB()
# t0 = time()
# clf.fit(features_train, labels_train)
# t1 = time()
# pred = clf.predict(features_test)
# accuracy = (labels_test == pred).sum()/float(len(labels_test))
# print "Reduced NB Accuracy: ", accuracy
# print "Reduced NB Training Time:" , round(t1-t0,3), "s"

#########################################################


