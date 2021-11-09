"""
models.py

Contains the different methods for classification

KNeighbors Classification               (Not Implemented)
Linear SVC/SVM                          (Not Implemented)
SVC                                     (Not Implemented)
HOG-SVM - Stochastic Gradient Descent   (Not Implemented)
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

# KNearestNeighbors
def KNN():
    knn = KNeighborsClassifier()
    return knn

# SVC
# Images for SVC/SVM must be the same dimensions
def SVC():
    svc = svm.SVC(probability=True)
    return svc

#Gaussian Naive Bayes
def GNaiveBayes():
    gnb = GaussianNB()
    return gnb
   
