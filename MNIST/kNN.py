import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import string
import timeit
from cgi import test
from scipy.spatial import distance
from sklearn.cluster import KMeans


def createClassDict(data, labels, nrClasses = 10):
    classDict = {}
    for i in range(nrClasses):
        indexes = np.where(labels == i)[0]
        classVecs = data[indexes,:]
        classDict[i] = classVecs
    return classDict

def createClusters(M, data, labels, nrClasses = 10, vecSize = 784):
    classDict = createClassDict(data,labels)

    clusters = np.zeros((M*nrClasses, vecSize),np.uint32)
    labels = np.empty((M*nrClasses,),int)
    for i in range(nrClasses):
        clusters[M*i:M*(i+1)][:] = KMeans(n_clusters = M).fit(classDict[i]).cluster_centers_
    for i in range(nrClasses):
        for j in range(M):
            labels[i*M + j] = i
    return clusters, labels

def NN(trainData, labels, test):
    distances = [] #[(distance, label),...]
    for i in range(len(trainData)):
        distances.append((np.linalg.norm(trainData[i]-test),labels[i]))
    distances = sorted(distances,key=lambda tup: tup[0])
    nn = distances[0][1]
    return nn

def NN_Classification(trainData, labels, testData):
    result = np.array([],int)
    for test in testData:
        nn = NN(trainData, labels, test)
        result = np.append(result,nn)
    return result

def kNN(trainData, labels, test,k):
    distances = [] #[(distance, label),...]
    for i in range(len(trainData)):
        distances.append((np.linalg.norm(trainData[i]-test),labels[i]))
    distances = sorted(distances,key=lambda tup: tup[0])
    kNearest = distances[0:k]
    kNearestLabels = [el[1] for el in kNearest]
    return kNearestLabels

def findMajority(kNearest):
    frequencies = [0 for i in range(10)]
    for el in kNearest:
        frequencies[el] +=1
    majority = frequencies.index(max(frequencies))
    return majority


def kNN_Classification(trainData, labels, testData, k):
    result = np.array([],int)
    for test in testData:
        kNearest = kNN(trainData, labels, test, k)
        majority = findMajority(kNearest)
        result = np.append(result,majority)
    return result

def ConfusionMatrix(testLabels, classified):
    confusionMatrix = np.zeros((10,10),int)
    for i in range(len(classified)):
        if classified[i] == testLabels[i]:
            confusionMatrix[classified[i]][classified[i]] += 1
        else:
            confusionMatrix[testLabels[i]][classified[i]] += 1
    return confusionMatrix

def ErrorRate(testLabels, classified):
    errors = 0
    for i in range(len(classified)):
        if classified[i] != testLabels[i]:
            errors += 1
    return float(errors)/float(len(testLabels))


    

            

