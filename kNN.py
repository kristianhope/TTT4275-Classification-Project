import numpy as np
import scipy.io as sio
import math
import plotting as plot
import string
import matplotlib.pyplot as plt
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

def EuclideanDistance(array1, array2):
    distance = 0
    for i in range(len(array2)):
        distance += (int(array1[i]) - int(array2[i]))**2
    return np.sqrt(distance)

def kNN(trainData, labels, test,k):
    distances = [] #[(distance, label),...]
    for i in range(len(trainData)):
        #distances.append((EuclideanDistance(test,trainData[i]),labels[i]))
        distances.append((np.linalg.norm(trainData[i]-test),labels[i]))
    distances = sorted(distances,key=lambda tup: tup[0])
    kNearest = distances[0:k]
    labels = [el[1] for el in kNearest]
    return labels

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
    confusionMatrix = np.zeros((10,10))
    for i in range(len(classified)):
        if classified[i] == testLabels[i]:
            confusionMatrix[classified[i]][classified[i]] += 1
        else:
            confusionMatrix[classified[i]][testLabels[i]] += 1
    return confusionMatrix

def ErrorRate(testLabels, classified):
    errors = 0
    for i in range(len(classified)):
        if classified[i] != testLabels[i]:
            errors += 1
    return float(errors)/float(len(testLabels))


    

            




#index = classDict[0][kmeans.labels_==2]
#im = Image.fromarray(index[8].reshape(28,28))
#im.save("test.png")
