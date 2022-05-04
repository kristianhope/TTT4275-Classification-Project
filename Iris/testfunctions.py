# Functions used during testeing 
from sklearn.datasets import load_iris
import numpy as np

# Print the iris data set 
def printDataSet(dataSet):
    for i,element in enumerate(dataSet):
        print(i+1 , ":" , element)  


# A simple test for cheching how each sample is classified wrt the true labels
def hitOrMissTest(trueLabels, classifiedLabels):
    N = np.shape(trueLabels)[0]
    for i in range(N):
        if (trueLabels[i] == classifiedLabels[i]).all():
            print("Hit")
        else:
            print("Miss:",trueLabels[i], "was wrongly classified as", classifiedLabels[i])