import numpy as np
import scipy.io as sio
import timeit
import plotting as plot
import kNN
import data as dt
import matplotlib.pyplot as plt

def runTask1(trainData, trainLabels, testData, testLabels):
    print("Task 1: NN Classifier\n")
    start = timeit.default_timer()
    result = kNN.kNN_Classification(trainData,trainLabels,testData,1)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print("\nConfusion Matrix: \n",kNN.ConfusionMatrix(testLabels,result),"\n","Error rate: \n",kNN.ErrorRate(testLabels,result))
    plot.plotClassifiedPix(testData,testLabels,result,10, "Task 1 C")
    plot.plotMisclassifiedPix(testData,testLabels,result,10, "Task 1 M")

def runTask2b(trainData, trainLabels, testData, testLabels):
    print("Task 2b: NN Classifier w clusters as templates\n")
    NUM_CLUSTERS = 64
    clusters, clusterLabels = kNN.createClusters(NUM_CLUSTERS, trainData,trainLabels)
    start = timeit.default_timer()
    result = kNN.kNN_Classification(clusters,clusterLabels,testData,1)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print("\nConfusion Matrix: \n",kNN.ConfusionMatrix(testLabels,result),"\n","Error rate: \n",kNN.ErrorRate(testLabels,result))
    plot.plotClassifiedPix(testData,testLabels,result,10,"Task 2b C")
    plot.plotMisclassifiedPix(testData,testLabels,result,10,"Task 2b M")

def runTask2c(trainData, trainLabels, testData, testLabels):
    print("Task 2c: kNN Classifier w k = 7\n")
    start = timeit.default_timer()
    result = kNN.kNN_Classification(trainData,trainLabels,testData,7)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print("\nConfusion Matrix: \n",kNN.ConfusionMatrix(testLabels,result),"\n","Error rate: \n",kNN.ErrorRate(testLabels,result))
    plot.plotClassifiedPix(testData,testLabels,result,10,"Task 2c C")
    plot.plotMisclassifiedPix(testData,testLabels,result,10,"Task 2c M")

11
def main():
    TEST_CHUNK_SIZE = 100
    TRAIN_CHUNK_SIZE = 60000

    NUM_TEST = 100
    NUM_TRAIN = 10000

    testData,trainData,testLabels,trainLabels = dt.loadData('data_all.mat')
    testDataChunks,testLabelsChunks,trainDataChunks,trainLabelsChunks = dt.createDataChunks(TEST_CHUNK_SIZE,TRAIN_CHUNK_SIZE,testData,testLabels,trainData,trainLabels)
    testData,testLabels,trainData,trainLabels = dt.truncateData(NUM_TEST,NUM_TRAIN,testData,testLabels,trainData,trainLabels)

    while True:
        task = input("Which task do you want to run? \n")
        if task == '1':
            runTask1(trainData,trainLabels,testData,testLabels)
            #plt.show()
        elif task == '2b':
            runTask2b(trainData,trainLabels,testData,testLabels)
            #plt.show()
        elif task == '2c':
            runTask2c(trainData,trainLabels,testData,testLabels)
            #plt.show()
        elif task == 'all':
            runTask1(trainData,trainLabels,testData,testLabels)
            runTask2b(trainData,trainLabels,testData,testLabels)
            runTask2c(trainData,trainLabels,testData,testLabels)
            #plt.show()
        else:
            print("Not a valid task")


main()