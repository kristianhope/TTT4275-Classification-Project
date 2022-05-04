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
    result = kNN.NN_Classification(trainData,trainLabels,testData)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print("\nConfusion Matrix: \n",kNN.ConfusionMatrix(testLabels,result),"\n","Error rate: \n",kNN.ErrorRate(testLabels,result))
    plot.plotConfusionMatrix(kNN.ConfusionMatrix(testLabels,result),kNN.ErrorRate(testLabels,result),"NN classifier")
    plot.plotClassifiedPix(testData,testLabels,result,10, "Task 1 C")
    plot.plotMisclassifiedPix(testData,testLabels,result,10, "Task 1 M")

def runTask2b(trainData, trainLabels, testData, testLabels):
    print("Task 2b: NN Classifier w clustered data as templates\n")

    start = timeit.default_timer()
    result = kNN.NN_Classification(trainData,trainLabels,testData)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print("\nConfusion Matrix: \n",kNN.ConfusionMatrix(testLabels,result),"\n","Error rate: \n",kNN.ErrorRate(testLabels,result))
    plot.plotConfusionMatrix(kNN.ConfusionMatrix(testLabels,result),kNN.ErrorRate(testLabels,result),"NN classifier with clustering")


def runTask2c(trainData, trainLabels, testData, testLabels):
        print("Task 2c: kNN Classifier w k = 7 and clustered data as templates")
        start = timeit.default_timer()
        result = kNN.kNN_Classification(trainData,trainLabels,testData,7)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        print("\nConfusion Matrix: \n",kNN.ConfusionMatrix(testLabels,result),"\n","Error rate: \n",kNN.ErrorRate(testLabels,result))
        plot.plotConfusionMatrix(kNN.ConfusionMatrix(testLabels,result),kNN.ErrorRate(testLabels,result),"7-NN classifier with clustering")

def main():
    TEST_CHUNK_SIZE = 100
    TRAIN_CHUNK_SIZE = 60000
    NUM_CLUSTERS = 64

    testData,trainData,testLabels,trainLabels = dt.loadData('MNIST/data_all.mat')
    testDataChunks,testLabelsChunks,trainDataChunks,trainLabelsChunks = dt.createDataChunks(TEST_CHUNK_SIZE,TRAIN_CHUNK_SIZE,testData,testLabels,trainData,trainLabels)

    while True:
        fullTest = input("Use full datasets? 1/0\n")
        if int(fullTest) == 0:
            numTest = input("How many test samples?\n")
            numTrain = input("How many training samples?\n")
            testData,testLabels,trainData,trainLabels = dt.truncateData(int(numTest),int(numTrain),testData,testLabels,trainData,trainLabels)
        task = input("Which task do you want to run? \n 1/2b/2c/all\n")
        if task == '1':
            runTask1(trainData,trainLabels,testData,testLabels)
            plt.show()
        elif task == '2b':
            clusters, clusterLabels = kNN.createClusters(NUM_CLUSTERS, trainData,trainLabels)
            runTask2b(clusters,clusterLabels,testData,testLabels)
            plt.show()
        elif task == '2c':
            clusters, clusterLabels = kNN.createClusters(NUM_CLUSTERS, trainData,trainLabels)
            runTask2c(clusters,clusterLabels,testData,testLabels)
            plt.show()
        elif task == 'all':
            runTask1(trainData,trainLabels,testData,testLabels)
            clusters, clusterLabels = kNN.createClusters(NUM_CLUSTERS, trainData,trainLabels)
            runTask2b(clusters,clusterLabels,testData,testLabels)
            runTask2c(clusters,clusterLabels,testData,testLabels)
            plt.show()
        else:
            print("Not a valid task")


main()