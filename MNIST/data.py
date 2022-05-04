import numpy as np
import scipy.io as sio
import math

def loadData(filename):
    mat = sio.loadmat(filename)
    return mat['testv'].astype('int16'), mat['trainv'].astype('int16'), mat['testlab'].astype('int16').reshape(len(mat['testlab']),), mat['trainlab'].astype('int16').reshape(len(mat['trainlab']),)

def createDataChunks(chunkSizeTest, chunkSizeTrain, testData, testLabels, trainData, trainLabels):
    rows, columns = testData.shape
    nrChunksTest = int(rows/chunkSizeTest)
    testDataChunks = []
    for i in range(nrChunksTest):
        testDataChunks.append(testData[chunkSizeTest*i:chunkSizeTest*(i+1)][:])

    rows,columns = trainData.shape
    nrChunksTrain = int(rows/chunkSizeTrain)
    trainDataChunks = []
    for i in range(nrChunksTrain):
        trainDataChunks.append(trainData[chunkSizeTrain*i:chunkSizeTrain*(i+1)][:])

    testLabelsChunks = []
    for i in range(nrChunksTest):
        testLabelsChunks.append(testLabels[chunkSizeTest*i:chunkSizeTest*(i+1)][:])   


    trainLabelsChunks = []
    for i in range(nrChunksTrain):
        trainLabelsChunks.append(trainLabels[chunkSizeTrain*i:chunkSizeTrain*(i+1)][:])   
    
    return testDataChunks, testLabelsChunks, trainDataChunks, trainLabelsChunks

def truncateData(numTest, numTrain ,testData, testLabels, trainData, trainLabels):
    MAX_TEST_SAMPLES = 10000
    MAX_TRAIN_SAMPLES = 60000
    numTest = min(MAX_TEST_SAMPLES,numTest)
    numTrain = min(MAX_TRAIN_SAMPLES,numTrain)
    return testData[:numTest][:], testLabels[:numTest], trainData[:numTrain][:], trainLabels[:numTrain]