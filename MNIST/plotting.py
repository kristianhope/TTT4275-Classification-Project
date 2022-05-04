import numpy as np
import scipy.io as sio
import math
import string
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plotMisclassifiedPix(testData, labels, classified, numImages, task):
    indices = np.where(labels != classified)[0]
    numImages = min(np.size(indices),numImages)
    indices = indices[0:numImages]
  
    numRows = np.uint16(np.floor(np.sqrt(numImages))-1)
    numCols = np.uint16(numImages / numRows)

    fig = plt.figure(task,figsize=(10, 5))
    fig.suptitle("Misclassified images\n\n True | Predicted\n",size = 10)
    plt.subplots_adjust(top = 0.8, bottom=0.1, hspace=0.3, wspace=0.1)
    for row in range(numRows):
        for col in range(numCols):
            k = (row * numCols) + col
            index = indices[k]

            sp = fig.add_subplot(numRows, numCols, k+1, autoscale_on = True)
            sp.axis("off")
            sp.set_title(f'{labels[index]} | {classified[index]}')
            image = np.reshape(testData[index, :], (28,28))
            
            plot = sp.imshow(image)
            plot.set_interpolation("nearest")
            
    

def plotClassifiedPix(testData, labels, classified, numImages, task):
    indices = np.where(labels == classified)[0]
    numImages = min(np.size(indices),numImages)
    indices = indices[0:numImages]
  
    numRows = np.uint16(np.floor(np.sqrt(numImages))-1)
    numCols = np.uint16(numImages / numRows)

    fig = plt.figure(task,figsize=(10, 5))
    fig.suptitle("Correctly classified images\n\n True | Predicted\n",size = 10)
    plt.subplots_adjust(top = 0.8, bottom=0.1, hspace=0.3, wspace=0.1)
    for row in range(numRows):
        for col in range(numCols):
            k = (row * numCols) + col
            index = indices[k]

            sp = fig.add_subplot(numRows, numCols, k+1, autoscale_on = True)
            sp.axis("off")
            sp.set_title(f'{labels[index]} | {classified[index]}')
            image = np.reshape(testData[index, :], (28,28))
            
            plot = sp.imshow(image)
            plot.set_interpolation("nearest")


def plotConfusionMatrix(cm, errorRate, title):
    disp = ConfusionMatrixDisplay(cm,display_labels=np.arange(10))
    disp.plot()
    disp.ax_.set_title(f"{title}\nError rate = {round(errorRate*100,1)}%",fontsize=11)