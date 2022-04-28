import numpy as np
import scipy.io as sio
import math
import string
import matplotlib.pyplot as plt

def plotMisclassifiedPix(testData, labels, classified, numImages, task):
    indices = np.where(labels != classified)[0]
    numImages = min(np.size(indices),numImages)
    indices = indices[:numImages]
  
    numRows = np.uint16(np.floor(np.sqrt(numImages)))
    numCols = np.uint16(numImages / numRows)

    fig = plt.figure(task,figsize=(5, 5))
    fig.suptitle("Misclassified digits\n\n True/Classification\n",size = 10)
    plt.subplots_adjust(top = 0.8, bottom=0.4, hspace=0.5, wspace=0.1)
    for row in range(numRows):
        for col in range(numCols):
            i = (row * numCols) + col
            imageIndex = indices[i]

            ax = fig.add_subplot(numRows, numCols, i+1, autoscale_on = True)
            
            ax.axis("off")
            ax.set_title(f'{labels[imageIndex]}/{classified[imageIndex]}')
            image = np.reshape(testData[imageIndex, :], (28,28))
            
            img_plot = ax.imshow(image,cmap="bone")
            img_plot.set_interpolation("nearest")
            
    

def plotClassifiedPix(testData, labels, classified, numImages, task):
    indices = np.where(labels == classified)[0]
    numImages = min(np.size(indices),numImages)
    indices = indices[:numImages]
  
    numRows = np.uint16(np.floor(np.sqrt(numImages)))
    numCols = np.uint16(numImages / numRows)

    fig = plt.figure(task,figsize=(5, 5))
    fig.suptitle("Correctly classified digits\n\n True/Classification\n",size = 10)
    plt.subplots_adjust(top = 0.8, bottom=0.4, hspace=0.5, wspace=0.1)
    for row in range(numRows):
        for col in range(numCols):
            i = (row * numCols) + col
            imageIndex = indices[i]

            ax = fig.add_subplot(numRows, numCols, i+1, autoscale_on = True)
            ax.axis("off")
            ax.set_title(f'{labels[imageIndex]}/{classified[imageIndex]}')

            image = np.reshape(testData[imageIndex, :], (28,28))

            img_plot = ax.imshow(image,cmap="bone")
            img_plot.set_interpolation("nearest")