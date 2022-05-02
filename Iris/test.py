from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Just for testeing: Print the iris data set 
def printDataSet(dataSet):
    for i,element in enumerate(dataSet):
        print(i+1 , ":" , element)    


# Add a 1 at the end of each sample 
def augmentData(dataSet):
    N = np.shape(dataSet)[0]
    D = np.shape(dataSet)[1]

    augmentedDataSet = np.zeros((N,D+1))
    for i,sample in enumerate(dataSet):
        augmentedDataSet[i] = np.append(sample,[1])

    return augmentedDataSet 
    

# Takes in a data set and returns a splited version, all as 2D numpy-arrays
def splitDataIntoTrainigAndTest(dataSet, labelSet, lastThirtyBool):
    Nc = 50 #samplesOfEachClass
    C = 3 #numberOfClasses
    splitIndex = 30

    if lastThirtyBool:
        splitIndex = 20

    trainingSamples  = []
    trainingLabels = []
    testSamples = []
    testLabels = []

    for i in range(Nc):
        for j in range(C):
            if i < splitIndex:
                trainingSamples.append(dataSet[i + Nc*j])
                trainingLabels.append(labelSet[i + Nc*j])
            else:
                testSamples.append(dataSet[i + Nc*j])
                testLabels.append(labelSet[i + Nc*j])

    # Make array into numpy array
    trainingSamples = np.array(trainingSamples)
    testSamples = np.array(testSamples)

    # Make 1D-array into 2D
    trainingLabels = np.reshape(trainingLabels, (np.shape(trainingLabels)[0], 1))
    testLabels = np.reshape(testLabels, (np.shape(testLabels)[0], 1))

    # Change order of train and test sets if we want thirty last samples 
    if lastThirtyBool:
        return testSamples, testLabels, trainingSamples, trainingLabels

    return trainingSamples, trainingLabels, testSamples, testLabels 


# One-Hot encoding of (vector of) classes
# Takes in classes reperesented by a scalar and returns a vector reperesentastion 
def oneHotEncoding(labelset):
    N = np.shape(labelset)[0]
    C = 3

    labelVectorSet = np.zeros((N,C))
    for i,clas in enumerate(labelset):
        # if clas == 0: labelVectorSet[i] = [1,0,0]
        # elif clas == 1: labelVectorSet[i] = [0,1,0]
        # else: #clas == 2: labelVectorSet[i] = [0,0,1]
        labelVectorSet[i][int(clas)] = 1

    return labelVectorSet


# Takes in a sample and a matrix W and returns sigmoid-classified label
def sigmoid(W,sample):
    z = np.matmul(W,sample)
    g = 1/(1 + np.exp(-z))
    return g


# Sigmoid on set of samples
def classifySamples(W,samples):
    C = 3
    N = np.shape(samples)[0]

    classifiedLabels = np.empty((N,C))
    for i,sample in enumerate(samples):
        classifiedLabels[i] = sigmoid(W,sample)

    return classifiedLabels 


# Takes in a set of classified labels (vector of floats) and returns a binary rounding, ie a vector of 0's and a 1
def binaryClassifySamples(classifiedLabels):
    N = np.shape(classifiedLabels)[0]

    # Define a vector for collecting the scalar rep of the class based on the index of the highest elem 
    scalarClassVec = np.zeros((N,1))

    for i,label in enumerate(classifiedLabels):
        scalarClassVec[i] = np.argmax(label)

    binaryClassifiedLabels = oneHotEncoding(scalarClassVec)
    return binaryClassifiedLabels 


# Update or improve W based on the algorithm W(m) = W(m − 1) − α∇_W(MSE)
def improveW(initialW,alpha,classifiedLabels,trueLabels,samples):
    g = classifiedLabels
    t = trueLabels
    
    x = samples

    N = np.shape(g)[0]
    C = np.shape(g)[1]
    D = np.shape(x)[1] - 1 
    
    vecOfOnes = np.ones((1,C))
    grad_WMSE = 0
    for i in range(N):
        grad_gMSE  = np.reshape(g[i]-t[i], (C,1)) 
        grad_zg    = np.reshape(np.multiply(g[i],(vecOfOnes-g[i])), (C,1)) 
        grad_Wz    = np.reshape(x[i], (1,D+1))
        grad_WMSE += np.matmul(np.multiply(grad_gMSE,grad_zg),grad_Wz)
                
    W = initialW - alpha*grad_WMSE
    return W

# Train classifier by improving W numberOfItertions times 
def trainClassifier(initialW, numberOfIterations, alpha, labels, samples):
    W = initialW 
    mseVec = np.zeros((numberOfIterations,1))
    errorRateVec = np.zeros((numberOfIterations,1))

    for i in range(numberOfIterations):
        trainClassifiedLabels = classifySamples(W,samples)
        W = improveW(W,alpha,trainClassifiedLabels,labels,samples )

        # Recording the performance/improvement
        mseVec[i] = MSE(trainClassifiedLabels,labels)
        errorRateVec[i] = errorRate(binaryClassifySamples(trainClassifiedLabels), labels)

    return W, mseVec, errorRateVec


# Returns the Minimum Square Error (MSE)
def MSE(classifiedLabels,trueLabels): 
    N = np.shape(classifiedLabels)[0]
    sum = 0 
    for i in range(N):
        error = classifiedLabels[i]-trueLabels[i] 
        sum += np.matmul(np.transpose(error),error)
    return sum/2


# Returns the error rate, ie number of errros wrt number of samples N
def errorRate(classifiedLabels, trueLabels):
    N = np.shape(classifiedLabels)[0]
    errors = 0
    for i in range(N):
        if not (classifiedLabels[i] == trueLabels[i]).all():
            errors += 1
    return errors/N 


# Just for testeing - A simple test for cheching how each sample is classified wrt the true labels
def HitOrMissTest(trueLabels, classifiedLabels):
    N = np.shape(trueLabels)[0]
    for i in range(N):
        if (trueLabels[i] == classifiedLabels[i]).all():
            print("Hit")
        else:
            print("Miss:",trueLabels[i], "was wrongly classified as", classifiedLabels[i])
  

# Returns a confusion matrix based on w, a set of samples and it's true labels 
def confusion(W, samples, labels):
    N = np.shape(labels)[0]
    C = np.shape(W)[0]
    classifiedLabels = binaryClassifySamples(classifySamples(W,samples))

    confusionMatrix = np.zeros((C,C))
    for i in range(N):
        trueLabel = np.argmax(labels[i])
        clasLabel = np.argmax(classifiedLabels[i])
        confusionMatrix[trueLabel][clasLabel] += 1    
    ##
    errorRateVal = errorRate(classifiedLabels, labels)
    ##
    
    return confusionMatrix, errorRateVal


# Plots confusion matrices for a training and testing set, respectively 
def plotConfusionMatrices(trainConfMat, trainErrorRate, testConfMat, testErrorRate):
    labelsStringVec = ["Setosa", "Versicolor", "Virginica"]

    _, axes = plt.subplots(1,2, figsize=(10,4))

    confVec = [trainConfMat,testConfMat]
    confVecNames = ["Training set", "Test set"]

    errorRates = [trainErrorRate, testErrorRate]

    for i in range (2):
        disp = ConfusionMatrixDisplay(confusion_matrix=confVec[i], display_labels=labelsStringVec)
        disp.plot(ax=axes[i])
        disp.ax_.set_title(f"{confVecNames[i]} \nEroor rate {round(errorRates[i]*100,1)} %")
        disp.im_.colorbar.remove()
    plt.subplots_adjust(wspace=0.40)
    plt.show() 


# Plotting of MSE and error rate
def plotMseAndErrorRate(alphas, mseVecs, errorRateVecs):
    _, axes = plt.subplots(1,2, figsize=(10,4))

    numberOfIterations = np.shape(mseVecs)[1]
    iterationsVec = np.arange(numberOfIterations)

    for i,mseVec in enumerate(mseVecs):
        # axes[0].set_title('MSE')
        axes[0].plot(iterationsVec,mseVec, label = "α = " + str(alphas[i]))
        axes[0].set(xlabel = 'Number of iterations', ylabel = 'MSE') 
        axes[0].legend()

    for i,errorRateVec in enumerate(errorRateVecs):
        # axes[1].set_title('Error rate')
        axes[1].plot(iterationsVec, errorRateVec, label = "α = " + str(alphas[i]))
        axes[1].set(xlabel = 'Number of iterations', ylabel = 'Error rate') 
        axes[1].legend()

    plt.show()


# Train the classifier for a set of different alphas and returns mse and error rates
def testDifferentAlphas(W,alphas, numberOfIterations,trainLabels,trainSamples):
    mseVecs = []
    errorRateVecs = []
    for alpha in alphas:
        _, mseVec, errorRateVec = trainClassifier(W, numberOfIterations, alpha, trainLabels, trainSamples)
        mseVecs.append(mseVec)
        errorRateVecs.append(errorRateVec)
    return np.array(mseVecs), np.array(errorRateVecs)


# Plot feature histogram for the different classes 
def plotFeatureHistogram():
    iris = load_iris()
    samples = iris.data
    C = 3
    Nc = 50
    
    flowers = []
    for i in range(C):
        flowers.append(samples[Nc*i:Nc*(i+1)])

    featuresTxt = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    flowersTxt = ["Setosa", "Versicolor", "Virginica"]

    _, axes = plt.subplots(2, 2, figsize = (8,7))
    for i,feature in enumerate(featuresTxt):
        for j,flower in enumerate(flowersTxt):
            axes.flat[i].hist(flowers[j][:,i], label = flower, alpha = 0.65)
        axes.flat[i].set_title(feature)
        axes.flat[i].set(xlabel = 'Length in cm', ylabel = 'Number of samples') 

    axes.flat[3].legend() # Much space in this plot, and legends only needed in one of the subplots  
    plt.tight_layout()
    plt.show()


# A large collection of functions put together to be called from the menu 
def run(lastThirtyBool,showDifferentAlphasBool, numberOfFeatures): 
    # Load data set
    iris      = load_iris()
    samples   = iris.data
    labels    = iris.target
    labelsTxt = iris.target_names

    # Remove features if desirable (the worst features are removed first)
    if numberOfFeatures == 3: 
        samples = np.delete(samples, 1, 1) 
    elif numberOfFeatures == 2:
        samples = np.delete(samples, [0,1], 1) 
    elif numberOfFeatures == 11:
        samples = np.delete(samples, [0,1,2], 1) 
    elif numberOfFeatures == 12:
        samples = np.delete(samples, [0,1,3], 1)

    # Add a 1 at the end of each sample 
    augmentedIris = augmentData(samples)

    # Split the data into traing and test, both into sets of samplels and labels   
    trainSamples, trainLabels, testSamples, testLabels = splitDataIntoTrainigAndTest(augmentedIris, labels, lastThirtyBool)

    # Tranfor the labels from scalar to vector represeatation 
    trainLabels = oneHotEncoding(trainLabels)
    testLabels = oneHotEncoding(testLabels)

    # Parameters of the problem 
    C = np.shape(labelsTxt)[0] # Number of classes = 3
    D = np.shape(samples)[1]   # Number of features (= 4 for the original problem)

    # Initial W
    W = np.zeros((C, D+1))

    # Test different alphas and number of iterations 
    alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    numberOfIterations = 2000  # Chosen based on a comprehensive testing    
    if showDifferentAlphasBool:
        mseVecs, errorRateVecs = testDifferentAlphas(W,alphas,numberOfIterations,trainLabels,trainSamples)
        plotMseAndErrorRate(alphas, mseVecs, errorRateVecs)
    
    # Train the classifier (again) with the chosen alpha (and number of iterations) based on testing 
    alpha = alphas[2]
    trainedW, _, _ = trainClassifier(W, numberOfIterations, alpha, trainLabels, trainSamples)    

    # Generate confusion matrices
    trainsetConfusionMatrix, trainErrorRate = confusion(trainedW, trainSamples, trainLabels) # print(trainsetConfusionMatrix)
    testsetConfusionMatrix, testErrorRate = confusion(trainedW, testSamples, testLabels) # print(testsetConfusionMatrix)

    # Plot the confusion matrices 
    plotConfusionMatrices(trainsetConfusionMatrix,trainErrorRate,testsetConfusionMatrix,testErrorRate)


# Menu
menuOpts = {
    1: 'Train classifier with 30 first samples',
    2: 'Train classifier with 30 last samples',
    3: 'Show feature and class histogram',
    4: 'Take away Sepal width',
    5: 'Take away Sepal width and lenght',
    6: 'Take away Sepal width and lenght, and Petal length',
    7: 'Take away Sepal width and lenght, and Petal width ',
    8: 'Close',
}

showDifferentAlphasBool = 0 #1 # Was turned of during testing to save computational power 
loop = True

while loop:
    # Default settings
    lastThirtyBool = 0
    numberOfFeatures = 4 # Number of the 4 features in use (Note: 1 has two options, as described in the menu)

    # Disp menu
    print('\nMenu options:')
    for key in menuOpts.keys():
        print(' ', key, '-', menuOpts[key])

    option = int(input('Enter your number: '))

    if option == 1:
        run(lastThirtyBool,showDifferentAlphasBool, numberOfFeatures)

    elif option == 2:
        lastThirtyBool = 1
        run(lastThirtyBool,showDifferentAlphasBool, numberOfFeatures)
    
    elif option == 3:
        plotFeatureHistogram()

    elif option == 4:
        numberOfFeatures = 3
        run(lastThirtyBool,showDifferentAlphasBool, numberOfFeatures)

    elif option == 5:
        numberOfFeatures = 2
        run(lastThirtyBool,showDifferentAlphasBool, numberOfFeatures)

    elif option == 6:
        numberOfFeatures = 11
        run(lastThirtyBool,showDifferentAlphasBool, numberOfFeatures)

    elif option == 7:
        numberOfFeatures = 12
        run(lastThirtyBool,showDifferentAlphasBool, numberOfFeatures)
    
    elif option == 8:
        print('Closing')
        loop = False

    else:
        print('Invalid option. Valid inputs are numbers between 1 and 8')