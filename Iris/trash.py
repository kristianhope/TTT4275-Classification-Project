import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# import Iris.test

# W = np.random.randint(0,2, size=(3, 5))

# for i in range(W.shape[0]):
#     print(W[i])

# tmp = np.empty(0)
# print(tmp)

# print("-------")

# print(W)
# print(np.argmax(W))

# print("------")
# onesVec = np.ones((3,1))
# print(onesVec)

# onesVec = np.ones((1,3))
# print(onesVec)

# classes = np.array([[1,0,0],[0,1,0],[0,0,1]])
# print(np.shape(classes))
# print(classes)

#Split data numpy from start attempt
    # trainingX = np.zeros((Tr*C,D+1))
    # trainingLabels = np.zeros((Tr*C,1))
    # testX = np.zeros((Te*C,D+1))
    # testLabels = np.zeros((Tr*C,1))

    # iTr = 0
    # iTe = 0
    # for i in range(N):
    #     for j in range(C):
    #         if i < sizeOfTrainingSet:
    #             trainingX[iTr] = dataSet[i + N*j]
    #             trainingLabels[iTr] = labelSet[i + N*j]
    #             iTr += 1
    #         else:
    #             testX[iTe] = dataSet[i + N*j]
    #             testLabels = labelSet[i + N*j]
    #             iTe += 1


    #71___________________

# print(np.shape(train))
# print(np.shape(trainLabels))
# print(np.shape(test))
# print(np.shape(testLabels))
# print(train)
# print(trainLabels)
# print(test)
# print(testLabels)
# for i,val in enumerate(train):
#     print("Trainingsample", train[i])
#     print("True label",trainLabels[i])
#     print("--------------------")
    
# for i,val in enumerate(test):
#     print("Testsample",test[i])
#     print("True label", testLabels[i])
#     print("--------------------")


# print(np.shape(trainLabels), np.shape(testLabels))

# print(np.shape(trainLabels), np.shape(testLabels))



# for index, line in enumerate(train):
#     print("Training", index+1, ': ', line, "- True label: ", trainLabels[index])

# for index, line in enumerate(test):
#     print("Test", index+1, ': ', line, "- True label: ", testLabels[index])

# encodedLabels = oneHotEncoding(trainLabels)
# for i, elem in enumerate(encodedLabels):
#     print(i+1, ":", elem)

# C = 3
# D = 4
# W = np.zeros((C, D+1))
# print(W) 

#113___________________

# t = np.empty((3,1))
# m = np.zeros((3,1))
# print(t)
# print(m)

#_____________________________________
# for i in range(N):
#         grad_gMSE = np.reshape(g[i]-t[i], (C,1)) #np.subtract(g[i],t[i])
#         # print(g[i],"-",t[i],"=",g[i]-t[i])
#         grad_zg = np.reshape(np.multiply(g[i],(onesVec-g[i])), (C,1))  #g[i].astype(int)
#         # print(onesVec,"-",g[i],"=",onesVec-g[i])
#         grad_Wz = np.reshape(x[i], (1,D+1))
#         # print(grad_gMSE,"\n times \n",grad_zg)
#         # print("-----")
#         tmp = np.multiply(grad_gMSE,grad_zg)
#         # print(grad_Wz)
#         # print("Has dim", np.shape(grad_Wz))
#         # grad_Wz = np.transpose(grad_Wz)
#         # print(grad_Wz)
#         # print("Has dim", np.shape(grad_Wz))
        
#         # print(tmp)
#         # print(grad_Wz)¨
#         # print
#         # print(np.transpose(grad_Wz))
#         # print(np.shape(tmp),"x",np.shape(grad_Wz))
#         # print("------------")
#         grad_WMSE += np.matmul(tmp,grad_Wz)
                
#     # print(np.shape(grad_WMSE))
#     # print(grad_WMSE)
#     W = initialW - alpha*grad_WMSE


# def testClassifier(W, numberOfIterations, alpha, sampels, labels):
#     mseVec = np.zeros((numberOfIterations,1))
#     errorRateVec = np.zeros((numberOfIterations,1))

#     for i in range(numberOfIterations):
#         predLabels = predictedLabels(W,sampels)
#         W = improveW(W,alpha,predLabels,labels,sampels)
#         classifiedLabels = binaryPredictedLabels(predictedLabels(W,sampels))
#         W = improveW(W,alpha,classifiedLabels,labels,sampels)
#         mseVec[i] = MSE(classifiedLabels,labels)
#         errorRateVec[i] = errorRate(classifiedLabels, labels)
#     return mseVec, errorRateVec

  # #Applied on the the test set 
        # classifiedTestLabels = predictedLabels(W,testSamples)
        # mseVec[i] = MSE(classifiedTestLabels,testLabels)
        # classifiedTestLabels = binaryPredictedLabels(classifiedTestLabels)
        # errorRateVec[i] = errorRate(classifiedTestLabels, testLabels)
# [1,2,3]


# mseVecs = np.zeros((1,1))
# mseVecs = np.append(mseVecs,[[1,2,3]])
# mseVecs = np.append(mseVecs,[[4,5,6]])
# print(mseVecs)
# print(np.shape(mseVecs))
# one = 1
# zero = 0

# def tester(param):
#     if param:
#         return "1"
#     return "0"

# print(tester(0))

#____________

# # tmp = np.reshape(setosa[:,0],(50,1))
# # print(tmp)
# # print(np.shape(tmp))

# # # fig, axes = plt.subplots(2,2, figsize=(10,4)) #,sharey='row')

# #     # confVec = [trainConfMat,testConfMat]
# #     # confVecNames = ["Training set", "Test set"]
# # plt.figure(figsize=(16,12))

# # for i,feature in enumerate(features):
# #     classOne = np.reshape(setosa[:,0],(50,1))
# # #     # axes[i].hist(classOne) #, title="Hei")
# # #     # axes[i].set_title(feature)
# # #     # disp.ax_.set_title(confVecNames[i])
# # #     # disp.im_.colorbar.remove()
    
# #     plt.subplot(2,2,i+1)

# # #     # plt.xticks(np.arange(0, len(x) + 1)[::365], x[::365])

# #     plt.hist(classOne)

# #     plt.title(feature)

# fig, axes = plt.subplots(2, 2, figsize = (8,7))
# # for i, ax in enumerate(axes.flat):#loop through every feature
# #     classOne = np.reshape(setosa[:,0],(50,1))
# #     ax.hist(classOne, label=clases[1], color=colors[1], stacked=True,alpha=0.5)

# for i,feature in enumerate(featuresTxt):
#     for j,flower in enumerate(flowersTxt):
#         axes.flat[i].hist(flowers[j][:,i], label = flower, alpha = 0.7)
#     axes.flat[i].set_title(feature)
#     axes.flat[i].set(xlabel = 'Length in cm', ylabel = 'Number of samples') 
#     # axes.flat[i].label_outer() 
    
    
# axes.flat[3].legend()
# plt.tight_layout()
    
#     # classOne = np.reshape(setosa[:,0],(50,1))
#     # axes.flat[i].hist(classOne) #, title="Hei")



# # plt.subplots_adjust(wspace=0.40)
# plt.show() #May be put outside of func and at end of script?
#___________________________________________

# match numberOfFeatures:
#     case 4:
#         print("4")
#     case 3:
#         print("3")
#     case _:        
#         print("Lavere") 

# arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# print(arr)
# arr = np.delete(arr, [1,2], 1)
# print(arr)

# 174-180_________________
# print("Trainingset:\n",train)
# print("Traininglabels:\n",trainLabels)
# print("Testset:\n",test)
# print("Testlabels:\n", testLabels)
# print("W:\n", W)
# trainedW = trainClassifier(W, numberOfIterations, alpha, train, trainLabels)
# print("trainedW:\n", trainedW)

# HitOrMissTest(trainLabels, binaryTrainedWLabels)

# iris      = load_iris()
# samples   = np.array(iris.data)
# labels    = iris.target
# labelsTxt = iris.target_names

# print(samples)



# iris = load_iris()
# samples = iris.data
# C = 3
# Nc = 50
# flowers = []
# _, axes = plt.subplots()
# for i in range(C):
#     flowers.append(samples[Nc*i:Nc*(i+1)])

# featuresTxt = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
# flowersTxt = ["Setosa", "Versicolor", "Virginica"]

# featOne = 0
# featThree = 2

# for i,flower in enumerate(flowersTxt):
#     axes.scatter(flowers[i][:,featOne],flowers[i][:,featThree], label = flower) 
# axes.set(xlabel = featuresTxt[featOne] + " in cm", ylabel = featuresTxt[featThree] + " in cm")
# axes.legend()
# # plt.show()

# print(iris.target)


# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

def sigmoidScalar(x):
    return 1/(1 + np.exp(-x))

_, ax = plt.subplots()
x = np.linspace(-10, 10, 100)
ax.plot(x, sigmoidScalar(x))
ax.grid(alpha=0.4)

ax.show()