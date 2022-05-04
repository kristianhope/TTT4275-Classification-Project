from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Scatter plot
def scatter(featOne, featTwo):
    iris = load_iris()
    samples = iris.data
    C = 3
    Nc = 50
    flowers = []
    _, axes = plt.subplots()
    for i in range(C):
        flowers.append(samples[Nc*i:Nc*(i+1)])

    featuresTxt = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    flowersTxt = ["Setosa", "Versicolor", "Virginica"]

    for i,flower in enumerate(flowersTxt):
        axes.scatter(flowers[i][:,featOne],flowers[i][:,featTwo], label = flower) 
    axes.set(xlabel = featuresTxt[featOne] + " in cm", ylabel = featuresTxt[featTwo] + " in cm")
    axes.legend()
    plt.show()


# Sigmoid function
def sigmoidScalar(x):
    return 1/(1 + np.exp(-x))

# Sigmoid function plot 
def plotSigmoid ():
    _, ax = plt.subplots()
    x = np.linspace(-10, 10, 100)
    ax.plot(x, sigmoidScalar(x))
    ax.grid(alpha=0.4)
    plt.show()


scatter(featOne=0,featTwo=2)
plotSigmoid()