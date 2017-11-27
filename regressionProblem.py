import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('F:\Studying\Semester9\Machine Learning\Programming Assignment\AS.csv').dropna(subset=['DISTANCE', 'AIR_TIME'], how='any')
X = df.iloc[0:,18].values
y = df.iloc[0:,17].values

class Perceptron(object):
    def __init__(self, learningRate, numberOfIterations):
        self.learningRate = learningRate
        self.numberOfIterations = numberOfIterations

    def fit(self,X,y):
        X = np.reshape(X, (-1,1))
        print(X.shape)
        self.w_ = np.zeros(1+X.shape[1])
        print(self.w_.shape)
        self.errors = []
        meanSquareError = 0
        meanSquareErrorOverTestSet = 0
        self.meanSquareErrorOverTestSetList =[]

        for _ in range(self.numberOfIterations):
            errors = 0
            meanSquareErrorOverTestSet = 0
            for  index, (xi, target) in enumerate(zip(X,y)):
                weightChange = self.learningRate * (target - self.prediction(xi))
                self.w_[0] += weightChange * 1
                self.w_[1:] += weightChange * xi
                errors += int(weightChange != 0)
                meanSquareError += math.pow((abs(target - self.prediction(xi))), 2)/(len(X))
                if (index >= int(X.shape[0] * 0.8) and index < len(X)):
                    meanSquareErrorOverTestSet += math.pow((abs(target - self.prediction(xi))), 2)/(len(X)- int(X.shape[0] * 0.8))
            self.errors.append(errors)
            self.meanSquareErrorOverTestSetList.append(meanSquareErrorOverTestSet)
        print(meanSquareError, "Mean sqaure error calculation for all the data")
        print(meanSquareErrorOverTestSet , "Mean sqaure error for Test set (20% of the data)")
        return self

    def prediction(self, X):
        return np.dot(X,self.w_[1:])+ self.w_[0]


ppn = Perceptron(learningRate=0.0000001, numberOfIterations=10)
ppn.fit(X,y)

# plotting 80% train data with the line of best fit #
fileDivision = int(X.shape[0]*0.8)
plt.scatter(X[:fileDivision,], y[:fileDivision,],color='red', marker='x')
plt.plot(np.unique(X), np.poly1d(np.polyfit(X,y,1))(np.unique(X)))
#plt.scatter(X[fileDivision+1:,], y[fileDivision +1:,],color='blue', marker='o')
plt.xlabel('distance')
plt.ylabel('time')
plt.show()

'''
#Error
plt.plot(range(1,len(ppn.errors) +1) ,ppn.errors)
plt.xlabel('attempts')
plt.ylabel('convergance')
plt.show()

#MSE
plt.plot(range(1, len(ppn.meanSquareErrorOverTestSetList) + 1), ppn.meanSquareErrorOverTestSetList)
plt.xlabel('attempts')
plt.ylabel('convergance')
plt.show()

'''