import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('F:\Studying\Semester9\Machine Learning\Programming Assignment\m_creditcard_24650.csv')
X = df.iloc[0:,[3,4]].values
y = df.iloc[0:,5].values
y = np.where(y == 0 ,-1,1)

class Perceptron(object):
    def __init__(self, learningRate, numberOfIterations):
        self.learningRate = learningRate
        self.numberOfIterations = numberOfIterations

    def fit(self,X,y):
        self.w_ = np.zeros(1+ X.shape[1])
        self.errors = []
        self.weight = []
        self.wnew = []

        while(True):
            if(len(self.weight)>2 and np.sum(np.abs(self.weight[len(self.weight)-2] - self.weight[len(self.weight)-1])) < 1):
                print("number of iterations is:" ,len(self.weight))
                print("The difference in the error was:" ,np.sum(np.abs(self.weight[len(self.weight)-2] - self.weight[len(self.weight)-1])))
                self.wnew = self.weight[len(self.weight) - 1]
                break
            else:
                errors = 0
                weights =0
                for xi, target in zip(X,y):
                    x0 = 1
                    weightChange =self.learningRate*(target - self.prediction(xi))
                    self.w_[1:] += weightChange * xi
                    self.w_[0] += weightChange  * x0
                    errors += int(weightChange != 0)
                self.errors.append(errors)
                weights += self.w_
                self.weight.append(weights)
                #print(self.w_)
            #print(len(self.weight))
        return self

    def prediction(self,X):
        dotProduct = np.dot(X,self.w_[1:])+ self.w_[0]
        return np.where(dotProduct >= 0.0,1,-1)


ppn = Perceptron(learningRate=0.2, numberOfIterations=6)
ppn.fit(X,y)

####################################plotting decision boundary graph####################################################
#print(ppn.shine[0] * X[0][0])
a = -ppn.wnew[1] / ppn.wnew[2]
b = - ppn.wnew[0] / ppn.wnew[2]
l = np.linspace(-10,10)
plt.gca().set_xlim([-30,40])
plt.gca().set_ylim([-5,15])
plt.plot(l,a*l+b ,'k-')

for x in X:
    xposition =int(np.where(X==x[0])[0][0])
    if(ppn.prediction([x[0] , x[1]]) == -1):
        plt.scatter(X[xposition:xposition+1, 0], X[xposition:xposition+1, 1], color='blue', marker='o', label='osama')
    else:
        plt.scatter(X[xposition:xposition+1, 0], X[xposition:xposition+1, 1], color='red', marker='x', label='ahmed')
plt.xlabel('v2')
plt.ylabel('v11')
plt.show()

'''
# Error
plt.plot(range(1,len(ppn.errors) +1) ,ppn.errors)
plt.xlabel('attempts')
plt.ylabel('convergance')
plt.show()
'''
