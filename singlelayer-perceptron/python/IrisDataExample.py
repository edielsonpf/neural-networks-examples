# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 16:03:56 2017

@author: edielson
"""

import numpy as np
import pandas as pd
from neural_networks.perceptron import perceptron
import matplotlib.pyplot as plt


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df)

# setosa and versicolor
Y = df.iloc[0:100, 4].values
print('Expected output:')
for y in Y:
    print(y)

Y = np.where(Y == 'Iris-setosa', -1, 1)
print('Converted expected output:')
for y in Y:
    print(y)

## sepal length and petal length
X = df.iloc[0:100, [0,1,2,3]].values
for x in X:
    print(x)


print('Separating data in training and testing set')

Xtrain = np.concatenate((X[0:36], X[50:86]),axis=0)
print('Training data size: '),
print(len(Xtrain))
print(Xtrain)

Ytrain =  np.concatenate((Y[0:36], Y[50:86]),axis=0)
print('Expected output data size: '),
print(len(Ytrain))

Xtest = np.concatenate((X[36:50],X[86:100]),axis=0)
print('Testing data size: '),
print(len(Xtest))

Ytest = np.concatenate((Y[36:50],Y[86:100]),axis=0)
print('Testing data size: '),
print(len(Ytest))

print('Training the perceptron')
    
learning_rate=0.0001
max_epochs=1000
limiar = 0.0001

nn = perceptron(None,4,learning_rate,max_epochs)
w,epoch,hist = nn.train_adaline(X,Y,limiar)
print('Numero de epocas: %s'%epoch)
plt.plot(hist)
plt.show()

print('Testing the perceptron')
accuracy=0.0
for x,d in zip(Xtest,Ytest):
    y=nn.classify(x,None)
    if(y==d):
        accuracy=accuracy+1
        
print 'Precision: %s' %((accuracy/len(Ytest))*100.0)     
    