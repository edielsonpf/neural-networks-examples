import numpy as np
from neural_networks.perceptron import perceptron

if __name__ == '__main__':
    
    X=np.matrix([[-1.0, 2.0,3.5],[-1.0,6.8,5.3],[-1.0,2.0,2.5],[-1.0,8.1,4.2]])
    print('\nTraining data: samples x number of features')
    print(X)
    D=np.matrix([[-1.0],[1.0],[-1.0],[1.0]])
    print('\nExpected output values')
    print(D)
    
    learning_rate=0.004
    max_epochs=100
    
    w = np.matrix([[0.84,0.68,0.88]])
    nn = perceptron(w,3,learning_rate,max_epochs)
    w,epoch = nn.train_hebbian(X,D)
    print('\nFinal synaptics weights:')
    print(w)
    print('\nFinal number of epochs:')
    print(epoch)
    
    x=np.matrix([[-1.0, 1.9,3.8]])
    y=nn.classify(x,w)
    print('\nClassification:')
    print(y)
    
    
    w = np.matrix([[0.84,0.68,0.88]])
    w,epoch,hist = nn.train_adaline(X, D, 0.001)
    print('\nFinal synaptics weights:')
    print(w)
    print('\nFinal number of epochs:')
    print(epoch)
    
    x=np.matrix([[-1.0, 1.9,3.8]])
    y=nn.classify(x,w)
    print('\nClassification:')
    print(y)