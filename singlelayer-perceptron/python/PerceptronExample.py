import numpy as np
from neural_networks.perceptron import perceptron

if __name__ == '__main__':
    
    X=np.array([[2.0,3.5],[6.8,5.3],[2.0,2.5],[8.1,4.2]])
    print('\nTraining data: samples x number of features')
    print(X)
    D=np.array([[-1.0],[1.0],[-1.0],[1.0]])
    print('\nExpected output values')
    print(D)
    
    
    print('########################################')
    print('Example using perceptron with Hebian training')
    
    learning_rate=0.004
    max_epochs=100
    
    #w = np.matrix([[0.84,0.68,0.88]])
    w=None
    nn = perceptron(w,2,learning_rate,max_epochs)
    w,epoch = nn.train_hebbian(X,D)
    print('\nFinal synaptics weights:')
    print(w)
    print('\nFinal number of epochs:')
    print(epoch)
    
    x=np.array([1.9,3.8])
    y=nn.classify(x,w)
    print('\nClassification:')
    print(y)
    
    print('########################################')
    print('Same example using perceptron withh Adaline training')
    #w = np.matrix([[0.84,0.68,0.88]])
    w=None
    w,epoch,hist = nn.train_adaline(X, D, 0.001)
    print('\nFinal synaptics weights:')
    print(w)
    print('\nFinal number of epochs:')
    print(epoch)
    
    x=np.array([1.9,3.8])
    y=nn.classify(x,w)
    print('\nClassification:')
    print(y)