import numpy as np
from neural_networks.perceptron import perceptron

if __name__ == '__main__':
    
    X=np.matrix([[-1.0, 2.0,3.5],[-1.0,6.8,5.3],[-1.0,2.0,2.5],[-1.0,8.1,4.2]])
    print(X)
    D=np.matrix([[-1.0],[1.0],[-1.0],[1.0]])
    print(D)
    
    learning_rate=0.7
    max_epochs=100
    
#     w=np.matrix([[0.84,0.68,0.88]])
#     
#     print(w)
#     
#     for x,d in zip(X,D):
#         print(x)
#         print(d)
#         print(np.dot(w,np.transpose(x)))

    nn = perceptron(3,learning_rate,max_epochs)
    w,epoch = nn.train(X,D)
    print(w)
    print(epoch)