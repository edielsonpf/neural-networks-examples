'''
Created on 20 de ago de 2016

@author: edielson
'''
import numpy as np

class perceptron_sigmoid(object):
    '''
    classdocs
    '''
   
    def __init__(self,initial_weights,n_inputs,learning_rate,max_epoch):
        '''
        Constructor
        '''
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        
        if initial_weights == None:
            self.synaptic_weights = np.random.rand(n_inputs)
        else:    
            self.synaptic_weights = initial_weights
                
    def __sigmoid(self,v):
        
        return 1.0/(1-np.exp(-v))
                
    def __sigmoid_prime(self,v):
        
        return 1.0*self.__sigmoid(v)*(1-self.__sigmoid(v))
            
    
    def __meanSquareError(self,X,D):
        error = 0
        num_samples=0
        for x,d in zip(X,D):
            v=np.dot(self.synaptic_weights,np.transpose(x))
            y=self.__sigmoid(v)
            delta=(d-y)**2
            error=error+delta
            num_samples=num_samples+1
        
        return 1.0*error/num_samples
                
    def train_stochastic(self,X,D,error):
        
        meanSquareError=np.zeros(self.max_epoch)
        epoch=0
        meanSquareError[epoch] = self.__meanSquareError(X,D)        
        
        while epoch < self.max_epoch-1:
        
            epoch = epoch+1
        
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                y=self.__sigmoid(v)
                error=d-y
                delta=self.learning_rate*error*self.__sigmoid_prime(v)*x
                self.synaptic_weights = self.synaptic_weights+delta
            
            meanSquareError[epoch] = self.__meanSquareError(X, D)        
            if np.abs(meanSquareError[epoch]-meanSquareError[epoch-1]) < error: break
                   
        return self.synaptic_weights, epoch
    
    def train_batch(self,X,D,error):
        
        meanSquareError=np.zeros(self.max_epoch)
        epoch=0
        meanSquareError[epoch] = self.__meanSquareError(X,D)        
                
        while epoch < self.max_epoch-1:
            epoch = epoch+1
            deltaW=0
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                y=self.__sigmoid(v)
                delta=(d-y)
                deltaW=deltaW+self.learning_rate*delta*self.__sigmoid_prime(v)*x
        
            self.synaptic_weights = self.synaptic_weights+deltaW
            
            meanSquareError[epoch] = self.__meanSquareError(X, D)        
            if np.abs(meanSquareError[epoch]-meanSquareError[epoch-1]) < error: break
            
        return self.synaptic_weights,epoch,meanSquareError
    
        
    def classify(self,x,w):
        
        if w==None:
            v=np.dot(self.synaptic_weights,np.transpose(x))
        else:
            v=np.dot(w,np.transpose(x))
#         print('v')
#         print(v)
        y=self.__sigmoid(v)
        
        if (y >= 0.5):
            y=1
        else:
            y=-1
        
        return y