'''
Created on 20 de ago de 2016

@author: edielson
'''
import numpy as np

class perceptron(object):
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
                
    def __sign(self,v):
        
        if(v >=0):
            return 1
        else:
            return -1
    
    def __meanSquareError(self,X,D):
        error = 0
        num_samples=0
        for x,d in zip(X,D):
            v=np.dot(self.synaptic_weights,np.transpose(x))
            delta=(d-v)**2
            error=error+delta
            num_samples=num_samples+1
        
        return 1.0*error/num_samples
                
    def train_hebbian(self,X,D):
        
        epoch=0
        errors = 1        
        
        while errors == 1 and epoch <= self.max_epoch:
            epoch = epoch+1
            errors=0
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                y=self.__sign(v)
                error=d-y
                if (d!=y):
                    errors = 1
                    delta=self.learning_rate*error*x
                    self.synaptic_weights = self.synaptic_weights+delta
                    
        return self.synaptic_weights, epoch-1
    
    def train_adaline(self,X,D,error):
        
        meanSquareError=np.zeros(self.max_epoch)
        
        epoch=0
        meanSquareError[epoch] = self.__meanSquareError(X,D)        
        epoch=epoch+1
        
        while epoch < self.max_epoch-1:
            
            epoch=epoch+1
            
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                delta=(d-v)
                self.synaptic_weights = self.synaptic_weights+self.learning_rate*delta*x
        
            meanSquareError[epoch] = self.__meanSquareError(X, D)        
            if np.abs(meanSquareError[epoch]-meanSquareError[epoch-1]) < error: break
        
        return self.synaptic_weights,epoch-1,meanSquareError

    def train_batch(self,X,D,error):
        
        meanSquareError=np.zeros(self.max_epoch)
        
        epoch=0
        meanSquareError[epoch] = self.__meanSquareError(X,D)        
        
        while epoch <= self.max_epoch-1:
            epoch = epoch+1
            deltaW=0
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                delta=(d-v)
                deltaW=deltaW+self.learning_rate*delta*x
        
            self.synaptic_weights = self.synaptic_weights+deltaW
            meanSquareError[epoch] = self.__meanSquareError(X, D)        
            if np.abs(meanSquareError[epoch]-meanSquareError[epoch-1]) < error: break
        
        return self.synaptic_weights,epoch-1,meanSquareError
    
        
    def classify(self,x,w):
        
        if w==None:
            v=np.dot(self.synaptic_weights,np.transpose(x))
        else:
            v=np.dot(w,np.transpose(x))
        y=self.__sign(v)
        return y