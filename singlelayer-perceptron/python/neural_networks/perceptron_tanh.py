'''
Created on 20 de ago de 2016

@author: edielson
'''
import numpy as np

class perceptron_tanh(object):
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
                
    def __tanh(self,v):
        
        return np.tanh(v)
        
    def __tanh_prime(self,v):
        
        return 1.0*(1-v)**2
    
    def __meanSquareError(self,X,D):
        error = 0
        num_samples=0
        for x,d in zip(X,D):
            v=np.dot(self.synaptic_weights,np.transpose(x))
            y=self.__tanh(v)
            delta=(d-y)**2/2.0
            error=error+delta
            num_samples=num_samples+1
        
        return (1.0*error/num_samples).item(0)
                
    def train_stochastic(self,X,D,error):
        
        meanSquareError=[]
        epoch=0
        meanSquareError.append(self.__meanSquareError(X,D))        
        
        while epoch < self.max_epoch-1:
            epoch = epoch+1
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                y=self.__tanh(v)
                delta=d-y
                deltaW=self.learning_rate*delta*self.__tanh_prime(v)*x
                self.synaptic_weights = self.synaptic_weights+deltaW
            
            meanSquareError.append(self.__meanSquareError(X, D))        
            if np.abs(meanSquareError[epoch]-meanSquareError[epoch-1]) < error: 
                print('%f < %f' %((np.abs(meanSquareError[epoch]-meanSquareError[epoch-1])),error))
                break
        
        
                   
        return self.synaptic_weights,epoch,meanSquareError
    
    def train_batch(self,X,D,error):
        
        meanSquareError=[]
        epoch=0
        meanSquareError.append(self.__meanSquareError(X,D))        
                
        while epoch < self.max_epoch-1:
            epoch = epoch+1
            deltaW=0
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                y=self.__tanh(v)
                delta=(d-y)
                deltaW=deltaW+self.learning_rate*delta*self.__tanh_prime(v)*x
        
            self.synaptic_weights = self.synaptic_weights+deltaW
            meanSquareError.append(self.__meanSquareError(X, D))        
            if np.abs(meanSquareError[epoch]-meanSquareError[epoch-1]) < error: break
            
        return self.synaptic_weights,epoch,meanSquareError
    
        
    def classify(self,x,w):
        
        if w==None:
            v=np.dot(self.synaptic_weights,np.transpose(x))
        else:
            v=np.dot(w,np.transpose(x))
        y=self.__tanh(v)
        
        if (y >=0):
            y=1
        else:
            y=-1
        
        return y