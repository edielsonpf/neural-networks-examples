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
        
        if initial_weights is None:
            self.synaptic_weights = np.random.rand(n_inputs+1)
        else:    
            self.synaptic_weights = initial_weights
        
        print('\nInicial synaptic weights:')
        print(self.synaptic_weights)    
                
    def __sign(self,v):
        
        if(v >=0):
            return 1
        else:
            return -1
    
    def __meanSquareError(self,X,D):
        error = 0
        num_samples=0
        for x,d in zip(X,D):
            x=np.concatenate(([-1], x),axis=0)
            v=np.dot(self.synaptic_weights,np.transpose(x))
            delta=(d-v)**2
            error=error+delta
            num_samples=num_samples+1
        
        return np.asscalar(1.0*error/num_samples)
                
    def train_hebbian(self,X,D):
        
        epoch=0
        errors = 1        
        
        while errors == 1 and epoch <= self.max_epoch:
            errors=0
            for x,d in zip(X,D):
                x=np.concatenate(([-1], x),axis=0)
                v=np.dot(self.synaptic_weights,np.transpose(x))
                y=self.__sign(v)
                error=d-y
                if (d!=y):
                    errors = 1
                    delta=self.learning_rate*error*x
                    self.synaptic_weights = self.synaptic_weights+delta
            epoch = epoch+1
                    
        return self.synaptic_weights, epoch
    
    def train_adaline(self,X,D,error):
        
        meanSquareError=[]
        epoch=0
        
        meanSquareError.append(self.__meanSquareError(X,D))        
        epoch=epoch+1
        
        while epoch < self.max_epoch-1:
            
            for x,d in zip(X,D):
                x=np.concatenate(([-1], x),axis=0)
                v=np.dot(self.synaptic_weights,np.transpose(x))
                delta=(d-v)
                self.synaptic_weights = self.synaptic_weights+self.learning_rate*delta*x
        
            meanSquareError.append(self.__meanSquareError(X, D))        
            epoch=epoch+1
            if np.abs(meanSquareError[-1]-meanSquareError[-2]) < error: break
        
        return self.synaptic_weights,epoch,meanSquareError

    def train_batch(self,X,D,error):
        
        meanSquareError=[]
        epoch=0
        
        meanSquareError.append(self.__meanSquareError(X,D))
        epoch=epoch+1        
        
        while epoch <= self.max_epoch-1:
            
            deltaW=0
            for x,d in zip(X,D):
                x=np.concatenate(([-1], x),axis=0)
                v=np.dot(self.synaptic_weights,np.transpose(x))
                delta=(d-v)
                deltaW=deltaW+self.learning_rate*delta*x
        
            self.synaptic_weights = self.synaptic_weights+deltaW
            meanSquareError.append(self.__meanSquareError(X, D))
            epoch = epoch+1
                    
            if np.abs(meanSquareError[-1]-meanSquareError[-2]) < error: break
        
        return self.synaptic_weights,epoch,meanSquareError
    
        
    def classify(self,x,w):
        x=np.concatenate(([-1], x),axis=0)
        if w is None:
            v=np.dot(self.synaptic_weights,np.transpose(x))
        else:
            v=np.dot(w,np.transpose(x))
        y=self.__sign(v)
        return y