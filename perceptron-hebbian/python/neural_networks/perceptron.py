'''
Created on 20 de ago de 2016

@author: edielson
'''
import numpy as np

class perceptron(object):
    '''
    classdocs
    '''
   
    def __init__(self, n_inputs, learning_rate,max_epoch):
        '''
        Constructor
        '''
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
    
        self.synaptic_weights = np.random.rand(n_inputs)
#         self.synaptic_weights = np.matrix([[0.84,0.68,0.88]])
                
    def __sign(self,v):
        
        if(v >=0):
            return 1
        else:
            return -1
            
    def train_hebbian(self,X,D):
        
        epoch=0
        errors = 1        
        
        while errors == 1 and epoch <= self.max_epoch:
            errors=0
            for x,d in zip(X,D):
                v=np.dot(self.synaptic_weights,np.transpose(x))
                y=self.__sign(v)
                error=d-y
                if (d!=y):
                    errors = 1
                    delta=self.learning_rate*error*x
                    self.synaptic_weights = self.synaptic_weights+delta
                    
            epoch = epoch+1    
        return self.synaptic_weights, epoch-1
    
    def classify(self,x):
        
        v=np.dot(self.synaptic_weights,np.transpose(x))
        y=self.__sign(v)
        return y