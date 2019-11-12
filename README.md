# Linear  Regression from scratch

import numpy as np

class LinearRegression:
    #Data
    #Initialize data
    # compute gradient
    #update weights
    #predict
    
    def fit(self, X, y, lr=0.001, iter=1000, thres=0.001):
        #initialize weight
        self.weights = np.zeros(X.shape[1] + 1)
        #X Add bias to X
        X = self.add_bias(X)
        
        while True:
            gradient =  np.dot((y - self.predict(X, False)), X) #compute the gradient and update weight
            update = lr*gradient
            self.weights = self.weights + update
            
            if np.max(np.abs(update))< thres:break
        
    def predict(self,X, no_bias = True):
        if no_bias :
            X = self.add_bias(X)
        return np.dot(X, self.weights)
    
    def add_bias(self, X):
        return np.insert(X, 0,np.ones(X.shape[0]), axis=1)
