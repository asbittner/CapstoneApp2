import pandas as pd
import numpy as np
import pickle
import dill
from sklearn import base

class stack_estimators(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, estimator1, estimator2):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        
    def fit(self,X,y):
        X2, y2 = ros.fit_resample(X, y) #balance
        self.estimator1 = self.estimator1.fit(X2, y2)
        
        X3 = np.vstack((X.T, self.estimator1.predict(X))).T
        X4, y4 = ros.fit_resample(X3,y)
        
        self.estimator2 = self.estimator2.fit(X4,y4)
        
        return self
    
    def predict(self, X):
        X2 = np.vstack((X.T, self.estimator1.predict(X))).T
        return self.estimator2.predict(X2)
