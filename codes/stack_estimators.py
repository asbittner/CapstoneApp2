import pandas as pd
import numpy as np
import pickle
import dill
from sklearn import base
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline

class_data = pd.read_csv('static/class_data.csv')
features = np.array(class_data[['Ratio Int Travelers', 'Known Unvax per 100,000', 'Population Density','Latitude','Longitude']])
target = np.array(class_data['Risk Level'])
ros = RandomOverSampler()
all_X,all_y = ros.fit_resample(features, target)

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
    
def LoadModel():
    est_1 = pickle.load(open('static/rf.sav', 'rb'))
    est_2 = pickle.load(open('static/lr.sav', 'rb'))
    model = Pipeline([('scale', StandardScaler()),('est',stack_estimators(est_1,est_2))])
    model.fit(all_X, all_y)
    return model
