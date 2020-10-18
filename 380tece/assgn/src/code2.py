import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Model_2_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.drop(['date'], axis=1).to_numpy()
        
        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)
        
        X[np.where(np.isnan(X))] = 0
        
        X = (X - X_mean)/X_std
        
        X = np.append(np.ones([len(X), 1]), X, 1)
        
        return X