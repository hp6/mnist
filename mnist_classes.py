import numpy as np
import pandas as pd
import joblib as jl

from sklearn.base import BaseEstimator, TransformerMixin

class MinValueCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, min_val=120, inplace=False):
        self.inplace = inplace
        self.min_val = min_val
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i, j] < self.min_val and X[i, j] != 0:
                    X[i, j] = 0
        return X