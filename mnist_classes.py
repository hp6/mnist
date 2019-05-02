import numpy as np
import pandas as pd
import joblib as jl

from sklearn.base import BaseEstimator, TransformerMixin


import mnist_functions as mf

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

class SumFtr(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    def fit(self, X, y=None):
        if not self.columns:
            self.columns = list(range(28*28))
        return self
    def transform(self, X, y=None):
        sums = np.sum(X[:, self.columns], axis=1)
        # print("sums shape =", sums.shape)
        # print("X shape =", X.shape)
        return np.append(X, sums.reshape(-1, 1), axis=1)
    
class DisconnectedSpacesCntFtr(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    def fit(self, X, y=None):
        if not self.columns:
            self.columns = list(range(28*28))
        return self
    def transform(self, X, y=None):
        disconnected_spaces_cnt = np.empty(shape=(X.shape[0], 1))
        for idx, symbol in enumerate(X[:, self.columns]):
            if idx%1000 == 0:
                print("index =", idx)
            num_of_groups, _ = mf.get_white_groups(symbol.reshape(28, 28))
            disconnected_spaces_cnt[idx, 0] = num_of_groups
        return np.append(X, disconnected_spaces_cnt, axis=1)



class ColorChangeFtr(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    def fit(self, X, y=None):
        if not self.columns:
            self.columns = list(range(28*28))
        return self
    def transform(self, X, y=None):
        changes = np.empty(shape=(X.shape[0], 28*2))
        for idx, symbol in enumerate(X[:, self.columns]):
            if idx%1000 == 0:
                print("index =", idx)
            row_color_changes, column_color_changes = mf.color_changes(symbol.reshape(28, 28))
            changes[idx, :28] = row_color_changes
            changes[idx, 28:] = column_color_changes
        return np.append(X, changes, axis=1)
