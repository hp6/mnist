import pandas as pd
import numpy as np

def lists_eq(list1, list2):
    return np.sum(np.sort(list1) == np.sort(list2)) == len(list1)

def unique_values(df):
    unique_values = []
    for column in df:
        unique_values.append(df[column].unique())
    return unique_values

def columns_with_missing_values(df):
    miss_val_columns = []
    for col in df:
        if df[col].isnull().values.any():
            miss_val_columns.append(col)
    return miss_val_columns

def confusion_matrix_df(matrix, columns=[]):
    if not columns:
        columns = list(range(len(matrix)))
    right_col = np.sum(matrix, axis=1)
    bottom_row = np.sum(matrix, axis=0)
    bottom_row = np.append(bottom_row, np.sum(bottom_row))
    right_col = right_col.reshape(-1, 1)
    bottom_row = bottom_row.reshape(1, -1)
    matrix = np.concatenate((matrix, right_col), axis=1)
    matrix = np.concatenate((matrix, bottom_row), axis=0)

    return pd.DataFrame(matrix, columns=columns+["sum"], index=columns+["sum"])