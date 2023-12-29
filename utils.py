import numpy as np
import pandas as pd

class Starting_values:
    def __init__(self):
        pass

    def __call__(self, input_size: int, units_size: int):
        raise NotImplementedError("starting_values.__call__() not implemented.")

 
class Range_random(Starting_values):
    def __call__(self, input_size: int, units_size: int):
        return np.random.uniform(low=-0.5, high=0.5, size=(input_size, units_size))
    

class Fan_in(Starting_values):
    def __call__(self, input_size: int, units_size: int):
        return np.random.uniform(low=-0.5, high=0.5, size=(input_size, units_size)) * 2 / np.sqrt(input_size)


# Controllare slides NN_part_2 pag 12 per altri metodi di inizializzazione

def read_monk(train_path, test_path):
    """Read the monk's problems train and test datasets, and preprocess the input
    data with one-hot encoding.
    """

    names = ['target', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'id']
    df_tr = pd.read_csv(train_path, skipinitialspace=True, names=names, sep=' ').drop(['id'], axis=1)
    df_ts = pd.read_csv(test_path, skipinitialspace=True, names=names, sep=' ').drop(['id'], axis=1)

    y_train = df_tr['target'].values
    y_test = df_ts['target'].values

    X_train = pd.get_dummies(data=df_tr, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6']).drop(['target'], axis=1)
    X_train = X_train.values.astype(int)
    X_test = pd.get_dummies(data=df_ts, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6']).drop(['target'], axis=1)
    X_test = X_test.values.astype(int)

    X_train.shape = (len(X_train), 17, 1)
    y_train.shape = (len(y_train), 1, 1)
    X_test.shape = (len(X_test), 17, 1)
    y_test.shape = (len(y_test), 1, 1)

    return X_train, X_test, y_train, y_test
