import numpy as np
import pandas as pd
from network import Network

class Starting_values:
    def __init__(self, low: float = -0.5, high: float = 0.5):
        self.low = low
        self.high = high

    def __call__(self, input_size: int, units_size: int):
        raise NotImplementedError("starting_values.__call__() not implemented.")

    def __str__(self) -> str:
        raise NotImplementedError("starting_values.__str__() not implemented.")


class Range_random(Starting_values):
    def __call__(self, input_size: int, units_size: int):
        return np.random.uniform(low=self.low, high=self.high, size=(input_size, units_size))

    def __str__(self) -> str:
        return f"Random uniform in range [{self.low}, {self.high}]"


class Fan_in(Starting_values):
    def __call__(self, input_size: int, units_size: int):
        return np.random.uniform(
            low=self.low,
            high=self.high,
            size=(input_size, units_size)
        ) * 2 / np.sqrt(input_size)

    def __str__(self) -> str:
        return f"Fan-in in range [{self.low}, {self.high}]"


# Controllare slides NN_part_2 pag 12 per altri metodi di inizializzazione

def read_monk(train_path, test_path):
    """Read the monk's problems training and test datasets, and preprocess the input
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


def read_cup(train_path, test_path):
    """Read the ML CUP training and blind test datasets."""

    inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    outputs = ['y1', 'y2', 'y3']
    df_tr = pd.read_csv(
        train_path,
        skipinitialspace=True,
        sep=',',
        comment='#',
        index_col=0,
        names=inputs + outputs
    )

    df_ts = pd.read_csv(
        test_path,
        skipinitialspace=True,
        sep=',',
        comment='#',
        index_col=0,
        names=inputs
    )

    X_train = df_tr[inputs].values
    test = df_ts[inputs].values

    y_train = df_tr[outputs].values

    X_train.shape = (len(X_train), 10, 1)
    y_train.shape = (len(y_train), 3, 1)
    test.shape = (len(test), 10, 1)

    return X_train, y_train, test


def print_cup_results(net: Network, blind_test, folder_path):
    outputs = []
    for x in blind_test:
        outputs.append(net.forward(x))

    index = 1
    with open(f'{folder_path}/ML-CUP23-TS.csv', 'w') as file:
        file.write('# Irene Dovichi, Marco Lavorini, Ilaria Ritelli\n')
        file.write('# BunnyProp\n')
        file.write('# ML-CUP23\n')
        file.write('# 30/01/2024\n')  # TODO Change if needed

        for output in outputs:
            line = f'{index},{output[0][0]},{output[1][0]},{output[2][0]}\n'
            file.write(line)
            index += 1
