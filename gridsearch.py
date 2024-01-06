import numpy as np
import itertools
from validation import kfold_crossval, create_all_net
from metrics import Metric
from losses import Loss


def grid_search(hyperparams: dict,
                X: np.ndarray,
                y: np.ndarray,
                metric: Metric,
                loss: Loss,
                k: int = 5,
                seed: int = None,
                epochs: int = 10000,
                patience: int = 250,
                verbose: bool = False
                ):
    data_to_csv = [[]]

    # Create a list of all the possible combinations using itertools.product
    hyperparams_comb = list(itertools.product(*hyperparams.values()))

    all_dics = []
    # Print the result or use it as needed
    for combination in hyperparams_comb:
        hyperparams_dict = dict(zip(hyperparams.keys(), combination))
        all_dics.append(hyperparams_dict)

    for combination in all_dics:
        layers_sizes_ = combination['layers_sizes']
        activations_ = []
        startings_ = []
        startings_range_ = []
        regularizers_ = []
        regularizers_lambda_ = []
        momentums_ = []
        etas_ = combination['etas']
        for i in range(len(combination['layers_sizes'])):
            activations_.append(combination['activations'] if i != len(layers_sizes_) - 1 else "Identity")
            startings_.append(combination['startings'])
            startings_range_.append(combination['startings_range'])
            regularizers_.append(combination['regularizers'])
            regularizers_lambda_.append(combination['regularizers_lambda'])
            momentums_.append(combination['momentums'])

        if verbose:
            print('Combination:', *combination.values())

        net = create_all_net(
            seed=seed,
            loss=str(loss),
            batch_size=1,
            input_size=len(X[0]),
            num_layer=len(layers_sizes_),
            layers_size=layers_sizes_,
            activations=activations_,
            starting=startings_,
            starting_range=startings_range_,
            regularizers=regularizers_,
            regularizers_lambda=regularizers_lambda_,
            momentums=momentums_,
            eta=etas_
        )

        statistics = kfold_crossval(
            X=X,
            y=y,
            k=k,
            net=net,
            epochs=epochs,
            patience=patience,
            seed=seed,
            metric=metric,
            verbose=verbose
        )

        if verbose:
            print('Statistics:', statistics)
            print("\n\n\n")
        data_to_csv.append(list(combination.values()) + list(statistics.values()))

    return data_to_csv
