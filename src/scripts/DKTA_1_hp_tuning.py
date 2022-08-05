# Basics
import numpy as np
import random
import json
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import SplitDataLoader
import constants
import visualization as vis
sys.path.insert(0, '../modeling')
from hp_tuner import HPTuner
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


N_MODELS = 15#20
BYTE_IDX = 0
TARGET = 'SBOX_OUT'
TOT_TRAIN_TRACES = 100000
EPOCHS = 100
HP = {
    # 'first_batch_norm':  [True, False],
    # 'second_batch_norm': [True, False],
    'hidden_layers':     [1, 2, 3, 4, 5],
    'hidden_neurons':    [100, 200, 300, 400, 500],
    'dropout_rate':      [0.0, 0.1, 0.2,  0.3, 0.4, 0.5],
    # 'l1':                [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'l2':                [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'optimizer':         ['adam', 'rmsprop', 'sgd'],
    'learning_rate':     [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'batch_size':        [128, 256, 512, 1024]
}


def main():

    """
    This script performs hyperparameter tuning considering the specified devices as train-devices.
    Train-devices and tuning method must be provided via command line in this order.
    
    Hyperparameter tuning is run considering all possible keys.
    """

    train_devs = sys.argv[1].upper().split(',')
    n_devs = len(train_devs)
    
    model_type = sys.argv[2]
    
    tuning_method = sys.argv[3] # rs (Random Search) or ga (Genetic Algorithm)
    
    metric = sys.argv[4]
    
    train_configs = [f'{dev}-{k}' for k in constants.KEYS
                     for dev in train_devs]
                     
    train_dl = SplitDataLoader(
        train_configs, 
        n_tot_traces=TOT_TRAIN_TRACES,
        train_size=0.9,
        byte_idx=BYTE_IDX,
        target=TARGET
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, _, _ = train_data
    x_val, y_val, _, _ = val_data
    
    hp_tuner = HPTuner(
        model_type=model_type, 
        hp_space=HP, 
        n_models=N_MODELS, 
        n_epochs=EPOCHS,
        metric=metric
    )
    
    if tuning_method == 'rs':
        best_hp = hp_tuner.random_search(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val
        )
    elif tuning_method == 'ga':
        best_hp = hp_tuner.genetic_algorithm(
            n_gen=20, #30,
            selection_perc=0.3,
            second_chance_prob=0.2,
            mutation_prob=0.2,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val
        )
    else:
        pass
        
    vis.plot_history(hp_tuner.best_history, f'{constants.RESULTS_PATH}/DKTA/best_history_{"".join(train_devs)}__{tuning_method}.png')
    
    with open(f'{constants.RESULTS_PATH}/DKTA/{n_devs}d/best_hp__{tuning_method}.json', 'w') as jfile:
        json.dump(best_hp, jfile)


if __name__ == '__main__':
    main()
