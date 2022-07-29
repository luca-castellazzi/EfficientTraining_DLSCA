# Basics
import numpy as np
from tqdm import tqdm
import random
import time
from datetime import datetime
import json
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom
import sys
sys.path.insert(0, '../../src/utils')
from data_loader import DataLoader
import constants
import visualization as vis
sys.path.insert(0, '../../src/modeling')
from hp_tuner import HPTuner
from network import Network
import evaluation as ev

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


N_MODELS = 20
BYTE_IDX = 0
GE_TEST_TRACES = 10
EPOCHS = 100
N_GE_EXP = 100
HP = {
    'first_batch_norm':  [True, False],
    'second_batch_norm': [True, False],
    'hidden_layers':     [1, 2, 3, 4, 5],
    'hidden_neurons':    [100, 200, 300, 400, 500],
    'dropout_rate':      [0.0, 0.1, 0.3, 0.5],
    'l1':                [0.0, 1e-2, 1e-3, 1e-4],
    'l2':                [0.0, 1e-2, 1e-3, 1e-4],
    'optimizer':         ['adam', 'rmsprop', 'sgd'],
    'learning_rate':     [1e-3, 1e-4, 1e-5, 1e-6],
    'batch_size':        [128, 256, 512, 1024]
}


def main():

    train_devs = sys.argv[1].upper().split(',')
    n_devs = len(train_devs)
    
    tuning_method = sys.argv[2]
    
    n_keys = int(sys.argv[3])
    
    train_configs = [f'{dev}-{k}' for k in list(constants.KEYS)[1:n_keys+1]
                     for dev in train_devs]
        
    train_paths = [constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{c}_train.json' for c in train_configs]
    #val_paths = [constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{c}_test.json' for c in train_configs]
    val_paths = [constants.CURR_DATASETS_PATH + f'/SBOX_OUT/D3-K0_test.json' for c in train_configs]

    train_dl = DataLoader(train_paths, BYTE_IDX)
    x_train, y_train, _, _ = train_dl.load_data()

    val_dl = DataLoader(val_paths, BYTE_IDX)
    x_val, y_val, _, _ = val_dl.load_data()
    
    hp_tuner = HPTuner('MLP', HP, N_MODELS, EPOCHS)
    
    if tuning_method == 'random_search':
        best_hp = hp_tuner.random_search(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val)
    elif tuning_method == 'genetic_algorithm':
        best_hp = hp_tuner.genetic_algorithm(
            n_gen=30,
            selection_perc=0.3,
            second_chance_prob=0.2,
            mutation_prob=0.2,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val)
    else:
        pass
        
    vis.plot_history(hp_tuner.best_history, constants.RESULTS_PATH + f'/{tuning_method}_history_{"".join(train_devs)}-{n_keys}.png')
    
    with open(constants.RESULTS_PATH + f'/{tuning_method}_hp_{"".join(train_devs)}-{n_keys}.json', 'w') as jfile:
        json.dump(best_hp, jfile)


if __name__ == '__main__':
    main()