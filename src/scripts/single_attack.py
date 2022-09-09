# Basics
import numpy as np
import random
import json
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader, SplitDataLoader
import constants
import visualization as vis
sys.path.insert(0, '../modeling')
from hp_tuner import HPTuner
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

BYTE_IDX = 0
EPOCHS = 100

TARGET = 'SBOX_OUT'
N_DEVS = 1
TRAIN_DEV = 'D1'
TEST_CONFIG = 'D3-K0'


def main():

    res_path = f'{constants.RESULTS_PATH}/DKTA/{TARGET}'
    
    with open(f'{res_path}/{N_DEVS}d/best_hp__ga.json', 'r') as jfile:
        hp = json.load(jfile)

    train_configs = [f'{TRAIN_DEV}-{k}' for k in list(constants.KEYS)[1:]]
    TOT_TRAIN_TRACES = N_DEVS * 50000

    train_dl = SplitDataLoader(
        train_configs,
        n_tot_traces=TOT_TRAIN_TRACES,
        train_size=0.9,
        target=TARGET,
        byte_idx=BYTE_IDX
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, _, _ = train_data
    x_val, y_val, _, _ = val_data # Val data is kept to consider callbacks

    net = Network('MLP', hp)
    net.build_model()
    model = net.model
    saved_model_path = f'./single_attack_models/best_model.h5'
    net.add_checkpoint_callback(saved_model_path)

    # Train ####################################################################
    model.fit(
        x_train, 
        y_train, 
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=net.hp['batch_size'],
        callbacks=net.callbacks,
        verbose=1
    )

    
    # Test #####################################################################
    test_dl = DataLoader(
        [TEST_CONFIG],
        n_tot_traces=5000,
        target=TARGET,
        byte_idx=BYTE_IDX
    )
    x_test, y_test, pbs_test, tkb_test = test_dl.load()

    test_model = load_model(saved_model_path)
    preds = test_model.predict(x_test)

    test_loss, test_acc = test_model.evaluate(x_test, y_test, verbose=0)

    
    # GE #######################################################################
    ge = net.ge(
        preds=preds, 
        pltxt_bytes=pbs_test, 
        true_key_byte=tkb_test, 
        n_exp=100, 
        n_traces=10, 
        target=TARGET
    )

    print()
    print(ge)


if __name__ == '__main__':
    main()
