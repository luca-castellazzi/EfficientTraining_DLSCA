# Basics
import numpy as np
import random
import json
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import polars as pl
from tqdm import tqdm

import matplotlib.pyplot as plt

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader, SplitDataLoader
import constants
import results
import visualization as vis
sys.path.insert(0, '../modeling')
from hp_tuner import HPTuner
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


EPOCHS = 100
TARGET = 'SBOX_OUT'
N_DEVS = 1
TEST_CONFIG = 'D3-K0'
BYTES = [1, 2, 3, 4, 5, 14]


def main():
    
    k0 = np.array([int(kb, 16) for kb in constants.KEYS['K0']])

    TOT_TRAIN_TRACES = N_DEVS * 50000

    for b in BYTES:
        
        print(f'::::: Byte {b} :::::')

        RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{b}/{N_DEVS}d'
        HP_PATH = RES_ROOT + f'/hp.json'
        SAVED_MODEL_PATH = f'./single_attack_models/model_b{b}.h5'
    
        if sys.argv[1] == 'train':

            with open(HP_PATH, 'r') as jfile:
                hp = json.load(jfile)

            train_configs = [f'D1-{k}' for k in list(constants.KEYS)[1:]]

            train_dl = SplitDataLoader(
                train_configs,
                n_tot_traces=TOT_TRAIN_TRACES,
                train_size=0.9,
                target=TARGET,
                byte_idx=b
            )
            train_data, val_data = train_dl.load()
            x_train, y_train, _, _ = train_data
            x_val, y_val, _, _ = val_data # Val data is kept to consider callbacks

            net = Network('MLP', hp)
            net.build_model()
            net.add_checkpoint_callback(SAVED_MODEL_PATH)

            # Train ####################################################################
            net.model.fit(
                x_train, 
                y_train, 
                validation_data=(x_val, y_val),
                epochs=EPOCHS,
                batch_size=net.hp['batch_size'],
                callbacks=net.callbacks,
                verbose=0
            )

        
        # Test #####################################################################
        test_model = load_model(SAVED_MODEL_PATH)

        test_dl = DataLoader(
            [TEST_CONFIG],
            n_tot_traces=5000,
            target=TARGET,
            byte_idx=b
        )
        x_test, y_test, pbs_test, tkb_test = test_dl.load()

        preds = test_model.predict(x_test)
            
        predicted_key_bytes = results.retrieve_key_byte(
            preds=preds, 
            pltxt_bytes=pbs_test,  
            target=TARGET,
            n_traces=10
        )

        predicted_key_bytes = np.array(predicted_key_bytes)

        print(f'Prediction: {predicted_key_bytes}')
        print(f'Correct key-byte: {k0[b]}')
        print()
    

if __name__ == '__main__':
    main()
