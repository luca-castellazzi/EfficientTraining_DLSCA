# Basics
import numpy as np
from tqdm import tqdm
import random
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

BYTE_IDX = 0

def main():

    train_devs = sys.argv[1].upper().split(',')
    test_dev = sys.argv[2].upper()
    used_tuning_method = sys.argv[3]
    model_type = sys.argv[4].upper()
    
    with open(constants.RESULTS_PATH + f'/single_dev-config_tuning/{used_tuning_method}_hp_{"".join(train_devs)}.json', 'r') as jfile:
        hp = json.load(jfile)
    
    test_config = f'{test_dev}-K0'
    test_path = constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{test_config}_test.json'
    test_dl = DataLoader([test_path], BYTE_IDX)
    x_test, y_test, _, _ = test_dl.load_data()
        
    ges = []
    for j in range(1, len(constants.KEYS)):
    
        print(f'=====  Number of keys: {j}  =====')
        
        train_configs = [f'{dev}-{k}' for k in list(constants.KEYS)[1:j+1]
                         for dev in train_devs]
        train_paths = [constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{c}_train.json' for c in train_configs]
        val_paths = [constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{c}_test.json' for c in train_configs] 
                         
        train_dl = DataLoader(train_paths, BYTE_IDX)
        x_train, y_train, _, _ = train_dl.load_data()

        val_dl = DataLoader(val_paths, BYTE_IDX)
        x_val, y_val, _, _ = val_dl.load_data()
                                 
        attack_net = Network(model_type)
        attack_net.set_hp(hp)
        attack_net.build_model()
        model = attack_net.model
        model.fit(
            np.concatenate((x_train, x_val), axis=0),
            np.concatenate((y_train, y_val), axis=0),
            epochs=100,
            batch_size=hp['batch_size'],
            verbose=0
        )
        
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f'Test Accuracy: {test_acc:.4f}')
        print()
        
        #ge = ev.guessing_entropy(
        #    model,
        #    test_dl=test_dl,
        #    n_exp=100,
        #    n_traces=10
        #)
        
        ge = ev.NEW_guessing_entropy(
            model,
            test_dl=test_dl,
            n_exp=100,
            n_traces=10
        )

        ges.append(ge)
        
        clear_session()
    
    ges = np.array(ges)
    np.save(constants.RESULTS_PATH + f'/single_dev-config_tuning/{used_tuning_method}_ges.npy', ges)
        
        
if __name__ == '__main__':
    main()