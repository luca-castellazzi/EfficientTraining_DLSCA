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
TARGET = 'SBOX_OUT'
TOT_TRAIN_TRACES = 50000 # 50,000 for 1 dev,  100,000 for 2 devs
EPOCHS = 100

def main():

    """
    This script performs a complete Device-Key Trade-off Analysis (DKTA).
    Train-devices, test-device, hyperparameters' tuning method and DL model type 
    must be provided via command line in this order.
    
    The script generates and trains a model for each possible number of keys (1 to 10).
    Each model will have its own training, but all models will have the same hyperparameters.
    
    At the end, each GE is stored as NPY file and the confusion matrix of each attack and 
    the attack losses are plotted.
    """

    train_devs = sys.argv[1].upper().split(',')
    n_devs = len(train_devs)
    test_dev = sys.argv[2].upper()
    used_tuning_method = sys.argv[3]
    model_type = sys.argv[4].upper()
    
    with open(f'{constants.RESULTS_PATH}/DKTA/{n_devs}d/best_hp__{used_tuning_method}.json', 'r') as jfile:
        hp = json.load(jfile)
    
    test_config = f'{test_dev}-K0'
        
    ges = []
    test_losses = []
    for n_keys in range(1, len(constants.KEYS)):
    
        n_keys_folder = f'{constants.RESULTS_PATH}/DKTA/{n_devs}d/{n_keys}k'
        if not os.path.exists(n_keys_folder):
            os.mkdir(n_keys_folder)
    
        print(f'=====  Number of keys: {n_keys}  =====')
        
        train_configs = [f'{dev}-{k}' for k in list(constants.KEYS)[1:n_keys+1]
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
        x_val, y_val, _, _ = val_data # Val data is kept to consider callbacks
               
        attack_net = Network(model_type, hp)
        attack_net.build_model()
        saved_model_path = f'{constants.RESULTS_PATH}/DKTA/{n_devs}d/{n_keys}k/best_model_{"".join(train_devs)}vs{test_dev}__{used_tuning_method}.h5'
        attack_net.add_checkpoint_callback(saved_model_path)
        train_model = attack_net.model
        
        # Training (with Validation)
        train_model.fit(
            x_train, 
            y_train, 
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=attack_net.hp['batch_size'],
            callbacks=attack_net.callbacks,
            verbose=0
        )
        
        # Testing (every time consider random test traces from the same set)
        test_dl = DataLoader(
            [test_config], 
            n_tot_traces=5000,
            byte_idx=BYTE_IDX,
            target=TARGET
        )
        x_test, y_test, pbs_test, tkb_test = test_dl.load()
        
        test_model = load_model(saved_model_path)
        preds = test_model.predict(x_test)
        
        # Compute GE
        ge = attack_net.ge(
            preds=preds, 
            pltxt_bytes=pbs_test, 
            true_key_byte=tkb_test, 
            n_exp=100, 
            n_traces=10, 
            target=TARGET
        )
        ges.append(ge)
        
        
        # Generate additional info about the attack performance #########################################
        
        # Test Loss and Acc
        test_loss, test_acc = test_model.evaluate(x_test, y_test, verbose=0)
        test_losses.append(test_loss)
        print(f'Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}')
        print()
        
        # Confusion Matrix
        y_true = [el.tolist().index(1) for el in y_test]
        y_pred = [el.tolist().index(max(el)) for el in preds]
        conf_matrix = confusion_matrix(y_true, y_pred)
        vis.plot_conf_matrix(
            conf_matrix, 
            f'{constants.RESULTS_PATH}/DKTA/{n_devs}d/{n_keys}k/conf_matrix_{"".join(train_devs)}vs{test_dev}__{used_tuning_method}.png'
        )
        #################################################################################################
        
        clear_session()
    
    ges = np.array(ges)
    np.save(f'{constants.RESULTS_PATH}/DKTA/{n_devs}d/ges_{"".join(train_devs)}vs{test_dev}__{used_tuning_method}.npy', ges)
    
    # Plot attack losses
    vis.plot_attack_losses(
        test_losses, 
        f'{constants.RESULTS_PATH}/DKTA/{n_devs}d/attack_loss_{"".join(train_devs)}vs{test_dev}__{used_tuning_method}.png'
    )
        
        
if __name__ == '__main__':
    main()