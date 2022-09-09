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

def main():

    """
    Performs a complete Device-Key Trade-off Analysis (DKTA) with the specified 
    settings.
    Settings parameters (provided in order via command line):
        - n_devs: Number of train devices
        - model_type: Type of model to consider (MLP or CNN)
        - tuning_method: HP searching method (Random Search (rs) or Genetic Algorithm (ga))
        - target: Target of the attack (SBOX_IN or SBOX_OUT)
    
    All possible permutations of devices are automatically considered.
    Each DKTA consists in model-building, model-training, attack: all models 
    have the same hyperparameters, but each time an increasing number of keys 
    is considered.
    
    The results are multiple NPY files containing each average GE.
    In addition, the confusion matrices relative to each attack are considered
    alongside plots about attack-loss (both as PNG files).
    """
    
    _, n_devs, model_type, tuning_method, target = sys.argv
    n_devs = int(n_devs)
    model_type = model_type.upper()
    target = target.upper()
    
    TOT_TRAIN_TRACES = n_devs * 50000
    
    res_path = f'{constants.RESULTS_PATH}/DKTA/{target}'
    
    with open(f'{res_path}/{n_devs}d/best_hp__{tuning_method}.json', 'r') as jfile:
        hp = json.load(jfile)
    
    dev_permutations = constants.PERMUTATIONS[n_devs]
    
    
    for train_devs, test_dev in dev_permutations:
      
        test_config = f'{test_dev}-K0'
            
        ges = []
        test_losses = []
        for n_keys in range(1, len(constants.KEYS)): 
        
            n_keys_folder = f'{res_path}/{n_devs}d/{n_keys}k'
            if not os.path.exists(n_keys_folder):
                os.mkdir(n_keys_folder)
        
            print(f'===== {"".join(train_devs)}vs{test_dev} | Number of keys: {n_keys}  =====')
            
            train_configs = [f'{dev}-{k}' for k in list(constants.KEYS)[1:n_keys+1]
                             for dev in train_devs]
            train_dl = SplitDataLoader(
                train_configs, 
                n_tot_traces=TOT_TRAIN_TRACES,
                train_size=0.9,
                target=target,
                byte_idx=BYTE_IDX
            )
            train_data, val_data = train_dl.load()
            x_train, y_train, _, _ = train_data 
            x_val, y_val, _, _ = val_data # Val data is kept to consider callbacks             
            
            attack_net = Network(model_type, hp)
            attack_net.build_model()
            saved_model_path = f'{res_path}/{n_devs}d/{n_keys}k/best_model_{"".join(train_devs)}vs{test_dev}__{tuning_method}.h5'
            attack_net.add_checkpoint_callback(saved_model_path)
            train_model = attack_net.model
            
            #Training (with Validation)
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
                target=target,
                byte_idx=BYTE_IDX
            )
            x_test, y_test, pbs_test, tkb_test = test_dl.load()
            
            test_model = load_model(saved_model_path)
            preds = test_model.predict(x_test)
            
            
            # Generate info about the attack performance #########################################
            
            # Test Loss and Acc
            test_loss, test_acc = test_model.evaluate(x_test, y_test, verbose=0)
            test_losses.append(test_loss)
            print(f'Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}')
            
            # Confusion Matrix
            y_true = [el.tolist().index(1) for el in y_test]
            y_pred = [el.tolist().index(max(el)) for el in preds]
            conf_matrix = confusion_matrix(y_true, y_pred)
            vis.plot_conf_matrix(
                conf_matrix, 
                f'{res_path}/{n_devs}d/{n_keys}k/conf_matrix_{"".join(train_devs)}vs{test_dev}__{tuning_method}.png'
            )
            ###############################################################################################
            
            
            # Compute GE
            ge = attack_net.ge(
                preds=preds, 
                pltxt_bytes=pbs_test, 
                true_key_byte=tkb_test, 
                n_exp=100, 
                n_traces=10, 
                target=target
            )
            ges.append(ge)
            
            
            clear_session()
            
        
        ges = np.array(ges)
        np.save(f'{res_path}/{n_devs}d/ges_{"".join(train_devs)}vs{test_dev}__{tuning_method}.npy', ges)
        
        # Plot attack losses
        vis.plot_attack_losses(
            test_losses, 
            f'{res_path}/{n_devs}d/attack_loss_{"".join(train_devs)}vs{test_dev}__{tuning_method}.png'
        )
        
        print()
        
        
if __name__ == '__main__':
    main()
