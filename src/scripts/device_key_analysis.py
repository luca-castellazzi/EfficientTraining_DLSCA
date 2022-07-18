# Basics
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import time
from datetime import datetime
import json
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
import visualization as vis
sys.path.insert(0, '../modeling')
from network import Network
import evaluation as ev


# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


# Global variables (constants)
BYTE_IDX = 0
TRAIN_EPOCHS = 300
GE_TEST_TRACES = 20
N_ZEROS = 5
HP = {
    'hidden_layers':      5,
    'hidden_neurons':     400,
    'optimizer':          'adam',
    'learning_rate':      1e-5,
    'batch_size':         256
}


# RESULTS_TO_CSV
def ges_to_csv(ges, labels, csv_path):
    
    ges = np.array(ges)
    ge_dict = {f'{i+1}traces': ges[:, i] for i in range(ges.shape[1])}
    ge_dict['train_config'] = labels

    ge_df = pd.DataFrame(ge_dict)
    ge_df.to_csv(csv_path, index=False)
    

# MAIN
def main():
    
    model_type = sys.argv[1].upper() # MLP or CNN
    tr_type = sys.argv[2].upper() # CURR or EM
    target = sys.argv[3].upper() # SBOX_OUT or HW or KEY
    #n_train_devs = int(sys.argv[4]) # Number of train_devices per experiment
    train_devs_str = sys.argv[4].upper() # Comma-separated str (NO SPACES) with all the train devices (Di,Dj,Dk,...)
    train_devs = train_devs_str.split(',')
    train_devs_label = train_devs_str.replace(',', '')
    test_dev = sys.argv[5].upper() # Single test device
    n_start_configs = int(sys.argv[6]) # First config will be (K1, ..., Kn_start_configs)
    
    test_config = f'{test_dev}-K0'
    
    test_path = constants.CURR_DATASETS_PATH + f'/{target}/{test_config}_test.json'
    test_dl = DataLoader([test_path], BYTE_IDX)
    
    
    start = time.time()
    res = []
    scores = []
    for i in tqdm(range(n_start_configs, len(constants.KEYS))): 
    
        train_configs = [f'{dev}-{k}' for k in list(constants.KEYS)[1:i+1]
                         for dev in train_devs]

        if tr_type == 'CURR':
            dset_path = constants.CURR_DATASETS_PATH
        else:
            dset_path = constants.EM_DATASETS_PATH
        train_paths = [f'{dset_path}/{target}/{c}_train.json'
                       for c in train_configs]
        val_paths = [f'{dset_path}/{target}/{c}_test.json'
                       for c in train_configs]
                       
        
        train_dl = DataLoader(train_paths, BYTE_IDX)
        x_train, y_train, _, _ = train_dl.load_data()

        val_dl = DataLoader(val_paths, BYTE_IDX)
        x_val, y_val, _, _ = val_dl.load_data()
        
        
        callbacks = []
        callbacks.append(EarlyStopping(
            monitor='val_loss', 
            patience=30)
        )

        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=15,
            min_lr=1e-8)
        )
            
        net = Network(model_type)
        net.set_hp(HP)
        net.build_model()
        model = net.model
        
        #print('Training the model... ')
        history = model.fit(
            x_train, 
            y_train, 
            validation_data=(x_val, y_val),
            epochs=TRAIN_EPOCHS,
            batch_size=HP['batch_size'],
            callbacks=callbacks,
            verbose=0
        ).history
        #print('Done.')
        
        
        ge = ev.guessing_entropy(
            model,
            test_dl=test_dl,
            n_exp=100,
            n_traces=GE_TEST_TRACES
        )

        score = ev.ge_score(ge, N_ZEROS)
        #print(f'Attack Score: {score}')
        
        
        start_key = train_configs[0].split("-")[1]
        end_key = train_configs[-1].split("-")[1]
        if start_key == end_key:
            config_label = f'{train_devs_label}_{start_key}'
        else:
            config_label = f'{train_devs_label}_{start_key}-{end_key.replace("K", "")}'
        
        # Plot history
        #h_output_file = constants.RESULTS_PATH + f'/plots/training_{config_label}.png'
        #vis.plot_history(history, h_output_file)
        
        res.append((config_label, ge, score))
        
        #print(f'Elapsed time: {(time.time() - start)/60:.2f} minutes')
        #print()
        
        
    config_labels = [el[0] for el in res]
    ges = [el[1] for el in res]
    scores = [el[2] for el in res]
    
    # Save GEs as .CSV
    ge_file_path = constants.RESULTS_PATH + f'/{len(train_devs)}d/ge-{train_devs_label}vs{test_dev}.csv'
    ges_to_csv(ges, config_labels, ge_file_path)
    
    # Plot GEs
    ge_plot_path = constants.RESULTS_PATH + f'/{len(train_devs)}d/ge-{train_devs_label}vs{test_dev}.png'
    ge_title = f'GE {train_devs_label}vs{test_dev}'
    vis.plot_ges(ges, GE_TEST_TRACES, config_labels, ge_title, ge_plot_path)
    
    # Plot scores
    scores_dict = {config_labels[i].split('_')[1]: scores[i] for i in range(len(res))}
    s_title = f'GE {train_devs_label}vs{test_dev}'
    s_plot_path = constants.RESULTS_PATH + f'/{len(train_devs)}d/scores-{train_devs_label}vs{test_dev}.png'
    vis.plot_scores(scores_dict, s_title, s_plot_path)
    
    
    print(f'Script ended in {(time.time() - start)/60:.2f} minutes')
    
    
    
if __name__ == '__main__':
    main()