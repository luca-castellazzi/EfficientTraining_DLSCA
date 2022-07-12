# Basics
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import time
from datetime import datetime
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
sys.path.insert(0, '../modeling')
from network import Network
import evaluation as ev
import visual as vis

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


# Global variables (constants)
BYTE_IDX = 0
TRAIN_EPOCHS = 200
GE_TEST_TRACES = 10000

TEST_CONFIG = 'D3-K0'

HP = {
    'kernel_initializer': 'glorot_uniform', # DEFAULT FOR DENSE  
    'activation':         'relu',
    'hidden_layers':      3,
    'hidden_neurons':     400,
    'dropout_rate':       0.0,
    'optimizer':          'adam',
    'learning_rate':      1e-4,
    'batch_size':         100
}


# CREATE_MODEL FUNCTION
def create_model(hp):
    
    model = Sequential()
    
    # Input
    model.add(Dense(constants.TRACE_LEN,
                    kernel_initializer=hp['kernel_initializer'],
                    activation=hp['activation']))

    # First BatchNorm
    # model.add(BatchNormalization())

    # Hidden
    for _ in range(hp['hidden_layers']):
        model.add(Dense(
            hp['hidden_neurons'], 
            kernel_initializer=hp['kernel_initializer'],
            activation=hp['activation']))

        # Dropout
        # model.add(Dropout(hp['dropout_rate']))

    # Second BatchNorm
    # model.add(BatchNormalization())

    # Output
    model.add(Dense(256, activation='softmax'))
    
    # Compilation
    lr = hp['learning_rate']
    if hp['optimizer'] == 'sgd':
        opt = SGD(learning_rate=lr)
    elif hp['optimizer'] == 'adam':
        opt = Adam(learning_rate=lr)
    else:
        opt = RMSprop(learning_rate=lr)
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    return model
    
    
# PLOT_HISTORY FUNCTION
def plot_history(history, train_dev, train_configs, i):
    f, ax = plt.subplots(2, 1, figsize=(10,12))
    
    ax[0].plot(history['loss'], label='train_loss')
    ax[0].plot(history['val_loss'], label='val_loss')
    ax[0].set_title('Train-Loss vs Val-Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    
    ax[1].plot(history['accuracy'], label='train_acc')
    ax[1].plot(history['val_accuracy'], label='val_acc')
    ax[1].set_title('Train-Acc vs Val-Acc')
    ax[1].set_ylabel('Acc')
    ax[1].set_xlabel('Epochs')
    
    first_key = train_configs[0].split("-")[1]
    last_key = train_configs[-1].split("-")[1]
    if first_key == last_key:
        keys = first_key
    else:
        keys = f'{first_key}to{last_key}'
        
    f.savefig(
        f'./train_history_{train_dev}-{keys}_{i}.png', 
        bbox_inches='tight', 
        dpi=600
    )
    
    
# PLOT_PERFORMANCE FUNCTION
def plot_performance(res):
    f, ax = f, ax = plt.subplots(figsize=(10,5))
    
    hp_values = [el[0] for el in res]
    accs = [el[1] for el in res]
    
    ax.plot(accs, marker='o')
    ax.set_title('HP Tuning Results')
    ax.set_ylabel('val_acc')
    ax.set_xlabel('HP')
    ax.set_xticks(range(len(hp_values)), labels=hp_values)
    
    f.savefig(
        f'./hp_tuning.png', 
        bbox_inches='tight', 
        dpi=600
    )

# MAIN
def main():
    
    #model_type = sys.argv[1].upper() # MLP or CNN
    tr_type = sys.argv[1].upper() # CURR or EM
    target = sys.argv[2].upper() # SBOX_OUT or HW or KEY
    train_dev = sys.argv[3].upper() # Single train device
    
    train_configs = [f'{train_dev}-{k}' for k in list(constants.KEYS)[1:]]
    assert TEST_CONFIG not in train_configs
    
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
    
    test_path = constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{TEST_CONFIG}_test.json'
    test_dl = DataLoader([test_path], BYTE_IDX)
    
    
    callbacks = []
    callbacks.append(EarlyStopping(
        monitor='val_loss', 
        patience=15)
    )

    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=1e-8)
    )
    
    
    start = time.time()
    res = []
    for i, tmp_hp in enumerate([50, 100, 400, 1000]):
           
        print(f'Trying {tmp_hp}...')
        
        HP['batch_size'] = tmp_hp

        model = create_model(HP)
        
        print('Training the model... ')
        history = model.fit(
            x_train, 
            y_train, 
            validation_data=(x_val, y_val),
            epochs=TRAIN_EPOCHS,
            batch_size=HP['batch_size'],
            callbacks=callbacks,
            verbose=0
        ).history
        print('Done.')
        
        plot_history(history, train_dev, train_configs, i)
        
        _, val_acc = model.evaluate(x_val, y_val, verbose = 0) 
        res.append((tmp_hp, val_acc))
        
        print(f'Elapsed time: {(time.time() - start)/60:.2f} minutes')
        print()
        
    plot_performance(res)
    
    print(f'Script ended in {(time.time() - start)/60:.2f} minutes')
    
    
    
if __name__ == '__main__':
    main()
    
