# Basics
import numpy as np
import pandas as pd
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom
import sys
sys.path.insert(0, '../../utils')
from data_loader import DataLoader
import constants
sys.path.insert(0, '../../modeling')
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

BYTE_IDX = 0
TRAIN_CONFIG = 'D1-K1'
TEST_CONFIGS = ['D1-K6', 'D1-K0', 'D2-K0', 'D3-K0']

# Best HP from Genetic Algorithm (1 device)
HP = {
    'first_batch_norm': True,
    'second_batch_norm': False,
    'dropout_rate': 0.0,
    'l1': 0.0,
    'l2': 0.0,
    "hidden_layers": 5, 
    "hidden_neurons": 200, 
    "optimizer": "adam", 
    "learning_rate": 0.0001, 
    "batch_size": 128
}

CALLBACKS = [
    EarlyStopping(
        monitor='val_loss', 
        patience=15
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=1e-8
    )
]




def main():
    
    print('Loading Train Data...')
    train_dl = DataLoader([f'{constants.CURR_DATASETS_PATH}/SBOX_OUT/{TRAIN_CONFIG}_train.json'], BYTE_IDX)
    x_train, y_train, _, _ = train_dl.load_data()
    print('Loading Val Data...')
    val_dl = DataLoader([f'{constants.CURR_DATASETS_PATH}/SBOX_OUT/{TRAIN_CONFIG}_test.json'], BYTE_IDX)
    x_val, y_val, _, _ = val_dl.load_data()
    
    print('Loading Test Data...')
    x_test_list = []
    y_test_list = []
    for c in TEST_CONFIGS:
        test_dl = DataLoader([f'{constants.CURR_DATASETS_PATH}/SBOX_OUT/{c}_test.json'], BYTE_IDX)
        x_test, y_test, _, _ = test_dl.load_data()
        x_test_list.append((c, x_test))
        y_test_list.append(y_test)
    
    net = Network('MLP')
    net.set_hp(HP)
    net.build_model()
    model = net.model
    
    model.fit(
        x_train, 
        y_train, 
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=HP['batch_size'],
        callbacks=CALLBACKS,
        verbose=1
    )
    
    _, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f'Val accuracy: {val_acc}')
    print()
    
    
    for i, (c, x) in enumerate(x_test_list):
        y = y_test_list[i]
        _, test_acc = model.evaluate(x, y, verbose=0)
        print(f'{c} Test accuracy: {test_acc}')
        print()



if __name__ == '__main__':
    main()