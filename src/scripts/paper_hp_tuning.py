# Basic
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import L1, L2, L1L2

# Custom
import sys
sys.path.insert(0, '../utils')
import aes
import constants
import results
from data_loader import DataLoader, SplitDataLoader

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


POSSIBLE_LAYERS = [1, 2, 3, 4]
POSSIBLE_NEURONS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
EPOCHS = 50
BATCH_SIZE = 256
BYTE_IDX = 0
TARGET = 'SBOX_OUT'
N_TOT_TRACES = 50000
TRAIN_CONFIG = ['D2-K6']
TEST_CONFIG = ['D3-K0']
METRIC = 'loss'


def build_model(layers, neurons):

    model = Sequential()

    model.add(
        Dense(neurons, activation='relu', input_shape=(constants.TRACE_LEN,))
    )

    for _ in range(layers - 1):
        model.add(
            Dense(neurons, activation='relu')
        )

    model.add(
        Dense(256, activation='softmax')
    )

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def hp_tuning(metric, x_train, y_train, x_val, y_val):

    res = []
    for layers in POSSIBLE_LAYERS:
        for neurons in POSSIBLE_NEURONS:

            print(f'::::::::::  layers = {layers}  |  neurons = {neurons}  ::::::::::')

            model = build_model(layers, neurons)
            model.fit(
                x_train, 
                y_train, 
                validation_data=(x_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            )
            val_loss, val_acc = model.evaluate(
                x_val, 
                y_val, 
                batch_size=BATCH_SIZE, 
                verbose=0
            )

            hps = (layers, neurons)
            
            if metric == 'loss':
                el = (val_loss, hps)
                reverse = False
            else:
                el = (val_acc, hps)
                reverse = True
            res.append(el)
    
    res.sort(key=lambda x: x[0], reverse=reverse)
    
    return res[0][1]
   

def main():

    train_dl = SplitDataLoader(
        TRAIN_CONFIG,
        n_tot_traces=N_TOT_TRACES,
        train_size=0.9,
        target=TARGET,
        byte_idx=BYTE_IDX
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, _, _ = train_data
    x_val, y_val, _, _ = val_data
    
    test_dl = DataLoader(
        TEST_CONFIG,
        n_tot_traces=5000,
        target=TARGET,
        byte_idx=BYTE_IDX
    )
    x_test, y_test, pbs_test, tkb_test = test_dl.load()

    print('HP Tuning...')
    n_layers, n_neurons = hp_tuning(METRIC, x_train, y_train, x_val, y_val)
    print('Finished HP Tuning')
    print(f'Selected Number of Layers: {n_layers}')
    print(f'Selected Number of Neurons:{n_neurons}')
    print()

    print('Final Training')
    final_model = build_model(n_layers, n_neurons)
    final_model.fit(
        x_train, 
        y_train, 
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    preds = final_model.predict(x_test)

    ge = results.ge(
        preds=preds, 
        pltxt_bytes=pbs_test, 
        true_key_byte=tkb_test, 
        n_exp=100, 
        target=TARGET,
        n_traces=100 # Default: 500
    )

    np.save('./paper_hp_tuning_ge.npy', ge)

    print()
    print('GE:')
    print(ge)


if __name__ == '__main__':
    main()
