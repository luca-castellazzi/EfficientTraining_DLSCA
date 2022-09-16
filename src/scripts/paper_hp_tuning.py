# Basic
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import matplotlib
matplotlib.use('agg') # Avoid interactive mode (and save files as .PNG as default)
import matplotlib.pyplot as plt

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
BYTE_IDX = 5
TARGET = 'SBOX_OUT'
N_TOT_TRACES = 50000
TRAIN_CONFIG = ['D1-K5']
TEST_CONFIG = ['D3-K0']
METRIC = 'acc'

MTP_RES_ROOT = '/prj/side_channel/Pinata/results/MTP_Reproduction'


def build_model(layers, neurons):

    # The total amount of layers of the network is (layers + 1)

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


def hp_tuning(x_train, y_train, x_val, y_val):

    res = []
    for layers in POSSIBLE_LAYERS:
        for neurons in POSSIBLE_NEURONS:

            model = build_model(layers, neurons)
            model.fit(
                x_train, 
                y_train, 
                validation_data=(x_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            )
            _, val_acc = model.evaluate(
                x_val, 
                y_val, 
                batch_size=BATCH_SIZE, 
                verbose=0
            )

            res.append((val_acc, layers, neurons))
    
    res.sort(key=lambda x: x[0], reverse=True)
    
    return res[0]
   

def plot_ge(ge, output_path):

    """
    Plots the provided GE vector.
    
    Parameters:
        - ge (np.array):
            GE vector to plot.
        - metric (str):
            Metric used during Hyperparameter Tuning.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """
    
    ge = ge[:10]
    
    # Plot GE
    f, ax = plt.subplots(figsize=(10,5))
    
    ax.plot(ge, marker='o', color='b')
        
    ax.set_title(f'Byte: {BYTE_IDX}  |  Train: {TRAIN_CONFIG[0]}  |  Test: {TEST_CONFIG[0]}  |  Metric: {METRIC}')
    ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1))
    ax.set_ylim([-3, 50]) 
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('GE')
    ax.grid()

    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    
    plt.close(f)



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

    MTP_GE_PATH = MTP_RES_ROOT + f'/ge.npy'
    MTP_GE_PLOT_PATH = MTP_RES_ROOT + f'/ge.png'
    
    print('HP Tuning...')
    val_acc, layers, neurons = hp_tuning(x_train, y_train, x_val, y_val)
    print()
    print(f'Selected Layers:  {layers}')
    print(f'Selected Neurons: {neurons}')
    print(f'Val Acc:          {val_acc*100:.2f}%')

    final_model = build_model(layers, neurons)
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

    np.save(MTP_GE_PATH, ge)

    plot_ge(ge, MTP_GE_PLOT_PATH)
        

if __name__ == '__main__':
    main()
