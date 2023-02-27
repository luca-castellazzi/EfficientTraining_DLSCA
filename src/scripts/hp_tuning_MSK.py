# Basics
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom
import sys
sys.path.insert(0, '../utils')
import helpers
import constants
import visualization as vis
from data_loader import SplitDataLoader
sys.path.insert(0, '../modeling')
from models import msk_mlp, msk_cnn
from genetic_tuner import GeneticTuner

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


N_MODELS = 15
N_GEN = 20
EPOCHS = 100
HP = {
    'dropout_rate':  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'add_hlayers':   [1, 2, 3, 4, 5],
    'add_hneurons':  [100, 200, 300, 400, 500], 
    'add_hl2':       [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'optimizer':     ['adam', 'rmsprop', 'sgd'],
    'learning_rate': [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'batch_size':    [128, 256, 512, 1024]
}
TARGET = 'SBOX_OUT'


def main():

    """
    Performs hyperparameter tuning for an MLP model (target: SBOX_OUT) with the specified settings.
    Settings parameters (provided in order via command line):
        - train_devs: Devices to use during training, provided as comma-separated string without spaces
        - tot_traces: Number of total train-traces
        - val_traces: Number of traces to use as validation set
        - b: Byte to be attacked
    
    The result is a JSON file containing the best hyperparameters.
    """
    
    _, train_devs, tot_traces, val_traces, b = sys.argv
    
    train_devs = train_devs.upper().split(',')
    n_devs = len(train_devs)
    tot_traces = int(tot_traces)
    val_traces = int(val_traces)
    b = int(b)

    train_files = [f'{constants.MSK_PC_TRACES_PATH}/second_order/{dev}-{k} + Resampled.trs' 
                    for k in list(constants.KEYS)[1:]
                    for dev in train_devs]
    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{b}/msk_{n_devs}d'
    
    LOSS_HIST_FILE = RES_ROOT + f'/loss_hist_data.csv'
    ACC_HIST_FILE = RES_ROOT + f'/acc_hist_data.csv'
    TOPK_HIST_FILE = RES_ROOT + f'/topK_hist_data.csv' 
    HISTORY_PLOT = RES_ROOT + f'/hp_tuning_history.svg' 
    HP_PATH = RES_ROOT + f'/hp.json'

    # Get data
    train_dl = SplitDataLoader(
        train_files, 
        tot_traces=tot_traces,
        train_size=1-(tot_traces/val_traces),
        target=TARGET,
        byte_idx=b
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, _, _ = train_data
    x_val, y_val, _, _ = val_data

    # Scale data to 0-mean and 1-variance
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    # HP Tuning via Genetic Algorithm
    tuner = GeneticTuner(
        model_fn=msk_mlp,
        trace_len=x_train.shape[1],
        n_classes=y_train.shape[1],
        hp_space=HP, 
        n_epochs=EPOCHS,
        pop_size=N_MODELS,
        n_gen=N_GEN,
        selection_perc=0.3,
        second_chance_prob=0.2,
        mutation_prob=0.2
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=15
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-7),
    ]

    metrics = [
        'accuracy',
        TopKCategoricalAccuracy(k=10, name='topK')
    ]  

    best_hp, best_history = tuner.tune(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        callbacks=callbacks,
        metrics=metrics
    )

    
    # Save history data to .CSV files
    actual_epochs = len(best_history['loss']) # Early Stopping can make the actual 
                                            # number of epochs different from the original one

    # Loss
    loss_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            best_history['loss'], # Y-axis values for 'loss'
            best_history['val_loss'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=loss_data, 
        columns=['Epochs', 'Loss', 'Val_Loss'],
        output_path=LOSS_HIST_FILE
    )

    # Accuracy
    acc_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            best_history['accuracy'], # Y-axis values for 'loss'
            best_history['val_accuracy'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=acc_data, 
        columns=['Epochs', 'Acc', 'Val_Acc'],
        output_path=ACC_HIST_FILE
    )

    # Top K (k=10) Accuracy
    topK_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            best_history['topK'], # Y-axis values for 'loss'
            best_history['val_topK'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=topK_data, 
        columns=['Epochs', 'TopK', 'Val_TopK'],
        output_path=TOPK_HIST_FILE
    )

    # Plot training history
    vis.plot_history(best_history, HISTORY_PLOT)
    
    with open(HP_PATH, 'w') as jfile:
        json.dump(best_hp, jfile)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')
