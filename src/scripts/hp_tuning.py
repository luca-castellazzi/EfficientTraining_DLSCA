# Basics
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

# Custom
import sys
sys.path.insert(0, '../utils')
import helpers
import constants
import visualization as vis
from data_loader import SplitDataLoader
sys.path.insert(0, '../modeling')
from hp_tuner import HPTuner

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


N_MODELS = 15
N_GEN = 20
EPOCHS = 100
HP = {
    'hidden_layers':  [1, 2, 3, 4, 5],
    'hidden_neurons': [100, 200, 300, 400, 500],
    'dropout_rate':   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
    'l2':             [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'optimizer':      ['adam', 'rmsprop', 'sgd'],
    'learning_rate':  [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'batch_size':     [128, 256, 512, 1024]
}
TARGET = 'SBOX_OUT'


def main():

    """
    Performs hyperparameter tuning for an MLP model (target: SBOX_OUT) with the specified settings.
    Settings parameters (provided in order via command line):
        - train_devs: Devices to use during training, provided as comma-separated string without spaces
        - tot_traces: Number of total train-traces
        - b: Byte to be attacked
    
    The result is a JSON file containing the best hyperparameters.
    """
    
    _, train_devs, tot_traces, b = sys.argv
    
    train_devs = train_devs.upper().split(',')
    n_devs = len(train_devs)
    tot_traces = int(tot_traces)
    b = int(b)

        
    train_files = [f'{constants.PC_TRACES_PATH}/{dev}-{k}_500MHz + Resampled.trs' 
                   for k in list(constants.KEYS)[1:]
                   for dev in train_devs]
    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{b}/{n_devs}d'
    
    LOSS_HIST_FILE = RES_ROOT + f'/loss_hist_data.csv'
    ACC_HIST_FILE = RES_ROOT + f'/acc_hist_data.csv'
    HISTORY_PLOT = RES_ROOT + f'/hp_tuning_history.svg' 
    HP_PATH = RES_ROOT + f'/hp.json'

    # Get data
    train_dl = SplitDataLoader(
        train_files, 
        tot_traces=tot_traces,
        train_size=0.9,
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
    hp_tuner = HPTuner(
        model_type='MLP', 
        hp_space=HP, 
        n_models=N_MODELS, 
        n_epochs=EPOCHS
    )
        
    best_hp = hp_tuner.genetic_algorithm(
        n_gen=N_GEN,
        selection_perc=0.3,
        second_chance_prob=0.2,
        mutation_prob=0.2,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )
        
    
    # Save history data to .CSV files
    b_history = hp_tuner.best_history
    actual_epochs = len(b_history['loss']) # Early Stopping can make the actual 
                                            # number of epochs different from the original one

    # Loss
    loss_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            b_history['loss'], # Y-axis values for 'loss'
            b_history['val_loss'] # Y-axis values for 'val_loss'
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
            b_history['accuracy'], # Y-axis values for 'loss'
            b_history['val_accuracy'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=acc_data, 
        columns=['Epochs', 'Acc', 'Val_Acc'],
        output_path=ACC_HIST_FILE
    )

    # Plot training history
    vis.plot_history(b_history, HISTORY_PLOT)
    
    with open(HP_PATH, 'w') as jfile:
        json.dump(best_hp, jfile)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')
