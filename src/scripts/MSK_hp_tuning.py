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
from data_generator import DataGenerator
from batch_scalers import BatchStandardScaler
sys.path.insert(0, '../modeling')
from models import msk_mlp, msk_cnn
from genetic_tuner import GeneticTuner

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


N_MODELS = 10
N_GEN = 10
EPOCHS = 100
HP = {
    # 'filters':       [1, 4, 8, 16, 32],
    # 'filter_size':   [3, 7, 11, 33, 101],
    'dropout_rate':  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'add_hlayers':   [1, 2, 3, 4, 5, 6, 7],
    'add_hneurons':  [50, 100, 200, 300, 400, 500], 
    'add_hl2':       [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'optimizer':     ['adam', 'rmsprop', 'sgd'],
    'learning_rate': [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'batch_size':    [128, 256, 512, 1024]
}
TARGET = 'SBOX_OUT'
CNN = False


def main():

    """
    Performs hyperparameter tuning for an MLP model (target: SBOX_OUT) with the specified settings.
    Settings parameters (provided in order via command line):
        - tot_traces: Number of total train-traces
        - b: Byte to be attacked
    
    The result is a JSON file containing the best hyperparameters.
    """
    
    _, tot_traces, val_size, b = sys.argv
    
    tot_traces = int(tot_traces)
    val_size = int(val_size)
    b = int(b)

        
    train_file = f'{constants.MSK_PC_TRACES_PATH}/second_order/D1-K1 + Resampled.trs'
    TR_LEN = 8736
    RES_ROOT = f'{constants.RESULTS_PATH}/MSK'
    
    LOSS_HIST_FILE = RES_ROOT + f'/loss_hist_data.csv'
    ACC_HIST_FILE = RES_ROOT + f'/acc_hist_data.csv'
    TOP_K_HIST_FILE = RES_ROOT + f'/topK_hist_data.csv'
    HISTORY_PLOT = RES_ROOT + f'/hp_tuning_history.svg' 
    HP_PATH = RES_ROOT + f'/hp.json'

    # Setup Generators
    batch_scaler = BatchStandardScaler(
        tr_file=train_file, 
        tr_tot=tot_traces-val_size,
        tr_len=TR_LEN,
        n_batch=100
    )
    batch_scaler.fit()

    train_indices = range(tot_traces - val_size)
    val_indices = range(tot_traces - val_size, tot_traces)

    train_gen = DataGenerator(
        tr_file=train_file,
        tr_indices=train_indices,
        tr_len=TR_LEN,
        target=TARGET,
        byte_idx=b,
        scaler=batch_scaler,
        cnn=CNN
    )

    val_gen = DataGenerator(
        tr_file=train_file,
        tr_indices=val_indices,
        tr_len=TR_LEN,
        target=TARGET,
        byte_idx=b,
        scaler=batch_scaler,
        cnn=CNN
    )
    
    # HP Tuning via Genetic Algorithm
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

    tuner = GeneticTuner(
        model_fn=msk_mlp,
        trace_len=TR_LEN,
        n_classes=constants.N_CLASSES[TARGET],
        hp_space=HP, 
        n_epochs=EPOCHS,
        pop_size=N_MODELS,
        n_gen=N_GEN,
        selection_perc=0.3,
        second_chance_prob=0.2,
        mutation_prob=0.2
    )

    metrics = [
        'accuracy',
        TopKCategoricalAccuracy(k=10, name='topK')
    ]  
        
    best_hp, best_history = tuner.tune(
        train_data=train_gen,
        val_data=val_gen,
        callbacks=callbacks,
        metrics=metrics,
        use_gen=True
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
        output_path=TOP_K_HIST_FILE
    )

    # Plot training history
    vis.plot_history(best_history, HISTORY_PLOT)
    
    with open(HP_PATH, 'w') as jfile:
        json.dump(best_hp, jfile)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')
