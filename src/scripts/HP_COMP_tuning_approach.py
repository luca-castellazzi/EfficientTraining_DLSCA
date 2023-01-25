# Basic
import json
import time
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader, SplitDataLoader
import helpers
import constants
import results
import visualization as vis
sys.path.insert(0, '../modeling')
import soa
from models import mlp

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


# Constants
BYTE = 5
TARGET = 'SBOX_OUT'
GE_EXP = 100
TOT_TRACES = 50000
ATT_TRACES = 15 # Number of attack-traces to consider when plotting GEs


def main():

    _, n_devs = sys.argv
    n_devs = int(n_devs)

    if n_devs == 1:
        train_files = [f'{constants.PC_TRACES_PATH}/D1-{k}_500MHz + Resampled.trs'
                       for k in list(constants.KEYS)[1:]]
    else:
        train_files = [f'{constants.PC_TRACES_PATH}/{d}-{k}_500MHz + Resampled.trs'
                       for k in list(constants.KEYS)[1:]
                       for d in ['D1', 'D2']]

    test_file = [f'{constants.PC_TRACES_PATH}/D3-K0_500MHz + Resampled.trs']


    # Definition of constant paths
    RES_ROOT = f'{constants.RESULTS_PATH}/HPComparison/TuningApproach/{n_devs}d'
    # State-of-the-Art 
    SOA_HP = RES_ROOT + '/soa_hp.json'
    SOA_LOSS_FILE = RES_ROOT + '/soa_loss_hist.csv'
    SOA_ACC_FILE = RES_ROOT + '/soa_acc_hist.csv'
    SOA_HISTORY_PLOT = RES_ROOT + '/soa_hist.svg'
    # Custom
    CSTM_LOSS_FILE = RES_ROOT + '/genAlg_loss_hist.csv'
    CSTM_ACC_FILE = RES_ROOT + '/genAlg_acc_hist.csv'
    CSTM_HISTORY_PLOT = RES_ROOT + '/genAlg_hist.svg'
    CSTM_SAVED_MODEL_PATH = RES_ROOT + '/genAlg_model.h5'
    CSTM_HP = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{BYTE}/{n_devs}d/hp.json'
    # Both
    COMP_FILE = RES_ROOT + '/comparison.csv'
    COMP_PLOT = RES_ROOT + '/comparison.svg'


    # Get Data
    print('Collecting Data...')
    train_dl = SplitDataLoader(
        train_files,
        tot_traces=TOT_TRACES,
        train_size=0.9,
        target=TARGET,
        byte_idx=BYTE
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, _, _ = train_data
    x_val, y_val, _, _ = val_data

    test_dl = DataLoader(
        test_file,
        tot_traces=TOT_TRACES,
        target=TARGET,
        byte_idx=BYTE
    )
    x_test, _, pbs_test, tkb_test = test_dl.load()

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)



##### State-of-the-Art Approach (HP Tuning + Attack) ###########################

    print('State-of-the-Art Approach...')

    # HP Tuning and Training 
    soa_start = time.time()
    soa_hp = soa.hp_tuning(x_train, y_train, x_val, y_val)
    soa_end = time.time()
    with open(SOA_HP, 'w') as jfile:
        json.dump(soa_hp, jfile)
    soa_model = soa.build_model(soa_hp['layers'], soa_hp['neurons'])
    soa_history = soa.fit(soa_model, x_train, y_train, x_val, y_val)

    # Save Results
    soa_epochs = len(soa_history['loss'])
    # Loss
    soa_loss_data = np.vstack(
        (
            np.arange(soa_epochs)+1, # X-axis values
            soa_history['loss'], # Y-axis values for 'loss'
            soa_history['val_loss'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=soa_loss_data, 
        columns=['Epochs', 'Loss', 'Val_Loss'],
        output_path=SOA_LOSS_FILE
    )
    # Accuracy
    acc_data = np.vstack(
        (
            np.arange(soa_epochs)+1, # X-axis values
            soa_history['accuracy'], # Y-axis values for 'loss'
            soa_history['val_accuracy'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=acc_data, 
        columns=['Epochs', 'Acc', 'Val_Acc'],
        output_path=SOA_ACC_FILE
    )
    # Plot Results
    vis.plot_history(soa_history, SOA_HISTORY_PLOT)

    # GE Computation
    soa_ge = results.ge(
        model=soa_model,
        x_test=x_test,
        pltxt_bytes=pbs_test, 
        true_key_byte=tkb_test, 
        n_exp=GE_EXP, 
        target=TARGET
    )



##### Custom Approach (HP Tuning already performed via Genetic Algorithm) ######

    print('Custom Approach...')

    # Get HPs
    with open(CSTM_HP, 'r') as jfile:
        cstm_hp = json.load(jfile)
    
    # Training
    cstm_model = mlp(cstm_hp)
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
        ModelCheckpoint(
            CSTM_SAVED_MODEL_PATH,
            monitor='val_loss',
            save_best_only=True
        )
    ]
    cstm_history = cstm_model.fit(
        x_train, 
        y_train, 
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=cstm_hp['batch_size'],
        callbacks=callbacks,
        verbose=0
    ).history

    # Save Results
    actual_epochs = len(cstm_history['loss'])
    # Loss
    loss_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            cstm_history['loss'], # Y-axis values for 'loss'
            cstm_history['val_loss'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=loss_data, 
        columns=['Epochs', 'Loss', 'Val_Loss'],
        output_path=CSTM_LOSS_FILE
    )
    # Accuracy
    acc_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            cstm_history['accuracy'], # Y-axis values for 'loss'
            cstm_history['val_accuracy'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=acc_data, 
        columns=['Epochs', 'Acc', 'Val_Acc'],
        output_path=CSTM_ACC_FILE
    )
    # Plot Results
    vis.plot_history(cstm_history, CSTM_HISTORY_PLOT)
    
    # GE Computation
    cstm_attack_model = load_model(CSTM_SAVED_MODEL_PATH)
    cstm_ge = results.ge(
        model=cstm_attack_model,
        x_test=x_test,
        pltxt_bytes=pbs_test, 
        true_key_byte=tkb_test, 
        n_exp=GE_EXP, 
        target=TARGET
    )



##### Comparison ###############################################################

    print('Comparison...')
    print()

    # Save Results
    plottable_soa_ge = soa_ge[:ATT_TRACES]
    plottable_cstm_ge = cstm_ge[:ATT_TRACES]
    comp_data = np.vstack(
        (
            np.arange(ATT_TRACES)+1, # X-axis values
            plottable_soa_ge, # Y-axis values for SoA
            plottable_cstm_ge # Y-axis values for Custom
        )
    ).T
    helpers.save_csv(
        data=comp_data, 
        columns=['AttackTraces', 'SoA', 'GenAlg'],
        output_path=COMP_FILE
    )

    # Plot GE Comparison
    vis.plot_soa_vs_custom(
        soa_ge=plottable_soa_ge,
        custom_ge=plottable_cstm_ge,
        threshold=0.5,
        title=f'HP Tuning Approach: State-of-the-Art vs Genetic Algorithm  |  Train-Devices: {n_devs}',
        ylim_max=50,
        output_path=COMP_PLOT
    )



##### Print training times #####################################################

    print(f'Time to complete SoA Hyperparameter Tuning: {((soa_end - soa_start) / 60):.2f} min')


if __name__ == '__main__':
    main()